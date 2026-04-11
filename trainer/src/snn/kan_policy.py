"""
KAN Policy Layer - Passo 8 do Projecto SNN Organismo.

Camada de política estratégica baseada em Kolmogorov-Arnold Networks (KAN).
Modula os biases da SNN reactiva a cada N=10 steps do mundo com base no
estado global do organismo.

Topologia KAN
-------------
  Input  6 nós:
    0  energia_actual  (normalizada)
    1  energia_max     (normalizada, tipicamente 1.0)
    2  tempo_de_vida   (normalizado)
    3  comida_vista_últimos_50_steps  (normalizada 0-1)
    4  distância_média_comida_vista   (normalizada 0-1)
    5  steps_sem_comer                (normalizado)

  Hidden 8 nós:  funções B-spline treináveis (order k=3, grid_size=5)

  Output 2 nós:
    0  bias_explorar   [0,1]  — aumenta tendência para explorar/virar
    1  bias_exploitar  [0,1]  — aumenta tendência para seguir comida conhecida

A saída é passada por sigmoid para garantir o intervalo [0,1].

Comunicação com a SNN
---------------------
  A cada N=10 steps:
    bias_mod = kan.forward(estado_global)   # shape [2]
    snn.bias_explorar  = bias_mod[0] * 0.3
    snn.bias_exploitar = bias_mod[1] * 0.3

Inicialização com directivas comportamentais
--------------------------------------------
  1. 'Não morrer com energia alta'
     spline energia → exploitar: slope negativo (energia baixa → mais exploitar)
  2. 'Procurar comida quando não a vê há muito tempo'
     spline steps_sem_comer → explorar: slope positivo
  3. Outros splines: inicialização próxima de zero

Notas de implementação
----------------------
  A API do pykan (MultKAN) utiliza:
    kan.act_fun[layer_idx]         →  KANLayer
    kan.act_fun[layer_idx].coef    →  Parameter de shape [in_dim, out_dim, G+k]
  onde G = grid_size = 5 e k = ordem = 3, logo n_coef = G+k = 8.

  Para KAN(width=[6, 8, 2], grid=5, k=3):
    Layer 0:  in_dim=6,  out_dim=8,  coef.shape=[6, 8, 8]
    Layer 1:  in_dim=8,  out_dim=2,  coef.shape=[8, 2, 8]
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .network import OrganismoSNN

# ---------------------------------------------------------------------------
# Importação condicional do pykan
# ---------------------------------------------------------------------------
try:
    from kan import KAN as _KAN
    _KAN_AVAILABLE = True
    _KAN_IMPORT_ERROR: str = ""
except ImportError as _exc:
    _KAN_AVAILABLE = False
    _KAN_IMPORT_ERROR = str(_exc)

# ---------------------------------------------------------------------------
# Hiperparâmetros (conforme especificação §8)
# ---------------------------------------------------------------------------
LR_KAN         = 5e-4   # taxa de aprendizagem da KAN
N_UPDATE_STEPS = 10     # actualizar a cada N steps do mundo

# Dimensões da topologia KAN
KAN_WIDTH = [6, 8, 2]   # [input, hidden, output]
KAN_GRID  = 5            # número de intervalos da grelha B-spline
KAN_ORDER = 3            # ordem do polinómio por troços (B-spline order k)

# n_coef = KAN_GRID + KAN_ORDER = 8  (coeficientes por spline)


# ---------------------------------------------------------------------------
# Inicialização com directivas comportamentais
# ---------------------------------------------------------------------------

def _inicializar_kan_com_directivas(kan: Any) -> None:
    """
    Inicializa os coeficientes B-spline da KAN com directivas comportamentais.

    Mapeamento spec → API pykan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    spec pseudocódigo::

        kan.layers[0].splines[0].coefs = ...    # input 0, todos os outputs
        kan.layers[0].splines[5].coefs = ...    # input 5, todos os outputs

    API real (MultKAN + KANLayer)::

        kan.act_fun[0].coef            # shape [in_dim, out_dim, G+k]
        kan.act_fun[0].coef[0]         # shape [out_dim, G+k]  → input 0
        kan.act_fun[0].coef[5]         # shape [out_dim, G+k]  → input 5

    Directivas
    ~~~~~~~~~~
    1. Energia (input 0) → Exploitar:  slope negativo
       Coeficientes decrescentes: quando energia é alta, output deve ser baixo.
       Os 5 valores do spec [-2, -1, 0, 1, 2] descrevem slope positivo (erro de
       sinal na especificação); a directiva comportamental é slope negativo,
       portanto usamos linspace(+2, -2, n_coef).

    2. Steps sem comer (input 5) → Explorar:  slope positivo
       Coeficientes crescentes: quanto mais tempo sem comer, mais explorar.
       Consistente com spec: [0.0, 0.5, 1.0, 1.5, 2.0].

    3. Outros inputs: inicialização nula (eliminar ruído inicial).

    Parâmetros
    ----------
    kan : MultKAN (KAN do pykan)
        Rede KAN já instanciada.
    """
    with torch.no_grad():
        layer0 = kan.act_fun[0]   # KANLayer: in_dim=6, out_dim=8
        n_out   = layer0.coef.shape[1]   # out_dim = 8
        n_coef  = layer0.coef.shape[2]   # G + k = 8

        # --- Directiva 1: energia (input 0) → slope negativo ---------------
        # Valores decrescentes: energia alta → output baixo (→ menos exploitar)
        coef_neg = torch.linspace(2.0, -2.0, n_coef, device=layer0.coef.device)
        # Aplicar a TODOS os outputs do input 0 (shape [n_out, n_coef])
        layer0.coef.data[0] = coef_neg.unsqueeze(0).expand(n_out, -1).clone()

        # --- Directiva 2: steps_sem_comer (input 5) → slope positivo -------
        # Valores crescentes: longo tempo sem comer → output alto (→ mais explorar)
        coef_pos = torch.linspace(0.0, 2.0, n_coef, device=layer0.coef.device)
        layer0.coef.data[5] = coef_pos.unsqueeze(0).expand(n_out, -1).clone()

        # --- Outros inputs: próximos de zero --------------------------------
        for inp_idx in [1, 2, 3, 4]:
            layer0.coef.data[inp_idx].zero_()

        # --- Camada de saída: inicialização quase nula ---------------------
        # Permite que a aprendizagem parta de uma baseline neutra
        layer1 = kan.act_fun[1]   # KANLayer: in_dim=8, out_dim=2
        layer1.coef.data.zero_()


# ---------------------------------------------------------------------------
# Função de estado ao nível do módulo (convenção do projecto)
# ---------------------------------------------------------------------------

def observacao_para_estado_kan(
    obs: Dict[str, Any],
    stats: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Converte observação e estatísticas do organismo num tensor [1, 6].

    Delega para :meth:`KANPolicy.estado_para_tensor`.

    Parâmetros
    ----------
    obs : dict
        Observação do mundo; precisa de 'energia_actual', 'energia_max',
        opcionalmente 'tempo_vida_norm'.
    stats : dict
        Estatísticas acumuladas: 'comida_vista_50steps',
        'distancia_media_vista', 'steps_sem_comer'.
    device : torch.device, opcional
        Dispositivo do tensor de saída.

    Devolve
    -------
    torch.Tensor de shape [1, 6], dtype=float32.
    """
    return KANPolicy.estado_para_tensor(obs, stats, device=device)


# ---------------------------------------------------------------------------
# Classe principal KANPolicy
# ---------------------------------------------------------------------------

class KANPolicy:
    """
    Camada de política estratégica baseada em KAN (Passo 8).

    Recebe o estado global do organismo a cada N=10 steps e injesta
    modulações de bias na SNN reactiva.

    Atributos de classe
    -------------------
    lr : float
        Taxa de aprendizagem do optimizador Adam da KAN (5e-4).

    Parâmetros
    ----------
    device : torch.device, opcional
        Dispositivo de computação; por defeito CPU.

    Exemplos
    --------
    >>> policy = KANPolicy(device=torch.device('cpu'))
    >>> obs   = {'energia_actual': 80.0, 'energia_max': 100.0,
    ...          'tempo_vida_norm': 0.3}
    >>> stats = {'comida_vista_50steps': 5, 'distancia_media_vista': 0.4,
    ...          'steps_sem_comer': 12}
    >>> estado = KANPolicy.estado_para_tensor(obs, stats)
    >>> bias = policy.forward(estado)   # torch.Tensor([bias_explorar, bias_exploitar])
    """

    lr: float = LR_KAN

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if not _KAN_AVAILABLE:
            raise ImportError(
                "pykan não está disponível: " + _KAN_IMPORT_ERROR + ". "
                "Instalar com: pip install pykan"
            )

        self.device = device or torch.device("cpu")
        self._step_counter: int = 0

        # MultKAN aceita device como string ('cpu', 'cuda:0', …)
        device_str = str(self.device)

        # Criar a KAN com auto_save=False para evitar I/O de checkpoints
        self.kan = _KAN(
            width=KAN_WIDTH,
            grid=KAN_GRID,
            k=KAN_ORDER,
            auto_save=False,
            device=device_str,
        )
        self.kan.to(self.device)

        # Inicializar com directivas comportamentais
        try:
            _inicializar_kan_com_directivas(self.kan)
        except Exception as exc:
            warnings.warn(
                f"Inicialização directiva da KAN falhou ({exc}). "
                "A usar inicialização aleatória padrão do pykan.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Optimizador Adam com lr=5e-4 conforme especificação
        self.optimizador = torch.optim.Adam(
            self.kan.parameters(), lr=self.lr
        )

    # ------------------------------------------------------------------
    def forward(self, estado: torch.Tensor) -> torch.Tensor:
        """
        Executa a passagem forward da KAN.

        Parâmetros
        ----------
        estado : torch.Tensor de shape [1, 6]
            Estado global normalizado do organismo.

        Devolve
        -------
        torch.Tensor de shape [2]
            [bias_explorar, bias_exploitar] comprimido a [0,1] por sigmoid.
        """
        estado = estado.to(self.device)
        saida = self.kan(estado)                       # [1, 2]
        return torch.sigmoid(saida).squeeze(0)         # [2]

    # ------------------------------------------------------------------
    def actualizar_snn(
        self,
        snn: "OrganismoSNN",
        estado: torch.Tensor,
    ) -> None:
        """
        Injesta os biases da KAN na SNN reactiva.

        Incrementa o contador interno e executa a KAN apenas quando o
        contador atinge um múltiplo de N=10.

        Regra de actualização (conforme spec §8):
            snn.bias_explorar  += bias_mod[0] * 0.3
            snn.bias_exploitar += bias_mod[1] * 0.3

        Parâmetros
        ----------
        snn : OrganismoSNN
            Rede SNN reactiva; deve expor 'bias_explorar' e 'bias_exploitar'.
        estado : torch.Tensor de shape [1, 6]
            Estado calculado por :meth:`estado_para_tensor`.
        """
        self._step_counter += 1
        if self._step_counter % N_UPDATE_STEPS != 0:
            return

        with torch.no_grad():
            bias_mod = self.forward(estado)   # [2], valores em [0,1]

        snn.bias_explorar  = bias_mod[0].item() * 0.3
        snn.bias_exploitar = bias_mod[1].item() * 0.3

    # ------------------------------------------------------------------
    @staticmethod
    def estado_para_tensor(
        obs: Dict[str, Any],
        stats: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Converte a observação e estatísticas do organismo num tensor [1, 6].

        Mapeamento dos 6 inputs da KAN:
          0  energia_actual  / energia_max          →  [0, 1]
          1  energia_max norm (sempre 1.0)           →  1.0
          2  tempo_de_vida normalizado               →  [0, 1]
          3  comida_vista_50steps / 20               →  [0, 1]
          4  distancia_media_vista  (já normalizada) →  [0, 1]
          5  steps_sem_comer / 50                    →  [0, 1]

        Parâmetros
        ----------
        obs : dict
            Observação do mundo. Chaves usadas:
            - 'energia_actual'  (float)
            - 'energia_max'     (float)
            - 'energia_norm'    (float, alternativa calculada; tem precedência)
            - 'tempo_vida_norm' (float, opcional; default 0.0)
        stats : dict
            Estatísticas acumuladas. Chaves usadas:
            - 'comida_vista_50steps'   (int ou float)
            - 'distancia_media_vista'  (float, esperado em [0,1])
            - 'steps_sem_comer'        (int ou float)
        device : torch.device, opcional
            Dispositivo do tensor de saída.

        Devolve
        -------
        torch.Tensor de shape [1, 6], dtype=float32.
        """
        dev = device or torch.device("cpu")

        # --- Energia ---------------------------------------------------------
        energia_max    = float(obs.get("energia_max", 1.0))
        energia_actual = float(obs.get("energia_actual", 0.0))
        # Se 'energia_norm' já foi calculada pelo ambiente, usá-la directamente
        energia_norm   = float(
            obs.get("energia_norm", energia_actual / max(energia_max, 1e-6))
        )
        energia_norm   = max(0.0, min(energia_norm, 1.0))

        # energia_max normalizada: por definição é sempre 1.0
        energia_max_norm = 1.0

        # --- Tempo de vida ---------------------------------------------------
        tempo_vida = float(obs.get("tempo_vida_norm", 0.0))
        tempo_vida = max(0.0, min(tempo_vida, 1.0))

        # --- Estatísticas ----------------------------------------------------
        comida_vista_50 = float(stats.get("comida_vista_50steps", 0.0))
        distancia_media = float(stats.get("distancia_media_vista", 0.0))
        steps_sem_comer = float(stats.get("steps_sem_comer", 0.0))

        # Normalizar para [0, 1]
        # comida_vista_50: típico 0-20 ocorrências por janela de 50 steps
        comida_norm    = min(comida_vista_50 / 20.0, 1.0)
        # distancia_media: já normalizada pelo mundo [0, 1]
        distancia_norm = max(0.0, min(distancia_media, 1.0))
        # steps_sem_comer: até ~50 steps sem comer é uma emergência
        sem_comer_norm = min(steps_sem_comer / 50.0, 1.0)

        vetor = [
            energia_norm,       # 0: energia_actual normalizada
            energia_max_norm,   # 1: energia_max normalizada (≡ 1.0)
            tempo_vida,         # 2: tempo_de_vida normalizado
            comida_norm,        # 3: comida_vista_últimos_50_steps
            distancia_norm,     # 4: distância_média_comida_vista
            sem_comer_norm,     # 5: steps_sem_comer normalizado
        ]

        return torch.tensor([vetor], dtype=torch.float32, device=dev)   # [1, 6]

    # ------------------------------------------------------------------
    def reset_contador(self) -> None:
        """Reinicia o contador de steps. Útil ao começar um novo episódio."""
        self._step_counter = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"KANPolicy(width={KAN_WIDTH}, grid={KAN_GRID}, k={KAN_ORDER}, "
            f"lr={self.lr}, N={N_UPDATE_STEPS}, device={self.device})"
        )
