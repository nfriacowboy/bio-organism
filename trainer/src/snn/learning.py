"""
Aprendizagem R-STDP (Reward-modulated STDP) com NACA + AGMP + Experience Replay.

Passo 7: adiciona controlo neuromodulador (NACA), modulação glial (AGMP) e replay.
         - NACAGate: modula o delta_w com novidade × |recompensa|.
         - AGMPGate: controla se ocorre aprendizagem (plasticidade) ou inferência pura.
         - ExperienceReplay: buffer offline para consolidação entre episódios.

Passo 6 (mantido): R-STDP base sem NACA nem AGMP — valida que o organismo aprende.

A eligibility trace é calculada por correlação pré-pós spike dentro de cada step
(T=8 timesteps). A actualização de pesos ocorre no fim de cada step do mundo,
modulada pelo sinal de recompensa e pelo gate NACA.

Critério Passo 7: após ~50 episódios a novidade (AGMPGate) cai abaixo do threshold
e o organismo estabiliza em modo 'inferencia'.
"""
from __future__ import annotations

import math
import random
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .network import OrganismoSNN, N_H1, N_H2, N_INPUT, N_OUTPUT, T_STEPS


# ---------------------------------------------------------------------------
# Dataclasses de configuração
# ---------------------------------------------------------------------------

@dataclass
class ConfigRSTDP:
    lr_snn: float = 5e-3            # taxa de aprendizagem base (aumentado 1e-3→5e-3 v9)
    tau_trace: float = 20.0         # constante de tempo da eligibility trace (ms ~ timesteps)
    tau_trace_fc3: float = 20.0     # v39: igual a tau_trace — trace longo (revert v38). Eq balanced @p_av=0.33 c/ turn_bonus=0.30
    clip_w: float = 1.0             # clipping máximo dos pesos
    grad_clip: float = 0.5          # clipping do delta_w por step


@dataclass
class ConfigNACA:
    """
    Configuração do NACA Gate.

    lambda_naca: taxa de esquecimento adaptativo quando o gate está fechado
                 (pouca novidade). Empurra os pesos inativos em direcção a zero.
    """
    lambda_naca: float = 0.000001   # 1e-6: ~10% decay total em 500ep (108k steps)


@dataclass
class ConfigAGMP:
    """
    Configuração do AGMP Gate.

    tau_slow  : constante de tempo da integração astroglial (passos de mundo).
                Valores maiores tornam a variável mais lenta a convergir.
    threshold : limiar de novidade (em taxa de spikes) para activar plasticidade.
                Quando novidade < threshold o organismo entra em modo inferência.
    """
    tau_slow: float = 20000.0
    threshold: float | None = None   # None = calibração automática em ep20 (v16)


# ---------------------------------------------------------------------------
# EligibilityTrace (inalterado de Passo 6)
# ---------------------------------------------------------------------------

class EligibilityTrace:
    """
    Mantém a eligibility trace de uma ligação (pre → post).
    trace = correlação temporal filtrada entre spikes pre e post.
    """

    def __init__(self, shape: tuple[int, int], tau: float = 20.0, device: torch.device | None = None):
        self.tau = tau
        self.device = device or torch.device("cpu")
        self.trace = torch.zeros(shape, device=self.device)
        self._decay: float = math.exp(-1.0 / self.tau)  # precomputed — avoids tensor alloc every step

    def actualizar(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """
        pre_spikes:  [T, batch, n_pre]
        post_spikes: [T, batch, n_post]
        Correlação 0-lag somada ao longo dos T timesteps.
        trace[post, pre] += sum_t(post[t] ⊗ pre[t]) / T
        """
        # Médias sobre batch e T para obter correlação [n_post, n_pre]
        pre_mean  = pre_spikes.mean(dim=(0, 1))    # [n_pre]
        post_mean = post_spikes.mean(dim=(0, 1))   # [n_post]
        corr = torch.outer(post_mean, pre_mean)    # [n_post, n_pre]
        # Decaimento exponencial + acumulação
        self.trace = self._decay * self.trace + corr

    def reset(self) -> None:
        self.trace.zero_()


# ---------------------------------------------------------------------------
# NACAGate — Neuromodulation-Astrocyte-Controlled Activation
# ---------------------------------------------------------------------------

class NACAGate:
    """
    Gate neuromodulador que escala o delta_w em função da novidade e da recompensa.

    O sinal neuromodulador (neuromod_signal) é:
        neuromod_signal = novidade × |recompensa|

    Quando o sinal é alto (o organismo encontra algo novo e relevante):
      - O update de pesos é aplicado com força total.
    Quando o sinal é baixo (padrões já conhecidos ou recompensa fraca):
      - O update é atenuado e é aplicada regularização L2 suave (esquecimento
        adaptativo), evitando que conexões irrelevantes acumulem peso.

    Relação com Passo 7: depois de ~50 episódios a novidade cai e o NACA gate
    atenua progressivamente os updates, estabilizando os pesos aprendidos.
    """

    def __init__(self, lambda_naca: float = 0.01):
        self.lambda_naca = lambda_naca
        self.neuromod_signal: float = 0.0

    def update(self, novidade: float, recompensa: float) -> float:
        """
        Calcula e armazena o sinal neuromodulador.
        Deve ser chamado antes de apply() em cada step.
        """
        self.neuromod_signal = novidade * abs(recompensa)
        return self.neuromod_signal

    def apply(self, delta_w: torch.Tensor, weights_antigos: torch.Tensor) -> torch.Tensor:
        """
        Aplica o gate NACA ao delta_w:
          delta_w_modulado = gate × delta_w - lambda_naca × (1 - gate) × W

        Quando gate ≈ 1: update forte, sem penalização.
        Quando gate ≈ 0: update fraco + regularização L2 suave nos pesos.
        """
        gate = self.neuromod_signal
        return gate * delta_w - self.lambda_naca * (1.0 - gate) * weights_antigos


# ---------------------------------------------------------------------------
# AGMPGate — Astrocyte-Gated Memory/Plasticity
# ---------------------------------------------------------------------------

class AGMPGate:
    """
    Gate glial que controla se o organismo está em modo aprendizagem ou inferência.

    Mantém uma variável lenta (astrocyte_var) que rastreia a taxa de spikes
    esperada com constante de tempo tau_slow. A novidade é o desvio absoluto
    entre a taxa actual e a esperada (pré-passo).

    Critério Passo 7:
      - Início: astrocyte_var ≈ 0, taxas de spikes variadas → novidade alta →
        plasticidade_activa = True → organismo aprende.
      - Após ~50 episódios: astrocyte_var converge para a taxa real →
        novidade < threshold → plasticidade_activa = False → modo inferência.

    O atributo público novidade guarda a última novidade calculada, usada
    externamente pelo NACAGate.
    """

    def __init__(self, tau_slow: float = 20000.0, threshold: float | None = None):
        self.astrocyte_var: float = 0.0
        self.tau_slow = tau_slow
        self.threshold = threshold          # None = ainda não calibrado
        self.novidade: float = 0.0          # última novidade (acesso externo)
        # Fase de observação (v16): medir novelty real antes de activar threshold
        self.episodio_actual: int = 0
        self.N_OBSERVACAO: int = 99999      # v43: desactivar AGMP freeze — nunca calibra durante treino (1500ep)
        self.novelty_hist: list[float] = [] # acumular novelty passo-a-passo

    def step(self, spike_rate_actual: float, spike_rate_esperado: float) -> bool:
        """
        Actualiza a variável astroglial e determina o modo.

        spike_rate_actual   : taxa média de spikes neste step.
        spike_rate_esperado : expectativa anterior (tipicamente astrocyte_var
                              capturado antes da chamada).

        Devolve True  → plasticidade activa (aprendizagem).
                False → plasticidade inactiva (inferência).
        """
        self.astrocyte_var += (spike_rate_actual - self.astrocyte_var) / self.tau_slow
        self.novidade = abs(spike_rate_actual - spike_rate_esperado)
        # Fase de observação: forçar aprendizagem e acumular novelty para calibração
        if self.episodio_actual < self.N_OBSERVACAO:
            # N_OBSERVACAO=99999 é sentinel de "nunca calibrar" — não acumular lista infinita
            if self.N_OBSERVACAO < 99999:
                self.novelty_hist.append(self.novidade)
            return True
        # Threshold ainda não calibrado (v43: N_OBSERVACAO=99999 → nunca chega aqui durante treino)
        if self.threshold is None:
            return True
        return self.novidade > self.threshold

    def fim_episodio(self, episodio_num: int) -> dict | None:
        """
        Chamar no fim de cada episódio.

        No ep N_OBSERVACAO (100 em v28): calibra o threshold como percentil 75
        da novelty observada durante a fase de observação.

        P75 → o organismo aprende nos 25% de steps mais 'surpreendentes',
        evitando saturação (100% como v14) e extinção (0% como v15).

        Devolve dict com métricas de calibração no ep100, None nos outros eps.
        """
        self.episodio_actual = episodio_num
        if episodio_num == self.N_OBSERVACAO and self.threshold is None:
            import statistics
            hist = self.novelty_hist
            nov_media = statistics.mean(hist)
            nov_p75   = sorted(hist)[int(len(hist) * 0.75)]
            self.threshold = nov_p75
            print(
                f"[AGMP] Threshold calibrado: {self.threshold:.5f}"
                f"  (media={nov_media:.5f}, p75={nov_p75:.5f},"
                f" n={len(hist)} amostras)"
            )
            return {
                "agmp_threshold_calibrado": self.threshold,
                "novelty_media_obs":        nov_media,
                "n_amostras_obs":           len(hist),
            }
        return None


# ---------------------------------------------------------------------------
# ExperienceReplay — consolidação offline ("fase de sono")
# ---------------------------------------------------------------------------

class ExperienceReplay:
    """
    Buffer circular de experiências para consolidação entre episódios.

    Cada experiência é um tuplo:
      (estado, accao, recompensa, proximo_estado, spike_patterns)

    O método consolidar() amostra aleatoriamente do buffer e chama
    snn.replay_update() para reforçar padrões de spikes passados.
    Deve ser invocado externamente entre episódios (fase de "sono").
    """

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def store(
        self,
        estado: Any,
        accao: Any,
        recompensa: float,
        proximo_estado: Any,
        spike_patterns: Any,
    ) -> None:
        """Guarda uma experiência no buffer circular."""
        self.buffer.append((estado, accao, recompensa, proximo_estado, spike_patterns))

    def consolidar(self, snn: OrganismoSNN, batch_size: int = 32) -> None:
        """
        Fase de consolidação: replays um mini-batch de experiências passadas
        e reforça os padrões de spikes via snn.replay_update().

        Deve ser chamada entre episódios, nunca durante o loop de step().
        Não tem efeito se o buffer estiver vazio.
        """
        if not self.buffer:
            return
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        for estado, accao, recompensa, _, spike_patterns in batch:
            snn.replay_update(spike_patterns, recompensa)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# RSTDPLearner — actualizado com NACA + AGMP + Replay (Passo 7)
# ---------------------------------------------------------------------------

class RSTDPLearner:
    """
    Aprende os pesos da SNN via R-STDP modulado por NACA + AGMP.

    Mantém eligibility traces para as 3 camadas de pesos:
      fc1 (input→H1), fc2 (H1→H2), fc3 (H2→output)

    Pipeline por step do mundo:
      1. Forward pass → captura spikes por camada.
      2. Actualiza eligibility traces (correlação pré-pós).
      3. AGMPGate.step() → decide se plasticidade está activa.
      4. NACAGate.update() → calcula sinal neuromodulador (novidade × |reward|).
      5. _actualizar_pesos() com NACA.apply() — apenas se AGMP activo.
      6. Guarda experiência no ExperienceReplay.

    Modo inferência (propriedade .modo):
      Quando AGMPGate.novidade < threshold, o organismo entra em modo
      'inferencia' (updates de pesos suspensos). Isso ocorre após ~50 episódios
      quando os padrões de spikes estabilizam e a novidade cai.

    Retrocompatibilidade:
      - cfg, device mantêm a mesma interface de Passo 6.
      - naca_cfg, agmp_cfg e replay_capacity são keyword-only e opcionais.
      - step() aceita argumentos opcionais (estado, accao, proximo_estado) para
        o replay; código existente que chama step(x, r) não precisa de alteração.
    """

    def __init__(
        self,
        snn: OrganismoSNN,
        cfg: ConfigRSTDP | None = None,
        device: torch.device | None = None,
        *,
        naca_cfg: ConfigNACA | None = None,
        agmp_cfg: ConfigAGMP | None = None,
        replay_capacity: int = 10000,
    ):
        self.snn = snn
        self.cfg = cfg or ConfigRSTDP()
        self.device = device or torch.device("cpu")

        # --- Passo 7: NACA + AGMP ---
        _naca = naca_cfg or ConfigNACA()
        _agmp = agmp_cfg or ConfigAGMP()
        self.naca = NACAGate(lambda_naca=_naca.lambda_naca)
        self.agmp = AGMPGate(tau_slow=_agmp.tau_slow, threshold=_agmp.threshold)

        # --- Passo 7: Experience Replay ---
        self.replay = ExperienceReplay(capacity=replay_capacity)

        # Estado interno de plasticidade (True = aprendizagem, False = inferência)
        self._plasticidade_activa: bool = True
        # v4.0: gate externo — controlado pelo trainer com base em CV de recompensa.
        # Quando True, suspende actualizações de pesos independentemente do AGMP.
        self._inferencia_forcada: bool = False

        # --- Eligibility traces para cada camada ---
        self.et_fc1 = EligibilityTrace((N_H1, N_INPUT),   self.cfg.tau_trace,     self.device)
        self.et_fc2 = EligibilityTrace((N_H2, N_H1),      self.cfg.tau_trace,     self.device)
        self.et_fc3 = EligibilityTrace((N_OUTPUT, N_H2),  self.cfg.tau_trace_fc3, self.device)  # v39: tau_trace_fc3=tau_trace=20

        # Pré-alocado: evita torch.zeros() por step em step() e step_with_precomputed_spikes()
        self._fake_post: torch.Tensor = torch.zeros(T_STEPS, 1, N_OUTPUT, device=self.device)

        # Recompensa acumulada (média exponencial para suavizar)
        self._reward_ema: float = 0.0
        self._reward_tau: float = 10.0

        # Baseline global de recompensa: EMA lenta cross-episódio (não reiniciada).
        # Subtrai o offset sistemático (step cost + energia) do sinal de aprendizagem,
        # implementando o erro de predição de recompensa (RPE) dopaminérgico.
        # Com tau≈10000 steps e step_cost≈-0.031/step, converge em ~50 episódios.
        self._reward_baseline: float = 0.0

        # Spikes capturados pelos hooks no último forward
        self._last_input: torch.Tensor | None = None
        self._last_h1:    torch.Tensor | None = None
        self._last_h2:    torch.Tensor | None = None
        self._last_out:   torch.Tensor | None = None

        # Base lr guardada para warmup (imutável após init)
        self._lr_snn_base: float = self.cfg.lr_snn

        self._instalar_hooks()

    # -------------------------------------------------------------------
    @property
    def modo(self) -> str:
        """
        Modo actual do organismo:
          'aprendizagem' — AGMPGate activa, pesos a ser actualizados.
          'inferencia'   — AGMPGate inactiva, pesos congelados.
        """
        return "aprendizagem" if self._plasticidade_activa else "inferencia"

    def set_modo_externo(self, inferencia: bool) -> None:
        """
        Gate de plasticidade externo (v4.0).
        Chamado pelo trainer com base no CV da recompensa por episódio.

        Quando inferencia=True: R-STDP suspenso (pesos congelados).
        AGMP e NACA continuam a correr para rastrear taxa de spikes.
        """
        self._inferencia_forcada = inferencia

    # -------------------------------------------------------------------
    def _instalar_hooks(self) -> None:
        """Hooks que gravam spikes de cada camada durante o forward."""

        def hook_h1(module, input, output):
            if not hasattr(self, "_h1_buffer"):
                self._h1_buffer = []
            self._h1_buffer.append(output.detach().unsqueeze(0))  # [1, batch, N_H1]

        def hook_h2(module, input, output):
            if not hasattr(self, "_h2_buffer"):
                self._h2_buffer = []
            self._h2_buffer.append(output.detach().unsqueeze(0))

        def hook_out(module, input, output):
            if not hasattr(self, "_out_buffer"):
                self._out_buffer = []
            self._out_buffer.append(output.detach().unsqueeze(0))

        self.snn.lif1.register_forward_hook(hook_h1)
        self.snn.lif2.register_forward_hook(hook_h2)
        self.snn.lif3.register_forward_hook(hook_out)

    # -------------------------------------------------------------------
    def _colectar_spikes(self, x_input: torch.Tensor) -> None:
        """Chama o forward e recolhe os buffers de spikes."""
        self._h1_buffer = []
        self._h2_buffer = []
        self._out_buffer = []
        _ = self.snn(x_input)   # popula os buffers via hooks

        T = len(self._h1_buffer)
        if T == 0:
            return

        self._last_input = x_input.detach()                          # [T, batch, N_INPUT]
        self._last_h1    = torch.cat(self._h1_buffer, dim=0)         # [T, batch, N_H1]
        self._last_h2    = torch.cat(self._h2_buffer, dim=0)         # [T, batch, N_H2]
        self._last_out   = torch.cat(self._out_buffer, dim=0)        # [T, batch, N_OUTPUT]

    # -------------------------------------------------------------------
    def _spike_patterns_snapshot(self) -> dict[str, torch.Tensor] | None:
        """
        Captura um snapshot compacto dos padrões de spikes actuais.
        Usado pelo ExperienceReplay para guardar experiências.
        Devolve None se ainda não houve forward neste step.
        """
        if self._last_h1 is None:
            return None
        return {
            "h1":  self._last_h1.cpu().clone(),
            "h2":  self._last_h2.cpu().clone(),
            "out": self._last_out.cpu().clone(),
        }

    # -------------------------------------------------------------------
    def step(
        self,
        x_input: torch.Tensor,
        recompensa: float,
        *,
        estado: Any = None,
        accao: int | None = None,
        proximo_estado: Any = None,
    ) -> torch.Tensor:
        """
        Executa um step de aprendizagem R-STDP + NACA + AGMP:

        1. Forward pass (recolhe spikes via hooks).
        2. Actualiza eligibility traces.
        3. AGMPGate determina se a plasticidade está activa (novidade vs threshold).
        4. NACAGate actualiza o sinal neuromodulador (novidade × |recompensa_ema|).
        5. Actualiza pesos com NACA.apply() — apenas se AGMP indica aprendizagem.
        6. Guarda experiência no ExperienceReplay.

        Parâmetros opcionais (estado, accao, proximo_estado) são usados pelo
        ExperienceReplay. Código legado que omite estes argumentos funciona sem
        alteração — serão guardados como None no buffer.

        Devolve spike_counts [batch, N_OUTPUT].
        """
        self._colectar_spikes(x_input)

        if self._last_input is None:
            return torch.zeros(x_input.shape[1], N_OUTPUT, device=self.device)

        # 2. Actualizar eligibility traces
        self.et_fc1.actualizar(self._last_input, self._last_h1)
        self.et_fc2.actualizar(self._last_h1,    self._last_h2)
        # v33: fc3 usa one-hot da acção TOMADA (epsilon-greedy), não output interno SNN.
        # Corrige misalignment: SNN biasado para avançar reforçava avançar mesmo quando
        # a turn bonus recompensava viragens (et_fc3 correlacionava h2 com avançar interno).
        if accao is not None:
            self._fake_post.zero_()
            self._fake_post[:, 0, accao] = 1.0
            self.et_fc3.actualizar(self._last_h2, self._fake_post)
        else:
            self.et_fc3.actualizar(self._last_h2, self._last_out)

        # 3. AGMP: novidade = |taxa_actual - astrocyte_var (expectativa lenta)|
        spike_rate_actual   = self.snn.taxa_spikes_media
        spike_rate_esperado = self.agmp.astrocyte_var        # expectativa antes do update
        self._plasticidade_activa = self.agmp.step(spike_rate_actual, spike_rate_esperado)
        # v4.0: gate externo override — trainer controla via CV de recompensa
        if self._inferencia_forcada:
            self._plasticidade_activa = False

        # Baseline global + EMA de recompensa
        # Baseline: EMA lenta (tau≈10000) não reiniciada entre episódios.
        # r_adj = RPE (reward prediction error): sinal centrado em ~0 para comportamento típico.
        self._reward_baseline += 0.0001 * (recompensa - self._reward_baseline)
        r_adj = recompensa - self._reward_baseline
        self._reward_ema += (r_adj - self._reward_ema) / self._reward_tau

        # 4. NACA: gate usa recompensa bruta (detecta eventos significativos sem baseline)
        self.naca.update(self.agmp.novidade, recompensa)

        # 5. Actualizar pesos apenas se AGMP indica plasticidade activa
        if self._plasticidade_activa:
            self._actualizar_pesos(self._reward_ema)

        # 6. Guardar experiência no buffer de replay
        spike_patterns = self._spike_patterns_snapshot()
        if spike_patterns is not None:
            self.replay.store(estado, accao, recompensa, proximo_estado, spike_patterns)

        return self._last_out.sum(dim=0)   # [batch, N_OUTPUT]

    # -------------------------------------------------------------------
    @torch.no_grad()
    def _actualizar_pesos(self, recompensa: float) -> None:
        """
        Actualiza os pesos com R-STDP modulado pelo NACA gate:
          delta_w_base = lr × recompensa × eligibility_trace
          delta_w_naca = naca.apply(delta_w_base, W)
          W += delta_w_naca  (com clipping)
        """
        lr      = self.cfg.lr_snn
        clip_dw = self.cfg.grad_clip
        clip_w  = self.cfg.clip_w

        for camada, trace in [
            (self.snn.fc1, self.et_fc1.trace),
            (self.snn.fc2, self.et_fc2.trace),
            (self.snn.fc3, self.et_fc3.trace),
        ]:
            if camada.weight.shape != trace.shape:
                continue
            # R-STDP base
            delta_w = lr * recompensa * trace
            delta_w = delta_w.clamp(-clip_dw, clip_dw)
            # NACA gate: escala update pela novidade e penaliza pesos inactivos
            delta_w = self.naca.apply(delta_w, camada.weight)
            camada.weight.add_(delta_w)
            camada.weight.clamp_(-clip_w, clip_w)

    # -------------------------------------------------------------------
    def step_with_precomputed_spikes(
        self,
        x_input: torch.Tensor,      # [T, 1, N_INPUT] — input original deste organismo
        h1: torch.Tensor,           # [T, 1, N_H1]
        h2: torch.Tensor,           # [T, 1, N_H2]
        out: torch.Tensor,          # [T, 1, N_OUTPUT]
        recompensa: float,
        *,
        accao: int | None = None,
    ) -> None:
        """
        R-STDP update usando spikes pré-calculados pelo batched_forward_multi_snn.
        Evita um segundo forward pass redundante — usa as activações já computadas.
        Chamado pelo trainer_populacao quando device=cuda e N>1.
        """
        self._last_input = x_input.detach()
        self._last_h1    = h1.detach()
        self._last_h2    = h2.detach()
        self._last_out   = out.detach()

        # Eligibility traces
        self.et_fc1.actualizar(self._last_input, self._last_h1)
        self.et_fc2.actualizar(self._last_h1,    self._last_h2)
        if accao is not None:
            self._fake_post.zero_()
            self._fake_post[:, 0, accao] = 1.0
            self.et_fc3.actualizar(self._last_h2, self._fake_post)
        else:
            self.et_fc3.actualizar(self._last_h2, self._last_out)

        # AGMP: novidade a partir da taxa de spikes (já calculada pelo batched forward)
        spike_rate_actual   = self.snn.taxa_spikes_media
        spike_rate_esperado = self.agmp.astrocyte_var
        self._plasticidade_activa = self.agmp.step(spike_rate_actual, spike_rate_esperado)
        if self._inferencia_forcada:
            self._plasticidade_activa = False

        # Reward baseline + EMA
        self._reward_baseline += 0.0001 * (recompensa - self._reward_baseline)
        r_adj = recompensa - self._reward_baseline
        self._reward_ema += (r_adj - self._reward_ema) / self._reward_tau

        # NACA gate
        self.naca.update(self.agmp.novidade, recompensa)

        # Actualizar pesos (se AGMP activo)
        if self._plasticidade_activa:
            self._actualizar_pesos(self._reward_ema)

        # Experience replay — skipped in batch path: WorldRunner never calls consolidar_replay(),
        # so the 3× cpu().clone() snapshot cost is pure waste here.

    # -------------------------------------------------------------------
    def consolidar_replay(self, batch_size: int = 32) -> None:
        """
        Consolida experiências do buffer offline (fase de "sono").
        Deve ser chamada externamente entre episódios, nunca dentro do loop step().
        """
        self.replay.consolidar(self.snn, batch_size=batch_size)

    # -------------------------------------------------------------------
    def reset_traces(self) -> None:
        """Chamar no início de cada episódio para limpar traces e reward EMA."""
        self.et_fc1.reset()
        self.et_fc2.reset()
        self.et_fc3.reset()
        self._reward_ema = 0.0

    # -------------------------------------------------------------------
    def set_episode(self, ep: int) -> None:
        """
        Ajusta parâmetros dinâmicos por episódio.

        Lambda NACA (v42):
          lambda_naca = 1e-6 SEMPRE.
          Causa raiz v38-v41: lambda=0.1 eps 1-100 causava decay 10%/step em steps
          normais (gate≈0.0008 → ΔW≈-0.1×W). fc3 pesos → 0 em ep1. Pré-treino ANN
          destruído. RSTDP reconstruía pesos do zero com bias avançar (food |r|=1.29).

        LR warmup linear (v16 — correcção de v15):
          ep 1  : lr_snn × 0.5   (50% — protege sem congelar)
          ep 50 : lr_snn × 1.0   (100% — aprendizagem plena)
          ep >50: lr_snn base
        O warmup cresce linearmente, evitando o congelamento que ocorreu em v15 com ×0.15.
        """
        # v42: lambda_naca = 1e-6 sempre (não destruir pesos ANN pretrain)
        self.naca.lambda_naca = 1e-6

        if ep <= 50:
            factor = 0.5 + 0.5 * (ep / 50.0)   # linear: 0.5 (ep1) → 1.0 (ep50)
        else:
            factor = 1.0
        self.cfg.lr_snn = self._lr_snn_base * factor

    # -------------------------------------------------------------------
    def fim_episodio(self, ep: int) -> dict | None:
        """
        Chamar no fim de cada episódio (após _correr_episodio).
        Delega para AGMPGate.fim_episodio — retorna dict de calibração no ep20,
        None nos restantes episódios.
        """
        return self.agmp.fim_episodio(ep)


# ---------------------------------------------------------------------------
# Sinal de recompensa — v45: turn_bonus constante, sem TURN-OFF, sem WRONG-ADV
# ---------------------------------------------------------------------------

def calcular_recompensa(
    recompensa_ambiente: float,
    accao: str,
    dist_comida_antes: float | None,
    dist_comida_depois: float | None,
    morreu: bool,
    accao_idx: int = 0,
    accao_esperada: int | None = None,
    turn_bonus: float = 0.30,
    avancar_unconditional: float = 0.0,
    wrong_advance_penalty: float = 0.0,
    wrong_turn_penalty: float = 0.0,   # v45: 0.0 — v47 mostrou que reduz net turn reinforcement
) -> float:
    """
    Reward shaping v45: turn_bonus constante, lambda=1e-6, AGMP desactivado.

    Equilíbrio natural com turn_bonus=0.30: p_av≈5-10% (turn dominant).
    Organismo aprende directionality (prec_roll30→0.47) mas p_av baixo.
    prec_roll30 ainda em crescimento a ep300 (+0.029/50ep) → deixar correr 1500ep.
    """
    r = recompensa_ambiente - 0.01   # custo de movimento por step
    if accao_idx == 0:               # avançar
        r += avancar_unconditional
        # penalizar avançar errado (food lateral visible, organismo avança)
        if (accao_esperada is not None and accao_esperada != 0):
            r -= wrong_advance_penalty
    if accao_esperada is not None and accao_idx == accao_esperada:
        if accao_esperada == 0:      # avançar frontal (food a 0°)
            r += 0.30
        else:                        # viragem lateral (condicionada por turn_bonus)
            r += turn_bonus
    # v52: penalizar virar na direcção errada (food lateral, virou ao contrário)
    # Diferencial correcto→+0.30, errado→-0.10: discriminação 0.40 (vs 0.30 sem pena)
    if (accao_idx in {1, 2}
            and accao_esperada is not None
            and accao_esperada in {1, 2}
            and accao_idx != accao_esperada):
        r -= wrong_turn_penalty
    return r

