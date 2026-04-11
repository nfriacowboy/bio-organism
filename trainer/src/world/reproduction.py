"""
Reprodução e herança de pesos para Fase A (Mundo v2.0).

Implementa:
  - Extracção/carregamento de genoma (pesos SNN + KAN como "DNA")
  - Crossover layer-wise entre dois progenitores
  - Mutação gaussiana pós-crossover
  - Herança epigenética parcial (reward_baseline do RSTDPLearner)
  - Condições de reprodução e custo energético

Referências:
  Stanley & Miikkulainen (2002) NEAT
  Jablonka & Lamb (2005) — epigenetic inheritance
  Baldwin (1896) — learned characteristics influence evolution
"""
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from snn.network import OrganismoSNN
    from snn.kan_policy import KANPolicy
    from snn.learning import RSTDPLearner
    from world.environment import Organismo

# ---------------------------------------------------------------------------
# Parâmetros de reprodução (configuráveis)
# ---------------------------------------------------------------------------

ENERGIA_MIN_REPRODUCAO = 0.30       # fracção de energia_max mínima para ambos os pais (0.30 = 60/200)
RAIO_REPRODUCAO        = 100        # distância máxima em células (raio quimiotáxico)
FERTILIDADE_MAX_FRAC   = 0.70       # acima deste fracção de vida_max → infértil
VIDA_MAX_ORGANISMO     = 5000       # steps — para cálculo de fertilidade
CUSTO_ENERGIA_FRAC     = 0.30       # fracção de energia_max consumida por progenitor
COOLDOWN_REPRODUÇÃO    = 200        # steps mínimos entre reproduções do mesmo organismo
MUTACAO_SIGMA_DEFAULT  = 0.01       # desvio padrão da mutação gaussiana
EPIGETICA_DECAY        = 0.30       # fracção do reward_baseline herdado dos pais

# Fase B.2 — Anti-endogamia (Charlesworth & Charlesworth 1987)
INBREEDING_COS_THR     = 0.95       # cosine similarity fc1.weight acima deste → infértil

# Layers do SNN incluídas no genoma (por nome do state_dict)
SNN_LAYERS_GENOMA = [
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
    "fc2_rec.weight",
    "fc3.weight", "fc3.bias",
]


# ---------------------------------------------------------------------------
# Operações de genoma
# ---------------------------------------------------------------------------

def get_genome(snn: "OrganismoSNN", kan: "KANPolicy") -> dict[str, torch.Tensor]:
    """
    Extrai o genoma do organismo (pesos SNN + KAN) como cópias desligadas do grafo.

    Retorna um dict plano com chave = nome_da_layer.
    """
    genome: dict[str, torch.Tensor] = {}
    sd = snn.state_dict()
    for k in SNN_LAYERS_GENOMA:
        if k in sd:
            genome[f"snn.{k}"] = sd[k].clone().detach().cpu()
    # KAN completo (state_dict do multikan interno)
    for k, v in kan.kan.state_dict().items():
        genome[f"kan.{k}"] = v.clone().detach().cpu()
    return genome


def set_genome(
    snn: "OrganismoSNN",
    kan: "KANPolicy",
    genome: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Carrega genoma em SNN + KAN (in-place)."""
    # SNN
    sd = snn.state_dict()
    for k in SNN_LAYERS_GENOMA:
        gk = f"snn.{k}"
        if gk in genome and k in sd:
            sd[k].copy_(genome[gk].to(device))
    snn.load_state_dict(sd)
    # KAN
    kan_sd = kan.kan.state_dict()
    for k in list(kan_sd.keys()):
        gk = f"kan.{k}"
        if gk in genome:
            kan_sd[k] = genome[gk].to(device)
    kan.kan.load_state_dict(kan_sd)


def crossover(
    genome_a: dict[str, torch.Tensor],
    genome_b: dict[str, torch.Tensor],
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """
    Crossover layer-wise: cada entrada do genoma vem de A ou B com 50% de probabilidade.

    Nota: agrupa por prefixo layer (ex: "snn.fc1.*" é cruzado como unidade).
    Isto preserva a coerência interna de cada camada e evita incoerências
    weight/bias (ex: fc1.weight de A com fc1.bias de B).
    """
    filho: dict[str, torch.Tensor] = {}

    # Agrupar por layer (prefixo até ao 2.º ponto)
    layers: dict[str, list[str]] = {}
    for k in genome_a:
        # chave: "snn.fc1.weight" → group "snn.fc1"
        parts = k.rsplit(".", 1)
        group = parts[0] if len(parts) > 1 else k
        layers.setdefault(group, []).append(k)

    for group, keys in layers.items():
        usa_a = bool(rng.random() < 0.5)
        src = genome_a if usa_a else genome_b
        for k in keys:
            filho[k] = src[k].clone()

    return filho


def mutate(
    genome: dict[str, torch.Tensor],
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """
    Mutação gaussiana: W += N(0, σ) para cada elemento do genoma.

    KAN state_dict pode conter buffers não-treináveis (ex: grid points);
    só mutar tensors float com nomes típicos de pesos/biases.
    """
    mutado: dict[str, torch.Tensor] = {}
    for k, v in genome.items():
        if v.dtype in (torch.float32, torch.float64) and (
            "weight" in k or "bias" in k or "coef" in k or "scale" in k or "base_weight" in k
        ):
            noise = torch.tensor(rng.normal(0.0, sigma, v.shape), dtype=v.dtype)
            mutado[k] = v + noise
        else:
            mutado[k] = v.clone()
    return mutado


def herdar_epigenetica(
    learner_filho: "RSTDPLearner",
    learner_a: "RSTDPLearner",
    learner_b: "RSTDPLearner",
    decay: float = EPIGETICA_DECAY,
) -> None:
    """
    Herança epigenética parcial (Baldwin Effect):
    O filho herda uma fracção do reward_baseline dos pais.

    O reward_baseline representa experiência acumulada de recompensa.
    Com decay=0.3, o filho começa "pré-tendenciado" mas não com memória completa.
    """
    baseline_medio = (learner_a._reward_baseline + learner_b._reward_baseline) / 2.0
    learner_filho._reward_baseline = baseline_medio * decay


# ---------------------------------------------------------------------------
# Condições e execução de reprodução
# ---------------------------------------------------------------------------

def podem_reproduzir(
    org_a: "Organismo",
    org_b: "Organismo",
) -> bool:
    """
    Verifica se dois organismos satisfazem as condições de reprodução.

    Condições (todas obrigatórias):
      1. Ambos vivos
      2. Energia de cada um > ENERGIA_MIN_REPRODUCAO × energia_max
      3. Distância espacial < RAIO_REPRODUCAO
      4. Idade de cada um < FERTILIDADE_MAX_FRAC × VIDA_MAX_ORGANISMO
      5. Cooldown de reprodução esgotado para ambos
    """
    if not (org_a.vivo and org_b.vivo):
        return False
    if org_a is org_b:
        return False
    e_min = ENERGIA_MIN_REPRODUCAO * org_a.energia_max
    if org_a.energia < e_min or org_b.energia < e_min:
        return False
    dist = math.sqrt((org_a.x - org_b.x) ** 2 + (org_a.y - org_b.y) ** 2)
    if dist >= RAIO_REPRODUCAO:
        return False
    idade_max = FERTILIDADE_MAX_FRAC * VIDA_MAX_ORGANISMO
    if org_a.idade >= idade_max or org_b.idade >= idade_max:
        return False
    if org_a.cooldown_rep > 0 or org_b.cooldown_rep > 0:
        return False
    return True


def executar_reproducao(
    org_a: "Organismo",
    org_b: "Organismo",
    snn_a: "OrganismoSNN",
    kan_a: "KANPolicy",
    learner_a: "RSTDPLearner",
    snn_b: "OrganismoSNN",
    kan_b: "KANPolicy",
    learner_b: "RSTDPLearner",
    novo_id: int,
    nova_geracao: int,
    device: torch.device,
    rng: np.random.Generator,
    sigma: float = MUTACAO_SIGMA_DEFAULT,
) -> tuple["OrganismoSNN", "KANPolicy", "RSTDPLearner"]:
    """
    Executa a reprodução: crossover + mutação + herança epigenética.

    Efeitos colaterais:
      - Consome energia dos progenitores (-CUSTO_ENERGIA_FRAC × energia_max)
      - Reset cooldown_rep dos progenitores

    Retorna (snn_filho, kan_filho, learner_filho) prontos para inserir no mundo.
    """
    from snn.network import OrganismoSNN
    from snn.kan_policy import KANPolicy
    from snn.learning import RSTDPLearner

    # 1. Custo energético
    custo = CUSTO_ENERGIA_FRAC * org_a.energia_max
    org_a.energia = max(1.0, org_a.energia - custo)
    org_b.energia = max(1.0, org_b.energia - custo)
    org_a.n_filhos += 1
    org_b.n_filhos += 1
    org_a.cooldown_rep = COOLDOWN_REPRODUÇÃO
    org_b.cooldown_rep = COOLDOWN_REPRODUÇÃO

    # 2. Extrair genomas
    g_a = get_genome(snn_a, kan_a)
    g_b = get_genome(snn_b, kan_b)

    # 3. Crossover + mutação
    g_filho = crossover(g_a, g_b, rng)
    g_filho = mutate(g_filho, sigma, rng)

    # 4. Criar novos modelos e carregar genoma
    snn_f = OrganismoSNN(device=device).to(device)
    kan_f = KANPolicy(device=device)
    set_genome(snn_f, kan_f, g_filho, device)

    # 5. Learner com herança epigenética
    learner_f = RSTDPLearner(snn_f, device=device)
    herdar_epigenetica(learner_f, learner_a, learner_b)

    return snn_f, kan_f, learner_f


# ---------------------------------------------------------------------------
# Métricas de diversidade genética
# ---------------------------------------------------------------------------

def calcular_diversidade(
    snns: list["OrganismoSNN"],
) -> dict[str, float]:
    """
    Calcula métricas de diversidade genética da população.

    Usa os pesos fc1.weight de cada SNN como representação do genoma.

    Returns:
        cv_fc1  : coeficiente de variação (std/|mean|) dos pesos fc1
        dist_media : distância L2 média entre todos os pares
    """
    if len(snns) < 2:
        return {"cv_fc1": 0.0, "dist_media": 0.0}

    pesos = []
    for snn in snns:
        v = snn.fc1.weight.detach().cpu().flatten()
        pesos.append(v)

    stack = torch.stack(pesos)  # [N, D]
    mu = stack.mean().item()
    sigma = stack.std().item()
    cv = sigma / (abs(mu) + 1e-9)

    # Distância L2 média entre todos os pares
    n = len(pesos)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (pesos[i] - pesos[j]).norm().item()
            dists.append(d)
    dist_media = float(np.mean(dists)) if dists else 0.0

    return {"cv_fc1": float(cv), "dist_media": dist_media}


def cosine_sim_snn(
    snn_a: "OrganismoSNN",
    snn_b: "OrganismoSNN",
) -> float:
    """
    Cosine similarity entre dois SNNs baseada em fc1.weight.

    Retorna valor em [-1, 1]. Acima de INBREEDING_COS_THR → risco de endogamia.
    """
    v_a = snn_a.fc1.weight.detach().cpu().flatten().float()
    v_b = snn_b.fc1.weight.detach().cpu().flatten().float()
    dot = (v_a * v_b).sum().item()
    na = v_a.norm().item()
    nb = v_b.norm().item()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)
