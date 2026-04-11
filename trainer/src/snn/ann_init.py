"""
Geração de dados sintéticos e inicialização ANN-to-SNN para o Organismo.

Estratégia (spec §2.2):
  1. Gerar N amostras sintéticas com regras simples de orientação para comida.
  2. Pré-treinar o MLP equivalente (_ANNEquivalente) por `epochs` epochs com backprop.
  3. Copiar os pesos treinados para a SNN via converter_ann_para_snn.

Layout do vector de input v4.0 (45 elementos, intensidades em [0, 1]):
  [0:15]  visão (5 ângulos × 3 zonas)
  [15:18] estado (energia_norm, tempo_vida_norm, bias=1.0)
  [18:22] cópia eferente (one-hot acção anterior — neutro: [0,0,0,1]=parar)
  [22:37] fluxo óptico   (delta visual — neutro: 0.5 = sem mudança)
  [37:39] propriocepção  (em_movimento, direcao_rotacao — neutro: [0.0, 0.5])
  [39:42] olfacto        (esq, frente, dir — neutro: [0.0, 0.0, 0.0])
  [42:45] feromona       (esq, frente, dir — neutro: [0.0, 0.0, 0.0]) (Fase C)

Acções:
  0 = avancar
  1 = virar_esquerda
  2 = virar_direita
  3 = parar

Intensidades sintéticas são construídas directamente (sem simulação do mundo)
com valores compatíveis com os produzidos por visao_para_intensidades:
  zona curta → slot_curta ∈ [0.35, 0.90],  slot_media = slot_longa = 0.0
  zona media → slot_media ∈ [0.10, 0.50],  slot_curta = slot_longa = 0.0
  zona longa → slot_longa ∈ [0.05, 0.15],  slot_curta = slot_media = 0.0
  (consistente com vision.py: longa fixa em 0.1)
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .network import (
    OrganismoSNN,
    _ANNEquivalente,
    converter_ann_para_snn,
    N_INPUT,
)


# ---------------------------------------------------------------------------
# Constantes de layout do vector de input
# ---------------------------------------------------------------------------

# Índice base de cada ângulo nos 15 neurónios de visão.
# Ordem do layout: ângulos em [-15, -7.5, 0, +7.5, +15], cada um com 3 slots.
_BASE_IDX: dict[float, int] = {
    -15.0:  0,
    -7.5:   3,
     0.0:   6,
     7.5:   9,
    15.0:  12,
}

# Offset de zona dentro do triplo de cada ângulo (curta=0, media=1, longa=2)
_OFFSET_CURTA = 0
_OFFSET_MEDIA = 1
_OFFSET_LONGA = 2

# Índices dos neurónios de estado
_IDX_ENERGIA    = 15
_IDX_TEMPO_VIDA = 16
_IDX_BIAS       = 17

# Índices dos novos neurónios v2.0 (pré-treino usa valores neutros)
_IDX_EFERENTE_START = 18   # [18:22] one-hot acção anterior — neutro: parar=[0,0,0,1]
_IDX_FLOW_START     = 22   # [22:37] fluxo óptico          — neutro: 0.5
_IDX_PROP_START     = 37   # [37:39] propriocepção          — neutro: [0.0, 0.5]
# v3.0 — olfacto
_IDX_OLFATO_START   = 39   # [39:42] olfacto               — neutro: [0.0, 0.0, 0.0]
# v4.0 — feromona (Fase C)
_IDX_FEROMONA_START = 42   # [42:45] feromona              — neutro: [0.0, 0.0, 0.0]

# Índices de acção (winner-take-all da SNN)
AVANCAR        = 0
VIRAR_ESQUERDA = 1
VIRAR_DIREITA  = 2
PARAR          = 3

# Proporções de cada cenário no dataset sintético.
# sem_comida recebe o restante para garantir que sum == n_amostras exacto.
_FRAC_FRENTE     = 0.25   # v30: igualado a esq/dir → distribuição uniforme 33%/33%/33%
_FRAC_ESQUERDA   = 0.25
_FRAC_DIREITA    = 0.25
# _FRAC_SEM_COMIDA = 0.25  (calculado por diferença)


# ---------------------------------------------------------------------------
# Funções auxiliares de construção de vectores de intensidade
# ---------------------------------------------------------------------------

def _vetor_vazio() -> list[float]:
    """
    Devolve um vector de 42 zeros com bias=1.0, estado zero, e
    valores neutros para as dimensões v2.0 (eferente/flow/prop) e
    v3.0 (olfacto=0.0 — sem gradiente químico).

    v30: eferente aleatorio (não sempre parar=3).
    Durante o treino real a acção anterior é variada — fixar parar=3 criava
    distribution mismatch → SNN output avançar ~100% fora de distribuição.
    """
    v = [0.0] * N_INPUT   # 45 zeros — inclui [39:42]=0.0 (olfacto) e [42:45]=0.0 (feromona neutros)
    v[_IDX_BIAS] = 1.0
    # Eferente: aleatorio entre as 4 acções (v30: era sempre parar=3)
    acc_ant = random.randint(0, 3)
    v[_IDX_EFERENTE_START + acc_ant] = 1.0
    # Fluxo óptico neutro: 0.5 (sem mudança visual)
    for i in range(15):
        v[_IDX_FLOW_START + i] = 0.5
    # Propriocepção neutra: não em movimento, sem rotação
    v[_IDX_PROP_START]     = 0.0   # em_movimento=False
    v[_IDX_PROP_START + 1] = 0.5   # direcao_rotacao=neutral
    # Olfacto neutro: [39:42] já são 0.0 (inicializados por [0.0] * N_INPUT)
    return v


def _colocar_comida(
    v: list[float],
    angulo: float,
    offset_zona: int,
    intensidade: float,
) -> None:
    """
    Define a intensidade de um slot de visão específico (modificação in-place).

    Parâmetros
    ----------
    v           : vector de 18 floats (modificado in-place)
    angulo      : ângulo da raia de visão (-15.0, -7.5, 0.0, 7.5, 15.0)
    offset_zona : offset dentro do triplo (0=curta, 1=media, 2=longa)
    intensidade : valor de intensidade em [0.0, 1.0]
    """
    idx = _BASE_IDX[angulo] + offset_zona
    v[idx] = intensidade


def _intensidade_curta() -> float:
    """
    Intensidade aleatória para zona curta.
    Intervalo [0.35, 0.90] corresponde a dist ∈ [1, 6.5] na fórmula
    real de vision.py: 1.0 - dist / DIST_CURTA (DIST_CURTA=10).
    """
    return random.uniform(0.35, 0.90)


def _intensidade_media() -> float:
    """
    Intensidade aleatória para zona media.
    Intervalo [0.10, 0.50] corresponde a dist ∈ [10, 18] na fórmula
    real de vision.py: 1.0 - dist / DIST_MAX (DIST_MAX=20).
    """
    return random.uniform(0.10, 0.50)


def _estado_aleatorio(energia_min: float = 0.3) -> tuple[float, float]:
    """
    Devolve (energia_norm, tempo_vida_norm) com valores aleatórios realistas.

    Parâmetros
    ----------
    energia_min : valor mínimo de energia_norm (default 0.3 — organismo activo)
    """
    energia    = random.uniform(energia_min, 1.0)
    tempo_vida = random.uniform(0.0, 0.8)
    return energia, tempo_vida


# ---------------------------------------------------------------------------
# Geradores de amostras individuais por cenário
# ---------------------------------------------------------------------------

def _amostra_frente() -> tuple[list[float], int]:
    """
    Comida directamente em frente (ângulo 0.0), zona curta ou media.
    Acção correcta: avancar (0).
    Olfacto: frente≈0.44-0.60 (sector dominante), esq/dir≈0.15-0.28 — normalização relativa v4.0.
    """
    v = _vetor_vazio()
    # Escolher aleatoriamente entre curta e media com igual probabilidade
    if random.random() < 0.5:
        _colocar_comida(v, 0.0, _OFFSET_CURTA, _intensidade_curta())
    else:
        _colocar_comida(v, 0.0, _OFFSET_MEDIA, _intensidade_media())
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio()
    # v4.0: olfacto relativo-normalizado — frente dominante, esq/dir proporcionais
    v[_IDX_OLFATO_START]     = random.uniform(0.15, 0.28)   # esq
    v[_IDX_OLFATO_START + 1] = random.uniform(0.44, 0.60)   # frente (dominante)
    v[_IDX_OLFATO_START + 2] = random.uniform(0.15, 0.28)   # dir
    return v, AVANCAR


def _amostra_esquerda() -> tuple[list[float], int]:
    """
    Comida à esquerda (ângulo -7.5 ou -15.0), zona curta ou media.
    Acção correcta: virar_esquerda (1).
    Olfacto: esq≈0.44-0.60 (sector dominante), frente/dir≈0.15-0.28 — normalização relativa v4.0.
    """
    v = _vetor_vazio()
    angulo = random.choice([-7.5, -15.0])
    if random.random() < 0.5:
        _colocar_comida(v, angulo, _OFFSET_CURTA, _intensidade_curta())
    else:
        _colocar_comida(v, angulo, _OFFSET_MEDIA, _intensidade_media())
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio()
    # v4.0: olfacto relativo-normalizado — esq dominante
    v[_IDX_OLFATO_START]     = random.uniform(0.44, 0.60)   # esq (dominante)
    v[_IDX_OLFATO_START + 1] = random.uniform(0.15, 0.28)   # frente
    v[_IDX_OLFATO_START + 2] = random.uniform(0.15, 0.28)   # dir
    return v, VIRAR_ESQUERDA


def _amostra_direita() -> tuple[list[float], int]:
    """
    Comida à direita (ângulo +7.5 ou +15.0), zona curta ou media.
    Acção correcta: virar_direita (2).
    Olfacto: dir≈0.44-0.60 (sector dominante), esq/frente≈0.15-0.28 — normalização relativa v4.0.
    """
    v = _vetor_vazio()
    angulo = random.choice([7.5, 15.0])
    if random.random() < 0.5:
        _colocar_comida(v, angulo, _OFFSET_CURTA, _intensidade_curta())
    else:
        _colocar_comida(v, angulo, _OFFSET_MEDIA, _intensidade_media())
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio()
    # v4.0: olfacto relativo-normalizado — dir dominante
    v[_IDX_OLFATO_START]     = random.uniform(0.15, 0.28)   # esq
    v[_IDX_OLFATO_START + 1] = random.uniform(0.15, 0.28)   # frente
    v[_IDX_OLFATO_START + 2] = random.uniform(0.44, 0.60)   # dir (dominante)
    return v, VIRAR_DIREITA


def _amostra_sem_comida() -> tuple[list[float], int]:
    """
    Sem comida visível: todos os ângulos em zona longa com sinal fraco (≈0.1),
    consistente com o sinal de fundo produzido por vision.py para zona longa.

    Acção: uniforme entre as 3 acções motoras — sem PARAR nos labels de treino.
    (v30: usar randint(0,2) para que a distribuição final seja exactamente 33%/33%/33%.
     randint(0,3) incluía PARAR=3 → distribuição desequilibrada 35%/30%/30%/5% →
     ANN aprend bias para avancar → SNN iniciava com p_av≈70%.)
    """
    v = _vetor_vazio()
    # Sinal fraco de zona longa em todos os ângulos
    for angulo in _BASE_IDX:
        _colocar_comida(v, angulo, _OFFSET_LONGA, random.uniform(0.05, 0.15))
    # Estado variável incluindo energia baixa para cobrir mais cenários
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio(energia_min=0.1)
    # Distribuição uniforme sobre 3 acções motoras (sem PARAR — v30)
    label = random.randint(0, 2)
    return v, label


# ---------------------------------------------------------------------------
# Dataset sintético
# ---------------------------------------------------------------------------

def gerar_dados_sinteticos(
    n_amostras: int = 5000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gera um dataset sintético para pré-treino do MLP equivalente.

    Cada amostra é construída directamente como vector de intensidades
    rate-coded, sem necessidade de simulação do mundo.

    Distribuição de cenários:
      25% comida em frente   → avancar
      25% comida à esquerda → virar_esquerda
      25% comida à direita  → virar_direita
      25% sem comida visível → uniform{0,1,2} (sem PARAR)
    Distribuição efectiva de labels: 33.3% por acção motora (v30: equilibrado)

    Parâmetros
    ----------
    n_amostras : número total de amostras a gerar
    device     : dispositivo torch de destino (cpu por omissão)

    Devolve
    -------
    x : torch.Tensor shape [n_amostras, 42], dtype=float32
        Intensidades rate-coded (valores em [0, 1]) — input directo para o ANN
    y : torch.Tensor shape [n_amostras],     dtype=int64
        Índices de acção (0=avancar, 1=virar_esquerda, 2=virar_direita, 3=parar)
    """
    dev = device or torch.device("cpu")

    # Calcular contagens por cenário (sem_comida absorve arredondamentos)
    n_frente     = int(n_amostras * _FRAC_FRENTE)
    n_esquerda   = int(n_amostras * _FRAC_ESQUERDA)
    n_direita    = int(n_amostras * _FRAC_DIREITA)
    n_sem_comida = n_amostras - n_frente - n_esquerda - n_direita

    amostras: list[tuple[list[float], int]] = []

    for _ in range(n_frente):
        amostras.append(_amostra_frente())
    for _ in range(n_esquerda):
        amostras.append(_amostra_esquerda())
    for _ in range(n_direita):
        amostras.append(_amostra_direita())
    for _ in range(n_sem_comida):
        amostras.append(_amostra_sem_comida())

    # Baralhagem para eliminar viés de ordenação durante o treino
    random.shuffle(amostras)

    vetores, labels = zip(*amostras)

    x = torch.tensor(list(vetores), dtype=torch.float32, device=dev)  # [N, 42]
    y = torch.tensor(list(labels),  dtype=torch.int64,   device=dev)  # [N]

    return x, y


# ---------------------------------------------------------------------------
# Pré-treino e conversão ANN → SNN
# ---------------------------------------------------------------------------

def pretreinar_com_dados_sinteticos(
    snn: OrganismoSNN,
    n_amostras: int = 5000,
    epochs: int = 30,
    lr: float = 3e-3,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[_ANNEquivalente, float]:
    """
    Pré-treina o MLP equivalente com dados sintéticos e converte pesos para a SNN.
    Usa mini-batch SGD (batch_size=128) para garantir convergência rápida.
    """
    dev = device or getattr(snn, "device", torch.device("cpu"))

    x_treino, y_treino = gerar_dados_sinteticos(n_amostras=n_amostras, device=dev)

    ann         = _ANNEquivalente().to(dev)
    optimizador = torch.optim.Adam(ann.parameters(), lr=lr)
    criterio    = nn.CrossEntropyLoss()
    loader      = DataLoader(TensorDataset(x_treino, y_treino),
                             batch_size=batch_size, shuffle=True)

    ann.train()
    loss_final = float("inf")

    for _ep in range(epochs):
        for xb, yb in loader:
            optimizador.zero_grad()
            loss = criterio(ann(xb), yb)
            loss.backward()
            optimizador.step()
        loss_final = criterio(ann(x_treino), y_treino).item()

    ann.eval()
    converter_ann_para_snn(snn, ann)
    return ann, loss_final
