"""
Geração de dados sintéticos e inicialização ANN-to-SNN para o Organismo.

Estratégia (spec §2.2):
  1. Gerar N amostras sintéticas com regras simples de orientação para comida.
  2. Pré-treinar o MLP equivalente (_ANNEquivalente) por `epochs` epochs com backprop.
  3. Copiar os pesos treinados para a SNN via converter_ann_para_snn.

Layout do vector de input (18 elementos, intensidades em [0, 1]):
  [ang-15_curta, ang-15_media, ang-15_longa,
   ang-7.5_curta, ang-7.5_media, ang-7.5_longa,
   ang0_curta,   ang0_media,   ang0_longa,
   ang7.5_curta, ang7.5_media, ang7.5_longa,
   ang15_curta,  ang15_media,  ang15_longa,
   energia_norm, tempo_vida_norm, bias=1.0]

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

# Índices de acção (winner-take-all da SNN)
AVANCAR        = 0
VIRAR_ESQUERDA = 1
VIRAR_DIREITA  = 2
PARAR          = 3

# Proporções de cada cenário no dataset sintético.
# sem_comida recebe o restante para garantir que sum == n_amostras exacto.
_FRAC_FRENTE     = 0.30
_FRAC_ESQUERDA   = 0.25
_FRAC_DIREITA    = 0.25
# _FRAC_SEM_COMIDA = 0.20  (calculado por diferença)


# ---------------------------------------------------------------------------
# Funções auxiliares de construção de vectores de intensidade
# ---------------------------------------------------------------------------

def _vetor_vazio() -> list[float]:
    """
    Devolve um vector de 18 zeros com bias=1.0 e estado zero.
    Todos os slots de visão e energia/tempo_vida começam a zero.
    """
    v = [0.0] * N_INPUT
    v[_IDX_BIAS] = 1.0
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
    Variação de intensidade para robustez do treino.
    """
    v = _vetor_vazio()
    # Escolher aleatoriamente entre curta e media com igual probabilidade
    if random.random() < 0.5:
        _colocar_comida(v, 0.0, _OFFSET_CURTA, _intensidade_curta())
    else:
        _colocar_comida(v, 0.0, _OFFSET_MEDIA, _intensidade_media())
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio()
    return v, AVANCAR


def _amostra_esquerda() -> tuple[list[float], int]:
    """
    Comida à esquerda (ângulo -7.5 ou -15.0), zona curta ou media.
    Acção correcta: virar_esquerda (1).
    Ambos os ângulos de esquerda são usados para cobrir o campo visual esquerdo.
    """
    v = _vetor_vazio()
    angulo = random.choice([-7.5, -15.0])
    if random.random() < 0.5:
        _colocar_comida(v, angulo, _OFFSET_CURTA, _intensidade_curta())
    else:
        _colocar_comida(v, angulo, _OFFSET_MEDIA, _intensidade_media())
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio()
    return v, VIRAR_ESQUERDA


def _amostra_direita() -> tuple[list[float], int]:
    """
    Comida à direita (ângulo +7.5 ou +15.0), zona curta ou media.
    Acção correcta: virar_direita (2).
    Ambos os ângulos de direita são usados para cobrir o campo visual direito.
    """
    v = _vetor_vazio()
    angulo = random.choice([7.5, 15.0])
    if random.random() < 0.5:
        _colocar_comida(v, angulo, _OFFSET_CURTA, _intensidade_curta())
    else:
        _colocar_comida(v, angulo, _OFFSET_MEDIA, _intensidade_media())
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio()
    return v, VIRAR_DIREITA


def _amostra_sem_comida() -> tuple[list[float], int]:
    """
    Sem comida visível: todos os ângulos em zona longa com sinal fraco (≈0.1),
    consistente com o sinal de fundo produzido por vision.py para zona longa.

    Acção: avancar (60%) ou acção aleatória (40%) para ensinar exploração
    quando não há estímulo direcional claro.
    """
    v = _vetor_vazio()
    # Sinal fraco de zona longa em todos os ângulos
    for angulo in _BASE_IDX:
        _colocar_comida(v, angulo, _OFFSET_LONGA, random.uniform(0.05, 0.15))
    # Estado variável incluindo energia baixa para cobrir mais cenários
    v[_IDX_ENERGIA], v[_IDX_TEMPO_VIDA] = _estado_aleatorio(energia_min=0.1)
    # 60% avancar (comportamento por omissão), 40% acção aleatória
    label = AVANCAR if random.random() < 0.60 else random.randint(0, 3)
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
      30% comida em frente   → avancar
      25% comida à esquerda → virar_esquerda
      25% comida à direita  → virar_direita
      20% sem comida visível → avancar (60%) / aleatório (40%)

    Parâmetros
    ----------
    n_amostras : número total de amostras a gerar
    device     : dispositivo torch de destino (cpu por omissão)

    Devolve
    -------
    x : torch.Tensor shape [n_amostras, 18], dtype=float32
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

    x = torch.tensor(list(vetores), dtype=torch.float32, device=dev)  # [N, 18]
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
