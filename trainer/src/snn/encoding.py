"""
Encoding do input sensorial → tensores de spikes (Rate Coding).

v2.0 — loop sensoriomotor fechado (39 inputs):
  [0:15]  visual actual      (5 ângulos × 3 zonas)
  [15:18] estado             (energia_norm, tempo_vida_norm, bias=1.0)
  [18:22] cópia eferente     (one-hot acção anterior, 4 neurónios)
  [22:37] fluxo óptico       (delta visual actual vs anterior, 15 neurónios)
  [37:39] propriocepção      (em_movimento, direcao_rotacao, 2 neurónios)

v3.0 — olfacto adicionado (42 inputs):
  [39:42] olfacto            (esq, frente, dir — quimiocepção por sectores)

v4.0 — feromonas adicionadas (45 inputs):
  [42:45] feromona           (esq, frente, dir — detecção stigmérgica)

v1.0 (legacy): observacao_para_tensor() preenche [18:45] com valores neutros.
"""
from __future__ import annotations

import torch
import numpy as np

from world.vision import visao_para_intensidades

T_DEFAULT  = 8
N_VISAO    = 15
N_ESTADO   = 3
N_EFERENTE = 4
N_FLOW     = 15
N_PROP     = 2
N_OLFATO   = 3   # v3.0: olfacto — [esq, frente, dir]
N_FEROMONA = 3   # v4.0: feromona — [esq, frente, dir]
N_INPUT    = N_VISAO + N_ESTADO + N_EFERENTE + N_FLOW + N_PROP + N_OLFATO + N_FEROMONA   # 45


# ---------------------------------------------------------------------------
# Codificação temporal (rate coding)
# ---------------------------------------------------------------------------

def intensidade_para_spikes(intensidade: float, T: int = T_DEFAULT) -> list[int]:
    n_spikes = round(intensidade * T)
    n_spikes = max(0, min(n_spikes, T))
    if n_spikes == 0:
        return [0] * T
    if n_spikes == T:
        return [1] * T
    spikes = [0] * T
    step = T / n_spikes
    for i in range(n_spikes):
        idx = int(i * step)
        spikes[min(idx, T - 1)] = 1
    return spikes


# Module-level LUT for T=8 (T_DEFAULT): 9 possible spike counts (0..8)
# Each entry is a CPU float32 tensor of shape [T_DEFAULT].
_LUT8: list[torch.Tensor] = [
    torch.tensor(intensidade_para_spikes(n / T_DEFAULT, T=T_DEFAULT), dtype=torch.float32)
    for n in range(T_DEFAULT + 1)
]


def _intensidades_para_tensor(
    intensidades: list[float],
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """Converte lista de N intensidades → tensor [T, N] de spikes."""
    if T == T_DEFAULT:
        # Fast path: LUT lookup + single stack call (avoids 360-iteration Python loop)
        cols = [_LUT8[max(0, min(T_DEFAULT, round(v * T_DEFAULT)))] for v in intensidades]
        return torch.stack(cols, dim=1).to(device)  # [T, N]
    # Slow path for non-standard T
    n = len(intensidades)
    tensor = torch.zeros(T, n, dtype=torch.float32, device=device)
    for i, intens in enumerate(intensidades):
        seq = intensidade_para_spikes(intens, T)
        for t, s in enumerate(seq):
            tensor[t, i] = float(s)
    return tensor


# ---------------------------------------------------------------------------
# Encoders v2.0
# ---------------------------------------------------------------------------

def encode_efference_copy(last_action: int) -> torch.Tensor:
    """
    Cópia eferente: one-hot da acção executada no step anterior.
    last_action: 0=avancar, 1=virar_esq, 2=virar_dir, 3=parar
    Retorna tensor [4], valores em {0.0, 1.0}.
    """
    one_hot = torch.zeros(4)
    one_hot[last_action] = 1.0
    return one_hot


def encode_optical_flow(
    visual_actual: torch.Tensor,
    last_visual: torch.Tensor,
) -> torch.Tensor:
    """
    Fluxo óptico: delta entre campo visual actual e anterior, normalizado.
    visual_actual, last_visual: tensors [15], valores em [0, 1].
    Retorna tensor [15], valores em [0, 1]. (0.5 = sem mudança)
    """
    delta = visual_actual - last_visual     # [-1, 1]
    return (delta + 1.0) / 2.0             # [0, 1]


def encode_proprioception(last_action: int) -> torch.Tensor:
    """
    Propriocepção mínima: estado de movimento no step anterior.
    Neurónio 0: em_movimento (1.0=avancar, 0.0=outro)
    Neurónio 1: direcao_rotacao (0.0=esq, 0.5=nenhuma, 1.0=dir)
    Retorna tensor [2], valores em [0, 1].
    """
    prop = torch.zeros(2)
    if last_action == 0:      # avancar
        prop[0] = 1.0
        prop[1] = 0.5
    elif last_action == 1:    # virar esquerda
        prop[0] = 0.0
        prop[1] = 0.0
    elif last_action == 2:    # virar direita
        prop[0] = 0.0
        prop[1] = 1.0
    else:                     # parar
        prop[0] = 0.0
        prop[1] = 0.5
    return prop


# ---------------------------------------------------------------------------
# Encoder completo v2.0
# ---------------------------------------------------------------------------

def encode_full_state(
    organismo,
    obs: dict,
    T: int = T_DEFAULT,
    device: torch.device | None = None,
    tempo_vida_norm: float = 0.0,
) -> torch.Tensor:
    """
    Encoding completo v3.0 — 42 inputs.
    Usa organismo.visual_actual, organismo.last_visual, organismo.last_action.
    Devolve torch.Tensor [T, 1, 42].
    """
    dev = device or torch.device("cpu")

    # [0:15] visual actual
    intensidades_visao = list(organismo.visual_actual.cpu().numpy())   # 15 floats

    # [15:18] estado
    energia_norm = float(obs.get("energia_norm", obs["energia_actual"] / obs["energia_max"]))
    todas = intensidades_visao + [energia_norm, tempo_vida_norm, 1.0]   # 18 valores
    tensor_base = _intensidades_para_tensor(todas, T, dev)               # [T, 18]

    # [18:22] cópia eferente
    eferente = encode_efference_copy(organismo.last_action).tolist()
    eferente_spikes = _intensidades_para_tensor(eferente, T, dev)        # [T, 4]

    # [22:37] fluxo óptico
    flow = encode_optical_flow(
        organismo.visual_actual.to(dev),
        organismo.last_visual.to(dev),
    ).tolist()
    flow_spikes = _intensidades_para_tensor(flow, T, dev)                # [T, 15]

    # [37:39] propriocepção
    prop = encode_proprioception(organismo.last_action).tolist()
    prop_spikes = _intensidades_para_tensor(prop, T, dev)                # [T, 2]

    # [39:42] olfacto
    _olf = obs.get("olfato", {"esq": 0.0, "frente": 0.0, "dir": 0.0})
    olfato = [
        float(_olf.get("esq",    0.0)),
        float(_olf.get("frente", 0.0)),
        float(_olf.get("dir",    0.0)),
    ]
    olfato_spikes = _intensidades_para_tensor(olfato, T, dev)            # [T, 3]

    # [42:45] feromona (Fase C — stigmergy)
    _fer = obs.get("feromona", {"esq": 0.0, "frente": 0.0, "dir": 0.0})
    feromona = [
        float(_fer.get("esq",    0.0)),
        float(_fer.get("frente", 0.0)),
        float(_fer.get("dir",    0.0)),
    ]
    feromona_spikes = _intensidades_para_tensor(feromona, T, dev)        # [T, 3]

    tensor = torch.cat(
        [tensor_base, eferente_spikes, flow_spikes, prop_spikes, olfato_spikes, feromona_spikes],
        dim=1,
    )
    return tensor.unsqueeze(1)   # [T, 1, 45]


# ---------------------------------------------------------------------------
# Compatibilidade legacy (v1.0) — preenche [18:39] com valores neutros
# ---------------------------------------------------------------------------

def observacao_para_tensor(
    obs: dict,
    T: int = T_DEFAULT,
    device: torch.device | None = None,
    tempo_vida_norm: float = 0.0,
) -> torch.Tensor:
    """
    Encoding legacy com padding neutro para as dimensões v2.0 e v3.0.
    [0:18]  codificados normalmente.
    [18:22] eferente = [0,0,0,1] (parar — acção neutra).
    [22:37] flow = 0.5 (sem mudança visual).
    [37:39] prop = [0.0, 0.5] (parado, sem rotação).
    [39:42] olfato = [0.0, 0.0, 0.0] (neutro — sem gradiente).
    Devolve [T, 1, 42].
    """
    dev = device or torch.device("cpu")
    intensidades_visao = visao_para_intensidades(obs["visao"])
    energia_norm = float(obs.get("energia_norm", obs["energia_actual"] / obs["energia_max"]))
    todas = intensidades_visao + [energia_norm, tempo_vida_norm, 1.0]   # 18

    tensor_base     = _intensidades_para_tensor(todas, T, dev)
    eferente_spikes = _intensidades_para_tensor([0.0, 0.0, 0.0, 1.0], T, dev)
    flow_spikes     = _intensidades_para_tensor([0.5] * 15, T, dev)
    prop_spikes     = _intensidades_para_tensor([0.0, 0.5], T, dev)
    olfato_spikes   = _intensidades_para_tensor([0.0, 0.0, 0.0], T, dev)
    feromona_spikes = _intensidades_para_tensor([0.0, 0.0, 0.0], T, dev)

    tensor = torch.cat(
        [tensor_base, eferente_spikes, flow_spikes, prop_spikes, olfato_spikes, feromona_spikes],
        dim=1,
    )
    return tensor.unsqueeze(1)   # [T, 1, 45]


def batch_observacoes_para_tensor(
    obs_list: list[dict],
    tempo_vida_norms: list[float] | None = None,
    T: int = T_DEFAULT,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Converte lista de observações em batch de tensores (legacy).
    Devolve [T, batch, 39].
    """
    dev = device or torch.device("cpu")
    if tempo_vida_norms is None:
        tempo_vida_norms = [0.0] * len(obs_list)
    tensores = [
        observacao_para_tensor(obs, T, dev, tlv).squeeze(1)
        for obs, tlv in zip(obs_list, tempo_vida_norms)
    ]
    return torch.stack(tensores, dim=1)   # [T, batch, 42]
