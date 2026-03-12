"""
Encoding do input sensorial → tensores de spikes (Rate Coding).

Mapeamento:
  15 neurónios de visão (5 ângulos × 3 zonas)
  + 3 neurónios de estado (energia_norm, tempo_vida_norm, bias=1.0)
  = 18 inputs para a SNN

Fórmula:
  spikes = round(intensidade * T)    intensidade ∈ [0.0, 1.0]
  Resultado: tensor binário [T, N_INPUT] com 0s e 1s.
"""
from __future__ import annotations

import torch
import numpy as np

from world.vision import visao_para_intensidades

T_DEFAULT = 8
N_VISAO   = 15   # 5 ângulos × 3 zonas
N_ESTADO  = 3    # energia_norm, tempo_vida_norm, bias
N_INPUT   = N_VISAO + N_ESTADO


def intensidade_para_spikes(intensidade: float, T: int = T_DEFAULT) -> list[int]:
    """
    Converte uma intensidade [0,1] em sequência de spikes de comprimento T.
    Distribui os spikes uniformemente ao longo dos T timesteps.
    """
    n_spikes = round(intensidade * T)
    n_spikes = max(0, min(n_spikes, T))
    if n_spikes == 0:
        return [0] * T
    if n_spikes == T:
        return [1] * T
    # Distribuição uniforme: colocar spikes espaçados regularmente
    spikes = [0] * T
    step = T / n_spikes
    for i in range(n_spikes):
        idx = int(i * step)
        spikes[min(idx, T - 1)] = 1
    return spikes


def observacao_para_tensor(
    obs: dict,
    T: int = T_DEFAULT,
    device: torch.device | None = None,
    tempo_vida_norm: float = 0.0,
) -> torch.Tensor:
    """
    Converte uma observação do mundo em tensor de spikes.

    Parâmetros
    ----------
    obs : dict com 'visao', 'energia_norm'
    T   : timesteps
    device : torch device
    tempo_vida_norm : tempo de vida normalizado [0,1]

    Devolve
    -------
    torch.Tensor de shape [T, 1, N_INPUT] (float32, 0s e 1s)
    """
    dev = device or torch.device("cpu")

    # --- Visão → 15 intensidades ---
    intensidades_visao = visao_para_intensidades(obs["visao"])   # lista de 15 floats

    # --- Estado → 3 valores ---
    energia_norm    = float(obs.get("energia_norm", obs["energia_actual"] / obs["energia_max"]))
    intensidades_estado = [energia_norm, tempo_vida_norm, 1.0]  # bias sempre 1

    # --- Concatenar e converter para spikes ---
    todas_intensidades = intensidades_visao + intensidades_estado   # 18 valores

    # Construir tensor [T, 18]
    tensor = torch.zeros(T, N_INPUT, dtype=torch.float32, device=dev)
    for i, intens in enumerate(todas_intensidades):
        seq = intensidade_para_spikes(intens, T)
        for t, s in enumerate(seq):
            tensor[t, i] = float(s)

    return tensor.unsqueeze(1)   # [T, 1, N_INPUT]


def batch_observacoes_para_tensor(
    obs_list: list[dict],
    tempo_vida_norms: list[float] | None = None,
    T: int = T_DEFAULT,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Converte uma lista de observações em batch de tensores de spikes.
    Devolve [T, batch, N_INPUT].
    """
    dev = device or torch.device("cpu")
    if tempo_vida_norms is None:
        tempo_vida_norms = [0.0] * len(obs_list)

    tensores = [
        observacao_para_tensor(obs, T, dev, tlv).squeeze(1)   # [T, N_INPUT]
        for obs, tlv in zip(obs_list, tempo_vida_norms)
    ]
    return torch.stack(tensores, dim=1)   # [T, batch, N_INPUT]
