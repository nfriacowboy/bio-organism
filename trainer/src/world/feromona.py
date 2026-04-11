"""
Sistema de feromonas para Fase C — Comunicação stigmérgica inter-organismo.

Inspirado nos ascarosídeos de C. elegans (Srinivasan et al. 2012):
  Quando um organismo come → deposita feromona no grid.
  A feromona difunde-se radialmente e decai com o tempo.
  Outros organismos detectam a feromona por sectores (como o olfato).
  Organismos com SNN que respondem ao gradiente de feromona → vantagem selectiva.

Representação:
  feromona_grid: np.ndarray float32, shape = (H, W), valores em [0.0, 1.0].
  Partilha as dimensões do grid de comida (uint8).

Stigmergy (Grassé 1959): comunicação indirecta via modificação do ambiente.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Parâmetros físicos
# ---------------------------------------------------------------------------

FEROMONA_DEPOSIT   : float = 0.50   # intensidade depositada ao comer
FEROMONA_DECAY     : float = 0.003  # fracção decaída por step (half-life ≈ 231 steps)
FEROMONA_DIFF      : float = 0.06   # difusão para 4 vizinhos por step
RAIO_FEROMONA      : int   = 50     # raio de detecção (células)


# ---------------------------------------------------------------------------
# Gestão da grelha
# ---------------------------------------------------------------------------

def actualizar_feromona(
    grid: np.ndarray,
    decay: float = FEROMONA_DECAY,
    diff: float  = FEROMONA_DIFF,
) -> None:
    """
    Decaimento + difusão 4-neighbour na grelha de feromonas (in-place).

    Complexidade: O(H×W) com numpy vectorizado — adequado para grids até 1000×1000.
    Chamado 1× por step do mundo.
    """
    if grid.max() < 1e-9:
        return

    # 1. Decaimento exponencial
    grid *= 1.0 - decay

    # 2. Difusão 4-neighbour com roll (borders não difundem para fora)
    s_u = np.roll(grid, -1, axis=0); s_u[-1, :] = 0.0    # shift up
    s_d = np.roll(grid,  1, axis=0); s_d[0,  :] = 0.0    # shift down
    s_l = np.roll(grid, -1, axis=1); s_l[:, -1] = 0.0    # shift left
    s_r = np.roll(grid,  1, axis=1); s_r[:,  0] = 0.0    # shift right
    vizinhos = (s_u + s_d + s_l + s_r) * 0.25
    grid += diff * (vizinhos - grid)

    # 3. Clip [0, 1]
    np.clip(grid, 0.0, 1.0, out=grid)


def depositar_feromona(
    grid: np.ndarray,
    x: int,
    y: int,
    intensidade: float = FEROMONA_DEPOSIT,
) -> None:
    """Depositar feromona na posição (x, y) do organismo (saturação a 1.0)."""
    grid[y, x] = min(1.0, grid[y, x] + intensidade)


# ---------------------------------------------------------------------------
# Detecção por sectores (análogo a calcular_olfato)
# ---------------------------------------------------------------------------

def calcular_feromona_sectores(
    feromona_grid: np.ndarray,
    org_x: int,
    org_y: int,
    angulo: float,
    raio: int = RAIO_FEROMONA,
) -> dict[str, float]:
    """
    Detecção de feromona em 3 sectores direccionais (esq, frente, dir).

    Mesma lógica do olfato (v4.1): ponderação 1/d² pela distância.
    A soma esq+frente+dir = 1.0 se houver feromona no raio, 0.0 se não houver.

    Sectores (rel_deg = ângulo_célula - angulo, em [0°, 360°)):
      frente : [0°,45°) ∪ [315°,360°)   — ±45° à frente
      dir    : [45°,135°)               — +45° a +135°
      esq    : [225°,315°)              — -45° a -135°
      atrás  : [135°,225°)              — ignorado
    """
    H, W = feromona_grid.shape

    # 1. Bounding box clipada
    x0 = max(0, org_x - raio); x1 = min(W, org_x + raio + 1)
    y0 = max(0, org_y - raio); y1 = min(H, org_y + raio + 1)

    sub = feromona_grid[y0:y1, x0:x1]
    if sub.max() < 1e-9:
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}

    # 2. Posições dentro da bounding box com feromona > 0
    rows, cols = np.where(sub > 1e-9)
    if len(rows) == 0:
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}

    intensidades = sub[rows, cols].astype(np.float64)

    dy = (y0 + rows) - org_y   # positivo = Sul
    dx = (x0 + cols) - org_x   # positivo = Este

    # 3. Filtrar por raio circular
    dist_sq = dx * dx + dy * dy
    dentro = dist_sq <= raio * raio
    if not np.any(dentro):
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}

    dx = dx[dentro]; dy = dy[dentro]
    intensidades = intensidades[dentro]
    dist_sq_in = dist_sq[dentro].astype(np.float64)

    # 4. Ângulo relativo ao heading
    angle_abs = np.degrees(np.arctan2(dy, dx))
    rel_deg = (angle_abs - angulo) % 360.0

    # 5. Sectores
    mask_frente = (rel_deg < 45.0) | (rel_deg >= 315.0)
    mask_dir    = (rel_deg >= 45.0)  & (rel_deg < 135.0)
    mask_esq    = (rel_deg >= 225.0) & (rel_deg < 315.0)

    # 6. Ponderação 1/d² × intensidade de feromona
    _w = intensidades / (dist_sq_in + 1.0)
    n_frente = float(np.sum(_w[mask_frente]))
    n_dir    = float(np.sum(_w[mask_dir]))
    n_esq    = float(np.sum(_w[mask_esq]))

    total = n_esq + n_frente + n_dir
    if total < 1e-9:
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}
    return {
        "esq"   : n_esq    / total,
        "frente": n_frente / total,
        "dir"   : n_dir    / total,
    }
