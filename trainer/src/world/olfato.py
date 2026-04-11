"""
Sistema olfactivo do organismo: quimiocepção por sectores (inspirado em C. elegans).

v4.1: ponderação por proximidade (1/d²) + normalização relativa.
  Sem ponderação: 47 food uniformes em R=100 → frente≈esq≈dir≈0.33 (gradiente apagado).
  Com 1/d²: 1 food a d=10 + 46 background → frente≈0.61 — sinal claro para R-STDP.
  A soma esq+frente+dir = 1.0 se houver comida no raio, 0.0 se não houver.

v4.0: raio graduado por fase — olfato ultrapassa sempre o alcance visual (~15-20 células):
  Fase 1 (200×200, 3.75%): R=30  — gradiente direcional visível, olfato complementa visão
  Fase 2 (500×500, 1.2%):  R=60  — gradiente mais forte, olfato começa a dominar navegação
  Fase 3 (1000×1000, 0.15%): R=100 — sinal aguçado, ferramenta primária de navegação

3 sectores direccionais (relativo ao heading):
  esq   : -45° a -135°   (rel. 225° a 315°)
  frente: ±45°            (rel. 0° a 45° e 315° a 360°)
  dir   : +45° a +135°   (rel. 45° a 135°)
  atrás [135°, 225°): ignorado.

Totalmente vectorizado com numpy — sem loops Python por step.
Optimização bounding box: scan apenas nas células [org-R, org+R]×[org-R, org+R].
"""
from __future__ import annotations

import numpy as np

# Raio do campo olfactivo em células (default / legado)
RAIO_OLFATO: int = 100

# Legado v3.0 — não usado desde v4.0 (substituído por normalização relativa).
N_OLFATO_NORM: float = 40.0

# v4.0: raio do campo olfactivo por fase do curriculum.
# Garante que olfato ultrapassa sempre o alcance visual em todos os mundos.
RAIO_OLFATO_POR_FASE: dict[int, int] = {1: 30, 2: 60, 3: 100}


def calcular_olfato(
    grid: np.ndarray,
    org_x: int,
    org_y: int,
    angulo: float,
    raio: int = RAIO_OLFATO,
) -> dict[str, float]:
    """
    Calcula os sinais olfactivos ponderados por distância nos 3 sectores direccionais.

    v4.1: ponderação 1/d² — comida próxima domina sinal. Normalização relativa.
    Com 1 item a d=10 + 46 background R=100: frente≈0.61 (vs 0.33 sem ponderação).
    Sinal escala-invariante: esq+frente+dir=1.0 quando há comida no raio.

    Parâmetros
    ----------
    grid   : numpy array (H×W); 1=comida, 2=obstáculo, 0=vazio
    org_x  : coluna do organismo (coordenada X)
    org_y  : linha do organismo  (coordenada Y)
    angulo : heading em graus (convenção trigonométrica: 0=Este, 90=Sul)
    raio   : raio do campo olfactivo em células (default: RAIO_OLFATO=100)

    Devolve
    -------
    dict com chaves "esq", "frente", "dir" e valores float em [0.0, 1.0].
    A soma esq+frente+dir = 1.0 se houver comida no raio, 0.0 se não houver.

    Sectores (rel_deg = ângulo_comida - angulo, em [0°, 360°)):
      frente : rel_deg in [0°, 45°)  ∪  [315°, 360°)  — ±45° à frente
      dir    : rel_deg in [45°, 135°)                  — +45° a +135°
      esq    : rel_deg in [225°, 315°)                 — -45° a -135°
      atrás  : rel_deg in [135°, 225°)                 — ignorado
    """
    H, W = grid.shape

    # 1. Bounding box clipada às bordas do grid
    x0 = max(0, org_x - raio)
    x1 = min(W, org_x + raio + 1)
    y0 = max(0, org_y - raio)
    y1 = min(H, org_y + raio + 1)

    sub = grid[y0:y1, x0:x1]

    # 2. Posições de comida na bounding box (row=Y_rel, col=X_rel)
    posicoes = np.argwhere(sub == 1)   # shape [N_food, 2]

    if len(posicoes) == 0:
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}

    # 3. Deslocamentos relativos ao organismo (int64 por omissão no numpy)
    dy = (y0 + posicoes[:, 0]) - org_y   # positivo = Sul
    dx = (x0 + posicoes[:, 1]) - org_x   # positivo = Este

    # 4. Filtrar por raio circular exacto (bounding box é quadrada)
    dist_sq = dx * dx + dy * dy
    dentro = dist_sq <= raio * raio

    if not np.any(dentro):
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}

    dx = dx[dentro]
    dy = dy[dentro]

    # 5. Ângulo absoluto de cada célula de comida (graus, -180 a 180)
    angle_abs = np.degrees(np.arctan2(dy, dx))

    # 6. Ângulo relativo ao heading, mapeado para [0°, 360°)
    rel_deg = (angle_abs - angulo) % 360.0

    # 7. Classificação por sector (fronteiras sem overlap)
    mask_frente = (rel_deg < 45.0) | (rel_deg >= 315.0)
    mask_dir    = (rel_deg >= 45.0)  & (rel_deg < 135.0)
    mask_esq    = (rel_deg >= 225.0) & (rel_deg < 315.0)

    # v4.1: ponderação por proximidade (1/d²) — food próxima domina sinal direcional.
    # dist_sq[dentro] são as distâncias² filtradas ao raio (alinhadas com mask_*).
    _w = 1.0 / (dist_sq[dentro].astype(np.float64) + 1.0)
    n_frente = float(np.sum(_w[mask_frente]))
    n_dir    = float(np.sum(_w[mask_dir]))
    n_esq    = float(np.sum(_w[mask_esq]))

    # Normalização relativa — proporção por sector (exclui sector atrás).
    total = n_esq + n_frente + n_dir
    if total < 1e-9:
        return {"esq": 0.0, "frente": 0.0, "dir": 0.0}
    return {
        "esq"   : n_esq    / total,
        "frente": n_frente / total,
        "dir"   : n_dir    / total,
    }
