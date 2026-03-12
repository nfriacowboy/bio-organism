"""
Sistema de visão do organismo: cone de 30° com 5 raios de raycasting.
Zonas: curta (<10), media (10-20), longa (heurística).
"""
from __future__ import annotations

import numpy as np

# Ângulos relativos dos 5 raios dentro do cone de 30°
CONE_ANGULOS_REL = [-15.0, -7.5, 0.0, 7.5, 15.0]

# Distância máxima de visão (quadrículas)
DIST_MAX = 20

# Limites de zona
DIST_CURTA = 10   # < 10 → curta
# 10–20 → media; >20 → longa (não alcança directamente)


def calcular_visao(
    mundo: np.ndarray,
    pos_x: int,
    pos_y: int,
    angulo_orientacao: float,
) -> dict[str, dict]:
    """
    Calcula o que o organismo vê no cone de visão.

    Devolve dict keyed por ângulo relativo (str) com:
        zona: 'curta' | 'media' | 'longa' | 'nada'
        tipo: int (1=comida, 2=obstáculo) | 'desconhecido' | 'indefinido' | None
        dist: int | None
    """
    altura, largura = mundo.shape
    resultado: dict[str, dict] = {}

    for ang_rel in CONE_ANGULOS_REL:
        chave = str(ang_rel)
        raio_ang = angulo_orientacao + ang_rel
        rad = np.radians(raio_ang)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        encontrou = False
        for dist in range(1, DIST_MAX + 1):
            nx = pos_x + round(dist * cos_a)
            ny = pos_y + round(dist * sin_a)

            if nx < 0 or nx >= largura or ny < 0 or ny >= altura:
                # Fora dos limites — parar este raio
                resultado[chave] = {"zona": "nada", "tipo": None, "dist": None}
                encontrou = True
                break

            objecto = int(mundo[ny, nx])
            if objecto != 0:
                if dist <= DIST_CURTA:
                    zona = "curta"
                    tipo = objecto          # sabe exactamente o que é
                else:
                    zona = "media"
                    tipo = "desconhecido"   # detecta presença, não tipo
                resultado[chave] = {"zona": zona, "tipo": tipo, "dist": dist}
                encontrou = True
                break

        if not encontrou:
            # Nenhum objecto detectado nos 20 quadrículas → zona longa (heurística)
            resultado[chave] = {"zona": "longa", "tipo": "indefinido", "dist": DIST_MAX}

    return resultado


def visao_para_intensidades(visao: dict[str, dict]) -> list[float]:
    """
    Converte a visão em 15 intensidades [0.0, 1.0] para o encoder de spikes.
    5 ângulos × 3 zonas (curta, media, longa) = 15 valores.

    Layout: [ang0_curta, ang0_media, ang0_longa, ang1_curta, ...]
    """
    intensidades = []
    for ang_rel in CONE_ANGULOS_REL:
        chave = str(ang_rel)
        info = visao.get(chave, {"zona": "nada", "tipo": None, "dist": None})
        zona = info["zona"]
        dist = info["dist"]

        if zona == "nada" or dist is None:
            intensidades.extend([0.0, 0.0, 0.0])
        elif zona == "curta":
            intens = max(0.0, 1.0 - dist / DIST_CURTA)
            intensidades.extend([intens, 0.0, 0.0])
        elif zona == "media":
            intens = max(0.0, 1.0 - dist / DIST_MAX)
            intensidades.extend([0.0, intens, 0.0])
        else:  # longa
            intensidades.extend([0.0, 0.0, 0.1])   # sinal fraco de zona longa

    return intensidades
