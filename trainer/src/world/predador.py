"""
Predador para Fase C.2 — Pressão evolutiva de predação.

Inspirado em *Pristionchus pacificus* (predador natural de C. elegans):
  O predador persegue o organismo mais próximo e mata por contacto.
  Velocidade superior à dos organismos jovens (1.5× base).

Efeitos evolutivos previstos:
  - Selecção para dispersão espacial (evitar clustering perto do predador)
  - Selecção para velocidade (organismos mais rápidos sobrevivem mais)
  - Grupo detecta predador mais cedo (vigilância colectiva via feromona alarme)
  - Selfish Herd geometry (Hamilton 1971): organismos tendem a posicionar-se
    no centro do grupo para reduzir probabilidade de predação.

Referências:
  Hamilton (1971) — Geometry for the Selfish Herd
  Lima & Dill (1990) — Behavioral decisions under predation risk
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Parâmetros do predador
# ---------------------------------------------------------------------------

VELOCIDADE_PREDADOR  : float = 1.5     # células por step (vs 0.7-1.0 do organismo)
PREDADORES_POR_N     : int   = 10      # 1 predador por cada N organismos
RAIO_SPAWN_MIN       : int   = 100     # distância mínima de spawn aos organismos


# ---------------------------------------------------------------------------
# Classe Predador
# ---------------------------------------------------------------------------

class Predador:
    """
    Agente predador simples (não-SNN): persegue o organismo vivo mais próximo.

    Usa acumulação de velocidade fraccionária para velocidades não-inteiras:
    a cada step, self._vel_acum += velocidade; faz int(vel_acum) movimentos,
    guarda o resto — produz em média `velocidade` células por step.
    """

    def __init__(
        self,
        x: int,
        y: int,
        velocidade: float = VELOCIDADE_PREDADOR,
        id: int = 0,
    ):
        self.x = x
        self.y = y
        self._vel_acum: float = 0.0
        self.velocidade = velocidade
        self.id = id
        self.n_kills: int = 0

    def step(
        self,
        organismos: list,
        H: int,
        W: int,
    ) -> list[int]:
        """
        Move em direcção ao organismo vivo mais próximo.

        Devolve lista de índices de organismos mortos neste step.
        """
        alvos = [(i, org) for i, org in enumerate(organismos) if org.vivo]
        if not alvos:
            return []

        # Escolher alvo mais próximo (distância L2)
        alvo_i, alvo = min(
            alvos,
            key=lambda t: (t[1].x - self.x) ** 2 + (t[1].y - self.y) ** 2,
        )

        mortos: list[int] = []
        self._vel_acum += self.velocidade
        n_steps = int(self._vel_acum)
        self._vel_acum -= n_steps

        for _ in range(n_steps):
            dx = alvo.x - self.x
            dy = alvo.y - self.y
            if dx == 0 and dy == 0:
                break

            # Movimento Manhattan (1 célula por micro-step, eixo dominante)
            if abs(dx) >= abs(dy):
                self.x += 1 if dx > 0 else -1
            else:
                self.y += 1 if dy > 0 else -1

            # Clamp às bordas do grid
            self.x = max(0, min(W - 1, self.x))
            self.y = max(0, min(H - 1, self.y))

            # Verificar kill na nova posição
            for i, org in alvos:
                if org.vivo and org.x == self.x and org.y == self.y:
                    org.vivo = False
                    org.energia = 0.0
                    self.n_kills += 1
                    mortos.append(i)

        return mortos


# ---------------------------------------------------------------------------
# Spawn de predadores
# ---------------------------------------------------------------------------

def spawn_predadores(
    n_predadores: int,
    H: int,
    W: int,
    organismos: list,
    rng: np.random.Generator,
) -> list[Predador]:
    """
    Criar N predadores em posições aleatórias distantes dos organismos.

    Tenta colocar cada predador a >= RAIO_SPAWN_MIN do organismo mais próximo.
    Fallback: posição aleatória.
    """
    predadores = []
    for pid in range(n_predadores):
        for _ in range(100):
            px = int(rng.integers(0, W))
            py = int(rng.integers(0, H))
            if organismos:
                dist_min = min(
                    (org.x - px) ** 2 + (org.y - py) ** 2
                    for org in organismos if org.vivo
                )
                if dist_min >= RAIO_SPAWN_MIN ** 2:
                    break
        predadores.append(Predador(px, py, id=pid))
    return predadores
