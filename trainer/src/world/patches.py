"""
Dinâmica de patches de comida para Mundo v2.0.

Modelo baseado em resource patch dynamics (MacArthur & Wilson 1967):
  growing  → patch cresce até max_food
  peak     → patch estabilizado; começa a depleted após peak_duration steps
  depleting → organismos comem; abaixo de deplete_thr → recovering
  recovering → cresce lentamente de volta a growing

O PatchManager escreve e lê directamente no numpy grid partilhado do Mundo.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

PatchState = Literal["growing", "peak", "depleting", "recovering"]


@dataclass
class FoodPatch:
    cx: int           # centro X (coluna)
    cy: int           # centro Y (linha)
    radius: int       # raio em células
    max_food: int     # capacidade máxima
    state: PatchState = "growing"
    age: int = 0      # steps no estado actual
    # parâmetros de ciclo (fracções de max_food por step)
    grow_rate: float = 0.05
    deplete_thr: float = 0.15    # fracção mínima antes de recuperar
    recover_rate: float = 0.02
    peak_duration: int = 200     # steps em peak antes de passar a depleting


class PatchManager:
    """
    Gere N patches de comida com ciclo de vida grow→peak→deplete→recover.

    Opera directamente sobre o numpy grid do Mundo (uint8, 1=comida).
    Não guarda comida internamente — lê sempre o grid para saber o estado actual.
    """

    def __init__(
        self,
        grid: np.ndarray,
        n_patches: int = 20,
        patch_radius: int = 40,
        max_food_per_patch: int = 80,
        rng: np.random.Generator | None = None,
    ):
        self.grid = grid
        self.H, self.W = grid.shape
        self.rng = rng or np.random.default_rng()
        self._radius = patch_radius
        self._max_food = max_food_per_patch

        self.patches: list[FoodPatch] = [
            self._novo_patch() for _ in range(n_patches)
        ]
        # semear comida inicial nos patches
        for p in self.patches:
            self._semear_comida(*self._celulas_patch(p), n=max(1, p.max_food // 4))

    # -----------------------------------------------------------------------
    def _novo_patch(self) -> FoodPatch:
        r = self._radius
        cx = int(self.rng.integers(r, self.W - r))
        cy = int(self.rng.integers(r, self.H - r))
        return FoodPatch(cx=cx, cy=cy, radius=r, max_food=self._max_food)

    # -----------------------------------------------------------------------
    def _celulas_patch(self, p: FoodPatch) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (ys, xs) vectorizadas para todas as células dentro do raio."""
        y0 = max(0, p.cy - p.radius)
        y1 = min(self.H, p.cy + p.radius + 1)
        x0 = max(0, p.cx - p.radius)
        x1 = min(self.W, p.cx + p.radius + 1)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        mask = (gy - p.cy) ** 2 + (gx - p.cx) ** 2 <= p.radius ** 2
        return gy[mask], gx[mask]

    # -----------------------------------------------------------------------
    def _semear_comida(self, ys: np.ndarray, xs: np.ndarray, n: int) -> None:
        """Coloca n unidades de comida em células vazias aleatórias do patch."""
        idxs_vazias = np.where(self.grid[ys, xs] == 0)[0]
        if idxs_vazias.size == 0:
            return
        escolhidos = self.rng.choice(
            idxs_vazias, size=min(n, idxs_vazias.size), replace=False
        )
        self.grid[ys[escolhidos], xs[escolhidos]] = 1

    # -----------------------------------------------------------------------
    def step(self) -> None:
        """Actualiza todos os patches e escreve no grid. Chamado 1× por step do mundo."""
        for p in self.patches:
            p.age += 1
            ys, xs = self._celulas_patch(p)
            food_actual = int(self.grid[ys, xs].sum())

            if p.state == "growing":
                if food_actual >= p.max_food:
                    p.state = "peak"
                    p.age = 0
                else:
                    self._semear_comida(ys, xs, max(1, int(p.max_food * p.grow_rate)))

            elif p.state == "peak":
                if p.age > p.peak_duration:
                    p.state = "depleting"
                    p.age = 0

            elif p.state == "depleting":
                if food_actual <= int(p.max_food * p.deplete_thr):
                    p.state = "recovering"
                    p.age = 0

            elif p.state == "recovering":
                self._semear_comida(ys, xs, max(1, int(p.max_food * p.recover_rate)))
                if food_actual >= int(p.max_food * 0.5):
                    p.state = "growing"
                    p.age = 0

    # -----------------------------------------------------------------------
    def estado(self) -> dict:
        """Resumo dos estados dos patches (para logging/debug)."""
        counts = Counter(p.state for p in self.patches)
        return {
            "n_growing":    counts.get("growing", 0),
            "n_peak":       counts.get("peak", 0),
            "n_depleting":  counts.get("depleting", 0),
            "n_recovering": counts.get("recovering", 0),
            "total_food":   int((self.grid == 1).sum()),
        }
