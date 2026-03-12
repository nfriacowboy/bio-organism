"""
Simulação do mundo 2D e física do organismo.
Grid NumPy uint8: 0=vazio, 1=comida, 2=obstáculo
Coordenadas: origem (0,0) canto superior esquerdo, X=coluna, Y=linha.
"""
from __future__ import annotations

import numpy as np

from .vision import calcular_visao
from .curriculum import CurriculumConfig, FASES

# Acções disponíveis
ACCOES = ["avancar", "virar_esquerda", "virar_direita", "parar"]
ACCAO_IDX = {a: i for i, a in enumerate(ACCOES)}


class Organismo:
    """Estado físico do organismo."""

    def __init__(self, x: int, y: int, angulo: float, energia_inicial: float = 100.0):
        self.x = x
        self.y = y
        self.angulo = angulo % 360.0      # 0–359°
        self.energia = energia_inicial
        self.energia_max = 150.0
        self.vivo = True

    # -------------------------------------------------------------------
    def step(self, accao: str, mundo: np.ndarray) -> dict:
        """Executa uma acção e devolve info do step (recompensa, comeu, etc.)."""
        altura, largura = mundo.shape
        recompensa = 0.0
        comeu = False

        if accao == "avancar":
            self.energia -= 0.5
            nx = self.x + round(np.cos(np.radians(self.angulo)))
            ny = self.y + round(np.sin(np.radians(self.angulo)))
            # Colisão com bordas — parar sem mover
            if 0 <= nx < largura and 0 <= ny < altura:
                self.x, self.y = nx, ny

        elif accao == "virar_esquerda":
            self.energia -= 0.2
            self.angulo = (self.angulo - 15.0) % 360.0

        elif accao == "virar_direita":
            self.energia -= 0.2
            self.angulo = (self.angulo + 15.0) % 360.0

        elif accao == "parar":
            self.energia -= 0.05

        # Comer comida na posição actual
        if mundo[self.y, self.x] == 1:
            mundo[self.y, self.x] = 0
            self.energia = min(self.energia + 30.0, self.energia_max)
            recompensa += 1.0
            comeu = True

        # Morte por energia
        if self.energia <= 0.0:
            self.energia = 0.0
            self.vivo = False
            recompensa -= 5.0

        return {"recompensa": recompensa, "comeu": comeu}


class Mundo:
    """
    Mundo 2D com curriculum learning.
    Fase 1: 200×200, 50 comidas estáticas, sem obstáculos.
    """

    def __init__(self, fase: int = 1, seed: int | None = None):
        self.fase = fase
        self.seed = seed
        self.cfg: CurriculumConfig = FASES[fase]
        self.rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros(
            (self.cfg.altura, self.cfg.largura), dtype=np.uint8
        )
        self.organismo: Organismo | None = None
        self.step_actual = 0
        self.comida_comida = 0
        self.energia_gasta = 0.0
        self._posicoes_comida_iniciais: list[tuple[int, int]] = []

        self._inicializar()

    # -------------------------------------------------------------------
    def _inicializar(self) -> None:
        self.grid[:] = 0
        h, w = self.cfg.altura, self.cfg.largura

        # Colocar comida
        posicoes: set[tuple[int, int]] = set()
        while len(posicoes) < self.cfg.num_comida:
            x = int(self.rng.integers(0, w))
            y = int(self.rng.integers(0, h))
            posicoes.add((x, y))
        for x, y in posicoes:
            self.grid[y, x] = 1
        self._posicoes_comida_iniciais = list(posicoes)

        # Obstáculos (fases 3+)
        for _ in range(self.cfg.num_obstaculos):
            while True:
                x = int(self.rng.integers(0, w))
                y = int(self.rng.integers(0, h))
                if self.grid[y, x] == 0:
                    self.grid[y, x] = 2
                    break

        # Organismo: posição inicial aleatória em célula vazia
        while True:
            ox = int(self.rng.integers(0, w))
            oy = int(self.rng.integers(0, h))
            if self.grid[oy, ox] == 0:
                break
        oang = float(self.rng.integers(0, 24) * 15)   # múltiplo de 15°
        self.organismo = Organismo(ox, oy, oang)
        self.step_actual = 0
        self.comida_comida = 0
        self.energia_gasta = 0.0

    # -------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        """Reinicia o mundo. Devolve a observação inicial."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self._inicializar()
        return self._observacao()

    # -------------------------------------------------------------------
    def step(self, accao: str) -> tuple[dict, float, bool]:
        """
        Executa uma acção.
        Devolve (observacao, recompensa_total, terminado).
        """
        assert self.organismo is not None and self.organismo.vivo

        org = self.organismo
        energia_antes = org.energia

        info = org.step(accao, self.grid)
        recompensa = info["recompensa"]
        if info["comeu"]:
            self.comida_comida += 1

        self.energia_gasta += max(0.0, energia_antes - org.energia + (30.0 if info["comeu"] else 0.0))
        self.step_actual += 1

        # Reaparecimento de comida (fases 2+)
        if self.cfg.reaparecimento_comida and self.step_actual % self.cfg.reaparecimento_comida == 0:
            self._reaparecimento_comida()

        obs = self._observacao()
        terminado = not org.vivo
        return obs, recompensa, terminado

    # -------------------------------------------------------------------
    def _observacao(self) -> dict:
        org = self.organismo
        visao = calcular_visao(self.grid, org.x, org.y, org.angulo)
        return {
            "pos_x": org.x,
            "pos_y": org.y,
            "angulo": org.angulo,
            "energia_actual": org.energia,
            "energia_max": org.energia_max,
            "energia_norm": org.energia / org.energia_max,
            "visao": visao,
            "step": self.step_actual,
            "fase": self.fase,
        }

    # -------------------------------------------------------------------
    def _distancia_comida_mais_proxima(self) -> float | None:
        org = self.organismo
        posicoes = np.argwhere(self.grid == 1)   # [[y, x], ...]
        if len(posicoes) == 0:
            return None
        dists = np.sqrt((posicoes[:, 1] - org.x) ** 2 + (posicoes[:, 0] - org.y) ** 2)
        return float(dists.min())

    # -------------------------------------------------------------------
    def _reaparecimento_comida(self) -> None:
        h, w = self.cfg.altura, self.cfg.largura
        comida_actual = int(self.grid == 1).sum() if self.grid.dtype == np.uint8 else (self.grid == 1).sum()
        a_adicionar = self.cfg.num_comida - int((self.grid == 1).sum())
        for _ in range(max(0, a_adicionar)):
            for _ in range(100):   # tentativas máximas para evitar loop infinito
                x = int(self.rng.integers(0, w))
                y = int(self.rng.integers(0, h))
                if self.grid[y, x] == 0:
                    self.grid[y, x] = 1
                    break

    # -------------------------------------------------------------------
    def eficiencia(self) -> float:
        """comida / energia_gasta (proxy da eficiência do organismo)."""
        if self.energia_gasta == 0:
            return 0.0
        return self.comida_comida / self.energia_gasta
