"""
Simulação do mundo 2D e física do organismo.
Grid NumPy uint8: 0=vazio, 1=comida, 2=obstáculo
Coordenadas: origem (0,0) canto superior esquerdo, X=coluna, Y=linha.
"""
from __future__ import annotations

import numpy as np
import torch

from .vision import calcular_visao, visao_para_intensidades
from .olfato import calcular_olfato, RAIO_OLFATO_POR_FASE, RAIO_OLFATO
from .feromona import (
    actualizar_feromona, depositar_feromona, calcular_feromona_sectores,
)
from .predador import Predador, spawn_predadores, PREDADORES_POR_N
from .curriculum import CurriculumConfig, FASES

# Acções disponíveis
ACCOES = ["avancar", "virar_esquerda", "virar_direita", "parar"]
ACCAO_IDX = {a: i for i, a in enumerate(ACCOES)}

# Fase B — Senescência (sincronizado com reproduction.py)
_VIDA_MAX = 5000          # steps máximos de vida (morte garantida)
_SENESC_VEL = 0.30        # redução máxima de velocidade ao fim da vida (30%)


class Organismo:
    """Estado físico do organismo."""

    def __init__(
        self,
        x: int,
        y: int,
        angulo: float,
        energia_inicial: float = 200.0,
        id: int = 0,       # identificador único na população
        geracao: int = 0,  # geração (0=base_v1, 1=filhos, etc.)
    ):
        self.x = x
        self.y = y
        self.angulo = angulo % 360.0      # 0–359°
        self.energia = energia_inicial
        self.energia_max = 200.0          # v62: revertido v57/v58 (200=initial → cap sempre; sem amplify advance reward)
        self.vivo = True
        self.id = id
        self.geracao = geracao
        # v2.0 — loop sensoriomotor
        self.last_action: int = 0                              # acção executada no step anterior
        self.last_visual: torch.Tensor = torch.zeros(15)      # intensidades visuais do step anterior
        self.visual_actual: torch.Tensor = torch.zeros(15)    # intensidades visuais do step actual
        # Fase A — evolução
        self.idade: int = 0               # steps vividos (para fertilidade e senescência)
        self.n_filhos: int = 0            # número de filhos gerados
        self.comida_lifetime: int = 0     # comida total comida na vida
        self.cooldown_rep: int = 0        # cooldown de reprodução (steps restantes)

    # -------------------------------------------------------------------
    def step(self, accao: str, mundo: np.ndarray) -> dict:
        """Executa uma acção e devolve info do step (recompensa, comeu, etc.)."""
        altura, largura = mundo.shape
        recompensa = 0.0
        comeu = False

        if accao == "avancar":
            self.energia -= 0.5
            # Fase B: senescência — velocidade degrada linearmente com a idade
            vel_factor = max(0.30, 1.0 - _SENESC_VEL * min(1.0, self.idade / _VIDA_MAX))
            if np.random.random() < vel_factor:
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
            self.comida_lifetime += 1

        # Morte por energia
        if self.energia <= 0.0:
            self.energia = 0.0
            self.vivo = False
            recompensa -= 5.0

        # Fase A: contadores de vida e cooldown de reprodução
        self.idade += 1
        if self.cooldown_rep > 0:
            self.cooldown_rep -= 1

        # Fase B: morte garantida por senescência
        if self.vivo and self.idade >= _VIDA_MAX:
            self.vivo = False
            recompensa -= 5.0

        return {"recompensa": recompensa, "comeu": comeu}


class Mundo:
    """
    Mundo 2D com curriculum learning.
    Fase 1: 200×200, 50 comidas estáticas, sem obstáculos.
    """

    def __init__(
        self,
        fase: int = 1,
        seed: int | None = None,
        n_organismos: int = 1,
        usar_patches: bool = False,
        usar_predador: bool = False,
    ):
        self.fase = fase
        self.seed = seed
        self.cfg: CurriculumConfig = FASES[fase]
        self.rng = np.random.default_rng(seed)
        self.n_organismos = n_organismos
        self.usar_patches = usar_patches
        self.usar_predador = usar_predador

        self.grid: np.ndarray = np.zeros(
            (self.cfg.altura, self.cfg.largura), dtype=np.uint8
        )
        # Fase C: grelha de feromonas (float32, mesma dimensão que grid de comida)
        self.feromona_grid: np.ndarray = np.zeros(
            (self.cfg.altura, self.cfg.largura), dtype=np.float32
        )
        # v2.0: lista de organismos (N ≥ 1); self.organismo é property para organismo[0]
        self.organismos: list[Organismo] = []
        # Fase C.2: predadores
        self.predadores: list[Predador] = []
        self.patch_manager = None
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

        # Organismos: posição inicial aleatória em célula vazia para cada um
        self.organismos = []
        for i in range(self.n_organismos):
            for _ in range(1000):
                ox = int(self.rng.integers(0, w))
                oy = int(self.rng.integers(0, h))
                if self.grid[oy, ox] == 0:
                    break
            oang = float(self.rng.integers(0, 24) * 15)
            self.organismos.append(Organismo(ox, oy, oang, id=i))

        # v2.0: patches de comida dinâmicos (ciclo grow→peak→deplete→recover)
        if self.usar_patches:
            from .patches import PatchManager
            self.patch_manager = PatchManager(
                self.grid,
                n_patches=max(5, self.cfg.num_comida // 80),
                patch_radius=max(20, min(80, self.cfg.largura // 15)),
                max_food_per_patch=max(30, self.cfg.num_comida // 10),
                rng=self.rng,
            )

        # Fase C.2: predadores
        self.predadores = []
        if self.usar_predador and self.n_organismos > 0:
            n_preds = max(1, self.n_organismos // PREDADORES_POR_N)
            self.predadores = spawn_predadores(n_preds, h, w, self.organismos, self.rng)

        self.step_actual = 0
        self.comida_comida = 0
        self.energia_gasta = 0.0
        self.feromona_grid[:] = 0.0

    # -------------------------------------------------------------------
    @property
    def organismo(self) -> Organismo:
        """Compatibilidade com trainer.py single-organism (devolve organismos[0])."""
        return self.organismos[0]

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
        # v2.0: guardar visual ANTES do movimento (para fluxo óptico no próximo step)
        org.last_visual = org.visual_actual.clone()

        energia_antes = org.energia

        info = org.step(accao, self.grid)
        recompensa = info["recompensa"]
        if info["comeu"]:
            self.comida_comida += 1
            depositar_feromona(self.feromona_grid, org.x, org.y)

        # v2.0: guardar acção executada
        org.last_action = ACCAO_IDX[accao]

        self.energia_gasta += max(0.0, energia_antes - org.energia + (30.0 if info["comeu"] else 0.0))
        self.step_actual += 1

        # Patches (v2.0): actualizar ciclo de vida da comida a cada step
        if self.patch_manager is not None:
            self.patch_manager.step()

        # Fase C: actualizar grelha de feromonas (decay + difusão)
        actualizar_feromona(self.feromona_grid)

        # Reaparecimento de comida (fases 2+)
        if self.cfg.reaparecimento_comida and self.step_actual % self.cfg.reaparecimento_comida == 0:
            self._reaparecimento_comida()

        obs = self._observacao()
        terminado = not org.vivo
        return obs, recompensa, terminado

    # -------------------------------------------------------------------
    @property
    def terminado(self) -> bool:
        """True quando todos os organismos da população estão mortos."""
        return all(not o.vivo for o in self.organismos)

    # -------------------------------------------------------------------
    def step_populacao(self, accoes: list[str]) -> list[dict]:
        """
        Executa 1 step para todos os organismos vivos (Mundo v2.0).

        Arbitração de comida: ordem aleatória → primeiro a chegar come.
        Patches actualizados 1× no início do step.
        Devolve lista de dicts por organismo: {comeu, morreu, obs}.
        """
        # 1. Actualizar patches (1× por step)
        if self.patch_manager is not None:
            self.patch_manager.step()

        # Fase C: actualizar grelha de feromonas (decay + difusão, 1× por step)
        actualizar_feromona(self.feromona_grid)

        # 2. Contabilizar step global
        self.step_actual += 1

        # 3. Guardar visual_actual → last_visual para todos os organismos vivos
        for org in self.organismos:
            if org.vivo:
                org.last_visual = org.visual_actual.clone()

        # 4. Ordem aleatória para arbitração de comida simultanea
        ordem = list(range(len(self.organismos)))
        self.rng.shuffle(ordem)

        resultados: list[dict | None] = [None] * len(self.organismos)
        for i in ordem:
            org = self.organismos[i]
            if not org.vivo:
                resultados[i] = {"comeu": False, "morreu": True, "obs": None}
                continue

            energia_antes = org.energia
            info = org.step(accoes[i], self.grid)
            energia_gasta_step = max(0.0, energia_antes - org.energia + (30.0 if info["comeu"] else 0.0))
            self.energia_gasta += energia_gasta_step
            if info["comeu"]:
                self.comida_comida += 1
                # Fase C: depositar feromona quando organismo come
                depositar_feromona(self.feromona_grid, org.x, org.y)

            org.last_action = ACCAO_IDX[accoes[i]]
            obs = self._observacao(org)
            resultados[i] = {
                "comeu": info["comeu"],
                "morreu": not org.vivo,
                "obs": obs,
                "recompensa_base": info["recompensa"],
            }

        # Fase C.2: predadores — movem-se e matam organismos por contacto
        h_g, w_g = self.grid.shape
        for pred in self.predadores:
            mortos_pred = pred.step(self.organismos, h_g, w_g)
            for idx in mortos_pred:
                if resultados[idx] is not None:
                    resultados[idx]["morreu"] = True  # type: ignore[index]
                    resultados[idx]["recompensa_base"] = (  # type: ignore[index]
                        resultados[idx].get("recompensa_base", 0.0) - 5.0  # type: ignore[index]
                    )

        # Reaparecimento de comida (fases 2+ sem patches)
        if self.patch_manager is None and self.cfg.reaparecimento_comida and \
                self.step_actual % self.cfg.reaparecimento_comida == 0:
            self._reaparecimento_comida()

        return resultados  # type: ignore[return-value]

    # -------------------------------------------------------------------
    def _observacao(self, org: Organismo | None = None) -> dict:
        if org is None:
            org = self.organismo
        visao = calcular_visao(self.grid, org.x, org.y, org.angulo)
        # v2.0: actualizar intensidades visuais no organismo (usadas pelo encoder v2.0)
        org.visual_actual = torch.tensor(
            visao_para_intensidades(visao), dtype=torch.float32
        )
        # v4.0: raio por fase — olfato ultrapassa sempre o alcance visual (~15-20 células)
        _raio = RAIO_OLFATO_POR_FASE.get(self.fase, RAIO_OLFATO)
        olfato = calcular_olfato(self.grid, org.x, org.y, org.angulo, raio=_raio)
        # Fase C: feromonas
        feromona = calcular_feromona_sectores(self.feromona_grid, org.x, org.y, org.angulo)
        return {
            "pos_x": org.x,
            "pos_y": org.y,
            "angulo": org.angulo,
            "energia_actual": org.energia,
            "energia_max": org.energia_max,
            "energia_norm": org.energia / org.energia_max,
            "visao": visao,
            "olfato": olfato,
            "feromona": feromona,
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
