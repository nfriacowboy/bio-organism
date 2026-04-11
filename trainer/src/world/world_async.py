"""
Mundo assíncrono — grid thread-safe para população de organismos autónomos.

Filosofia:
- Sem tick global. Cada organismo age no seu próprio ritmo.
- Comida: first-come-first-served (lock atómico no write).
- Scent: emitido pelo alimento, difunde em background (leitura sem lock).
- Posições: dict partilhado com RLock para writes.

Organismos lêem food_grid e scent_grid sem lock — eventual consistency
é aceitável para decisões sensoriais. Apenas consumo de comida é atómico.
"""
from __future__ import annotations

import threading
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from .patches import PatchManager
from .olfato import calcular_olfato, RAIO_OLFATO_POR_FASE, RAIO_OLFATO
from .feromona import calcular_feromona_sectores
from .vision import calcular_visao, visao_para_intensidades

# Tamanho do mundo e densidade de comida por fase
_MUNDO_POR_FASE = {1: (200, 200), 2: (500, 500), 3: (1000, 1000)}
_COMIDA_POR_FASE = {1: 1500, 2: 5000, 3: 1500}


# ---------------------------------------------------------------------------
# Estado do organismo (partilhado entre OrganismoWorker e AsyncWorld)
# ---------------------------------------------------------------------------

@dataclass
class OrgState:
    """
    Estado local do organismo. Armazenado no mundo para visão/colisão.
    duck-type compatível com Organismo para encode_full_state e calcular_visao.
    """
    org_id:          int
    x:               int
    y:               int
    angulo:          float
    energia:         float
    energia_max:     float
    historico_accoes: list   = field(default_factory=lambda: [0, 0, 0, 0])
    visual_actual:   object  = None      # tensor actualizado pelo encoder
    last_visual:     object  = field(default_factory=lambda: torch.zeros(15))
    last_action:     int     = 0         # acção executada no step anterior
    vivo:            bool    = True
    idade:           int     = 0
    geracao:         int     = 0


# ---------------------------------------------------------------------------
# AsyncWorld
# ---------------------------------------------------------------------------

class AsyncWorld:
    """
    Grid thread-safe para simulação assíncrona multi-organismo.

    food_grid  : numpy float32 [H, W] — writes com _food_lock
    scent_grid : numpy float32 [H, W] — actualizado por ScentUpdater,
                 leitura sem lock (eventual consistency)
    _org_states: dict org_id → OrgState, writes com _pos_lock
    """

    def __init__(
        self,
        fase: int = 1,
        usar_patches: bool = True,
        rng: Optional[np.random.Generator] = None,
        world_size: Optional[int] = None,
        n_food: Optional[int] = None,
        n_patches: Optional[int] = None,
        patch_radius: Optional[int] = None,
    ):
        self.fase        = fase
        self.usar_patches = usar_patches
        self.rng         = rng or np.random.default_rng()

        if world_size is not None:
            self.W, self.H = world_size, world_size
        else:
            self.W, self.H = _MUNDO_POR_FASE[fase]

        self._n_food_target  = n_food    if n_food    is not None else _COMIDA_POR_FASE[fase]
        self._n_patches      = n_patches if n_patches is not None else None
        self._patch_radius   = patch_radius if patch_radius is not None else 40

        # Grids partilhadas
        self.food_grid  = np.zeros((self.H, self.W), dtype=np.float32)
        self.scent_grid = np.zeros((self.H, self.W), dtype=np.float32)

        # Locks
        self._food_lock = threading.Lock()
        self._pos_lock  = threading.RLock()

        # Registo de organismos
        self._org_states: dict[int, OrgState] = {}

        # Métricas globais
        self._total_comida_comida = 0
        self._metrics_lock = threading.Lock()

        # Patches dinâmicos
        self._patch_manager: Optional[PatchManager] = None

        self._init_food()

    # --- Inicialização ---

    def _init_food(self) -> None:
        if self.usar_patches:
            n_patches    = self._n_patches    if self._n_patches    is not None else max(4, self._n_food_target // 50)
            patch_radius = self._patch_radius
            # max_food proporcional à área do patch (~32% fill) para sinal olfativo forte
            max_food_per_patch = int(patch_radius ** 2)
            self._patch_manager = PatchManager(
                self.food_grid,
                n_patches=n_patches,
                patch_radius=patch_radius,
                max_food_per_patch=max_food_per_patch,
                rng=self.rng,
            )
        else:
            n_food = self._n_food_target
            placed = 0
            while placed < n_food:
                x = int(self.rng.integers(0, self.W))
                y = int(self.rng.integers(0, self.H))
                if self.food_grid[y, x] == 0:
                    self.food_grid[y, x] = 1.0
                    placed += 1

    # --- Registo de organismos ---

    def register_organism(
        self,
        org_id:      int,
        energia_max: float = 200.0,
        geracao:     int   = 0,
        near_xy:     Optional[tuple[int, int]] = None,
        near_radius: int   = 60,
    ) -> OrgState:
        """Regista novo organismo. Se near_xy dado, tenta spawnar perto desse ponto."""
        with self._pos_lock:
            placed = False
            if near_xy is not None:
                cx, cy = near_xy
                for _ in range(300):
                    dx = int(self.rng.integers(-near_radius, near_radius + 1))
                    dy = int(self.rng.integers(-near_radius, near_radius + 1))
                    x  = max(0, min(self.W - 1, cx + dx))
                    y  = max(0, min(self.H - 1, cy + dy))
                    if self.food_grid[y, x] == 0:
                        placed = True
                        break
            if not placed:
                for _ in range(1000):
                    x = int(self.rng.integers(0, self.W))
                    y = int(self.rng.integers(0, self.H))
                    if self.food_grid[y, x] == 0:
                        break
            angulo = float(int(self.rng.integers(0, 8)) * 45)
            state = OrgState(
                org_id=org_id, x=x, y=y, angulo=angulo,
                energia=energia_max, energia_max=energia_max,
                historico_accoes=[0, 0, 0, 0],
                vivo=True, idade=0, geracao=geracao,
            )
            self._org_states[org_id] = state
        return state

    def deregister_organism(self, org_id: int) -> None:
        with self._pos_lock:
            self._org_states.pop(org_id, None)

    # --- Consumo de comida (atómico) ---

    def try_eat(self, x: int, y: int) -> bool:
        """
        First-come-first-served. Atómico via lock.
        Devolve True se comeu, False se já não havia comida.
        Reaparece automaticamente uma comida noutro local.
        """
        with self._food_lock:
            if self.food_grid[y, x] > 0:
                self.food_grid[y, x] = 0.0
                self._respawn_one_unsafe()
                with self._metrics_lock:
                    self._total_comida_comida += 1
                return True
        return False

    def _respawn_one_unsafe(self) -> None:
        """Spawna uma comida em célula vazia. Chamar com _food_lock já held.
        Com patches activos, o PatchManager.step() trata da reposição — não spawnar aleatório."""
        if self._patch_manager is not None:
            return   # patches gerem a reposição via step()
        for _ in range(200):
            x = int(self.rng.integers(0, self.W))
            y = int(self.rng.integers(0, self.H))
            if self.food_grid[y, x] == 0:
                self.food_grid[y, x] = 1.0
                return

    # --- Movimento ---

    def move_organism(self, state: OrgState, dx: int, dy: int) -> tuple[int, int]:
        """Move organismo respeitando fronteiras. Actualiza state in-place."""
        nx = max(0, min(self.W - 1, state.x + dx))
        ny = max(0, min(self.H - 1, state.y + dy))
        with self._pos_lock:
            state.x, state.y = nx, ny
        return nx, ny

    # --- Observação ---

    def get_observation(self, state: OrgState) -> dict:
        """
        Compõe observação completa para o organismo.
        food_grid e scent_grid lidos sem lock (eventual consistency).
        """
        # Referências às grids partilhadas (leitura numpy é thread-safe via GIL)
        food_snap  = self.food_grid
        scent_snap = self.scent_grid

        visao   = calcular_visao(food_snap, state.x, state.y, state.angulo)
        # visual_actual: tensor de intensidades usado pelo encoder
        state.visual_actual = torch.tensor(
            visao_para_intensidades(visao), dtype=torch.float32
        )
        _raio   = RAIO_OLFATO_POR_FASE.get(self.fase, RAIO_OLFATO)
        olfato  = calcular_olfato(food_snap, state.x, state.y, state.angulo, raio=_raio)
        feromona = calcular_feromona_sectores(scent_snap, state.x, state.y, state.angulo)

        return {
            "pos_x":        state.x,
            "pos_y":        state.y,
            "angulo":       state.angulo,
            "energia_actual": state.energia,
            "energia_max":  state.energia_max,
            "energia_norm": state.energia / state.energia_max,
            "visao":        visao,
            "olfato":       olfato,
            "feromona":     feromona,
            "fase":         self.fase,
        }

    def step_patches(self) -> None:
        """Actualiza ciclo de vida dos patches. Chamar 1× por world-step quando usar_patches=True."""
        if self._patch_manager is not None:
            with self._food_lock:
                self._patch_manager.step()

    # --- Métricas ---

    @property
    def n_food(self) -> int:
        return int((self.food_grid > 0).sum())

    @property
    def total_comida_comida(self) -> int:
        with self._metrics_lock:
            return self._total_comida_comida

    @property
    def n_organismos_vivos(self) -> int:
        with self._pos_lock:
            return len(self._org_states)



# ---------------------------------------------------------------------------
# ScentUpdater — background thread
# ---------------------------------------------------------------------------

class ScentUpdater:
    """
    Thread de background que mantém o scent_grid actualizado.

    O scent é uma propriedade do alimento — cada célula de comida emite
    um sinal químico que difunde pelo espaço. Organismos detectam o gradiente
    direccional para localizar comida mesmo antes de a verrem.

    Usa gaussian blur sobre food_grid (scipy) para simular difusão.
    Actualiza world.scent_grid in-place — organismos lêem sem lock.
    """

    def __init__(self, world: AsyncWorld, update_interval_s: float = 0.2, sigma: float = 3.0):
        self.world    = world
        self.interval = update_interval_s
        self.sigma    = sigma
        self._running = False
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="ScentUpdater"
        )

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        try:
            from scipy.ndimage import gaussian_filter
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        while self._running:
            food = self.world.food_grid   # leitura sem lock (eventual consistency)

            if _has_scipy:
                blurred = gaussian_filter(food, sigma=self.sigma)
            else:
                # Fallback: 4-neighbor diffusion simples
                blurred = food.copy()
                blurred[1:-1, 1:-1] += 0.1 * (
                    food[:-2, 1:-1] + food[2:, 1:-1] +
                    food[1:-1, :-2] + food[1:-1, 2:]
                )

            # Normalizar [0, 1]
            mx = blurred.max()
            if mx > 1e-9:
                blurred /= mx

            # Actualizar in-place (write atómico para numpy array)
            self.world.scent_grid[:] = blurred

            time.sleep(self.interval)
