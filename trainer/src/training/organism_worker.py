"""
Organismo autónomo — thread independente por organismo.

Cada organismo tem o seu clock interno e age no seu próprio ritmo:
  1. Observa o mundo (leitura do grid partilhado)
  2. Submete pedido de inferência ao servidor GPU (batched com outros)
  3. Aplica a acção ao mundo (movimento, comer)
  4. Aprende via R-STDP com as activações recebidas do servidor
  5. Dorme (senescência: organismos velhos dormem mais)

Senescência: o clock abranda naturalmente com a idade.
  - Jovem (idade=0): passo a cada _BASE_STEP_MS ms
  - Velho (idade=5000): passo a cada _BASE_STEP_MS * (1 + _SENESC_FACTOR) ms

Morte: energia <= 0 OU idade >= VIDA_MAX.
Após morte, callbacks notificam o trainer para spawnar descendente.
"""
from __future__ import annotations

import math
import random
import threading
import time
from typing import Callable, Optional

import torch

from snn.encoding import encode_full_state
from snn.network import T_STEPS
from snn.kan_policy import KANPolicy, observacao_para_estado_kan
from snn.learning import RSTDPLearner
from world.world_async import AsyncWorld, OrgState

# --- Constantes ---
_BASE_STEP_MS   = 1.0      # ms por step quando jovem
_SENESC_FACTOR  = 2.0      # factor de abrandamento máximo (3× mais lento no fim da vida)
_VIDA_MAX       = 5000     # steps máximos de vida

_EPS_INICIO     = 0.90
_EPS_DECAY      = 0.990
_EPS_FLOOR      = 0.15

_ENERGIA_COMIDA    = 30.0  # energia ao comer
_ENERGIA_AVANCAR   = 0.5   # custo de avançar
_ENERGIA_VIRAR     = 0.1   # custo de virar
_ENERGIA_PARAR     = 0.05  # custo de parar

_DELTA_ANGULO   = {1: -45.0, 2: +45.0}  # virar_esq / virar_dir
_ACCAO_NOMES    = ["avancar", "virar_esquerda", "virar_direita", "parar"]

_STEPS_SEM_COMER_MAX = 100   # cap para stats KAN


class OrganismoWorker:
    """
    Thread autónoma por organismo.

    O organismo é responsável pela sua própria existência:
    não precisa de sincronizar com outros organismos para agir.
    Partilha apenas o mundo (grid de comida, scent) e o servidor de inferência.
    """

    def __init__(
        self,
        org_id:   int,
        world:    AsyncWorld,
        server,   # InferenceServer
        snn,      # OrganismoSNN
        kan:      KANPolicy,
        learner:  RSTDPLearner,
        device:   torch.device,
        geracao:  int = 0,
        ep_global: int = 0,
    ):
        self.org_id  = org_id
        self.world   = world
        self.server  = server
        self.snn     = snn
        self.kan     = kan
        self.learner = learner
        self.device  = device
        self.geracao = geracao

        # Estado do organismo no mundo
        self.state: OrgState = world.register_organism(
            org_id, energia_max=200.0, geracao=geracao
        )

        # Epsilon-greedy
        self._eps = max(_EPS_FLOOR, _EPS_INICIO * (_EPS_DECAY ** ep_global))

        # Métricas por lifetime
        self._lifetime_steps    = 0
        self._lifetime_food     = 0
        self._lifetime_reward   = 0.0
        self._steps_sem_comer   = 0
        self._janela_comida:    list = []

        # Thread
        self.alive   = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"Org-{org_id}",
        )

        # Callbacks invocados ao morrer
        self._on_death_callbacks: list[Callable] = []

    # --- API pública ---

    def start(self) -> None:
        self._thread.start()

    def on_death(self, callback: Callable) -> None:
        """Registar callback chamado quando o organismo morre."""
        self._on_death_callbacks.append(callback)

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    # --- Loop principal ---

    def _run(self) -> None:
        self.learner.reset_traces()

        while self.alive:
            # 1. Observar mundo
            obs = self.world.get_observation(self.state)

            # 2. KAN update (thread-local, rápido)
            stats_kan = {
                "comida_vista_50steps": len(self._janela_comida),
                "distancia_media_vista": 0.5,
                "steps_sem_comer": min(self._steps_sem_comer, _STEPS_SEM_COMER_MAX),
            }
            estado_kan = observacao_para_estado_kan(obs, stats_kan, device=self.device)
            self.kan.actualizar_snn(self.snn, estado_kan)

            # 3. Codificar estado
            x = encode_full_state(self.state, obs, T=T_STEPS, device=self.device)

            # 4. Inferência no servidor GPU (batched com todos os organismos vivos)
            result = self.server.request(self.snn, x)

            # 5. Epsilon-greedy
            if random.random() < self._eps:
                accao_idx = random.randint(0, 3)
            else:
                accao_idx = result.action

            # 6. Aplicar acção
            reward = self._aplicar_accao(accao_idx)

            # 7. R-STDP com activações pré-calculadas pelo servidor
            if not self._morreu():
                self.learner.step_with_precomputed_spikes(
                    x, result.h1, result.h2, result.out,
                    reward, accao=accao_idx,
                )

            # 8. Actualizar last_visual / last_action para próximo step
            if self.state.visual_actual is not None:
                self.state.last_visual = self.state.visual_actual.clone()
            self.state.last_action = accao_idx

            # 9. Actualizar estado
            self.state.idade       += 1
            self._lifetime_steps   += 1
            self._eps               = max(_EPS_FLOOR, self._eps * _EPS_DECAY)
            self._lifetime_reward  += reward
            self._janela_comida     = self._janela_comida[-50:]

            # 9. Verificar morte
            if self._morreu():
                break

            # 10. Sleep proporcional à idade (senescência)
            time.sleep(self._step_delay())

        self._die()

    # --- Acções ---

    def _aplicar_accao(self, accao_idx: int) -> float:
        """Aplica a acção ao mundo e devolve recompensa."""
        state = self.state

        if accao_idx == 0:  # avancar
            state.energia -= _ENERGIA_AVANCAR
            dx = round(math.cos(math.radians(state.angulo)))
            dy = round(math.sin(math.radians(state.angulo)))
            self.world.move_organism(state, dx, dy)

            # Comer: atómico — first-come-first-served
            comeu = self.world.try_eat(state.x, state.y)
            if comeu:
                state.energia = min(state.energia_max, state.energia + _ENERGIA_COMIDA)
                self._lifetime_food  += 1
                self._steps_sem_comer = 0
                self._janela_comida.append(1)
                reward = 1.0
            else:
                self._steps_sem_comer += 1
                reward = -0.01

        elif accao_idx in (1, 2):  # virar
            state.angulo  = (state.angulo + _DELTA_ANGULO[accao_idx]) % 360
            state.energia -= _ENERGIA_VIRAR
            self._steps_sem_comer += 1
            reward = 0.0

        else:  # parar
            state.energia -= _ENERGIA_PARAR
            self._steps_sem_comer += 1
            reward = -0.02

        state.historico_accoes = (state.historico_accoes + [accao_idx])[-4:]
        return reward

    # --- Senescência ---

    def _step_delay(self) -> float:
        """
        Clock interno: abrandamento proporcional à idade.
        Jovem: _BASE_STEP_MS ms/step
        Velho (VIDA_MAX): _BASE_STEP_MS * (1 + _SENESC_FACTOR) ms/step
        """
        age_frac = min(1.0, self.state.idade / _VIDA_MAX)
        ms = _BASE_STEP_MS * (1.0 + _SENESC_FACTOR * age_frac)
        return ms / 1000.0

    # --- Morte ---

    def _morreu(self) -> bool:
        return self.state.energia <= 0 or self.state.idade >= _VIDA_MAX

    def _die(self) -> None:
        self.alive = False
        self.world.deregister_organism(self.org_id)
        for cb in self._on_death_callbacks:
            try:
                cb(self)
            except Exception:
                pass

    # --- Métricas ---

    @property
    def stats(self) -> dict:
        return {
            "org_id":    self.org_id,
            "geracao":   self.geracao,
            "idade":     self.state.idade,
            "comida":    self._lifetime_food,
            "recompensa": self._lifetime_reward,
            "steps":     self._lifetime_steps,
        }
