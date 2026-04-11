"""
WorldRunner — gere um mundo completo numa única thread.

Design: uma thread por mundo, N organismos stepped sequencialmente dentro da thread.
O GPU batching ocorre via InferenceServer partilhado por todos os WorldRunners.

Diferenças para OrganismoWorker (per-thread):
- Sem threads por organismo → 1 thread por mundo (mais eficiente para 100 mundos)
- Reprodução sexual por proximidade MID-LIFE (organismos saudáveis e próximos reproduzem-se sem morrer)
- Reposição na morte: substitui pelo descendente dos melhores pais
- Logging directo para tabelas 007_multi_world (organismos_lifetimes, mundos_log, posicoes_snapshot)
"""
from __future__ import annotations

import math
import json
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from snn import OrganismoSNN
from snn.encoding import encode_full_state
from snn.network import T_STEPS, batched_forward_multi_snn
from snn.kan_policy import KANPolicy, observacao_para_estado_kan
from snn.learning import RSTDPLearner
from world.world_async import AsyncWorld, OrgState
from world.reproduction import (
    get_genome, crossover, mutate, set_genome, herdar_epigenetica,
    calcular_diversidade, cosine_sim_snn,
    INBREEDING_COS_THR, MUTACAO_SIGMA_DEFAULT,
)
from training.db_writer import DBWriter

# ---------------------------------------------------------------------------
# Constantes por organismo
# ---------------------------------------------------------------------------

_EPS_INICIO         = 0.90
_EPS_DECAY          = 0.9995   # mais lento para lifespan maior
_EPS_FLOOR          = 0.15

_ENERGIA_COMIDA     = 30.0
_ENERGIA_AVANCAR    = 0.5
_ENERGIA_VIRAR      = 0.1
_ENERGIA_PARAR      = 0.05
_DELTA_ANGULO       = {1: -45.0, 2: +45.0}

# Lookup table para movimento: angulo → (dx, dy) já arredondados
# Angulo é sempre múltiplo de 45° (só muda por _DELTA_ANGULO) → 8 possibilidades
_TRIG_LUT: dict[int, tuple[int, int]] = {
    a: (round(math.cos(math.radians(a))), round(math.sin(math.radians(a))))
    for a in range(0, 360, 45)
}

# Reprodução sexual mid-life
_MATING_RADIUS          = 50      # células de distância Manhattan máxima (não-alfa)
_MATING_MIN_ENERGY_NORM = 0.75   # energia mínima para reproduzir
_MATING_MIN_AGE         = 25     # steps mínimos
_MATING_COOLDOWN        = 30     # steps de cooldown após cada reprodução

# Dinâmicas alfa / excursão tribal
_ALPHA_EXCURSION_PROB   = 0.20   # P(alfa tenta acasalar fora do território por check)
_CONFLICT_ENERGY_LOSS   = 40.0   # energia perdida pelo alfa perdedor num conflito territorial


# ---------------------------------------------------------------------------
# Helper territorial (para patches)
# ---------------------------------------------------------------------------

def _nearest_patch_id(x: int, y: int, patches) -> int:
    """Retorna o índice do patch mais próximo de (x,y)."""
    best_id, best_d2 = 0, float("inf")
    for i, p in enumerate(patches):
        d2 = (x - p.cx) ** 2 + (y - p.cy) ** 2
        if d2 < best_d2:
            best_d2, best_id = d2, i
    return best_id


# ---------------------------------------------------------------------------
# Config do mundo
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:
    mundo_id:       int
    world_size:     int   = 500
    n_food:         int   = 1250    # usado só sem patches; com patches controlado por n_patches
    n_patches:      int   = 8       # nº de patches de comida (variável do sweep)
    patch_radius:   int   = 40      # raio de cada patch em células
    usar_patches:   bool  = True    # patches dinâmicos (grow→peak→deplete→recover)
    n_org_inicial:  int   = 10
    vida_max:       int   = 100
    max_pop:        int   = 25
    seed:           int   = 0
    energia_inicial: float = 200.0
    energia_max:    float  = 300.0
    base_snn_path:  str   = ""      # path para snn_ep*.pt de treino base (ex: v80p)

    @property
    def food_pct(self) -> float:
        # estimativa: n_patches × π×r² × 0.5 (meia capacidade média) / total
        import math
        cells_per_patch = math.pi * self.patch_radius ** 2 * 0.5
        return min(1.0, self.n_patches * cells_per_patch / (self.world_size ** 2))


# ---------------------------------------------------------------------------
# Entrada por organismo (sem thread própria)
# ---------------------------------------------------------------------------

@dataclass
class OrgEntry:
    org_id:       int
    state:        OrgState
    snn:          object   # OrganismoSNN
    kan:          KANPolicy
    learner:      RSTDPLearner
    geracao:      int
    eps:          float
    x_nasc:       int
    y_nasc:       int
    nasc_em_s:    float
    lifetime_food:       int   = 0
    lifetime_steps:      int   = 0
    steps_sem_comer:     int   = 0
    mating_cooldown:     int   = 0
    n_filhos:            int   = 0
    is_cross_territory:  bool  = False   # nasceu de excursão cross-territorial do alfa
    janela_comida:       deque = field(default_factory=lambda: deque(maxlen=50))


# ---------------------------------------------------------------------------
# WorldRunner
# ---------------------------------------------------------------------------

class WorldRunner(threading.Thread):
    """
    Thread gerente de um mundo.

    Loop principal:
      1. Para cada organismo vivo: observa → inferência → acção → aprende → envelhece
      2. Verifica mortes → handle lifetime record → substitui
      3. Verifica reprodução por proximidade (mid-life)
      4. Atualiza scent inline (scipy gaussian_filter a cada N steps)
      5. Loga métricas periódicas para DB
    """

    LOG_INTERVAL_S   = 30.0   # loga mundos_log a cada N segundos
    SNAPSHOT_INTERVAL_S = 30.0  # loga posicoes_snapshot a cada N segundos (era 120 — causava 150s freeze no canvas)
    SCENT_STEPS      = 20     # actualiza patches a cada N world-steps (era 200 — demasiado lento)

    def __init__(
        self,
        config:        WorldConfig,
        server,                      # InferenceServer partilhado
        device:        torch.device,
        simulacao_id:  str,
        db_writer:     Optional[DBWriter] = None,
    ):
        super().__init__(daemon=True, name=f"World-{config.mundo_id:03d}")
        self.config       = config
        self.server       = server
        self.device       = device
        self.simulacao_id = simulacao_id
        self._db_writer   = db_writer

        rng = np.random.default_rng(config.seed)
        self.world = AsyncWorld(
            fase         = 3,          # fase=3 → raio_olfato=100 (v80p foi treinado assim)
            usar_patches = config.usar_patches,
            rng          = rng,
            world_size   = config.world_size,
            n_food       = config.n_food,
            n_patches    = config.n_patches    if config.usar_patches else None,
            patch_radius = config.patch_radius if config.usar_patches else None,
        )
        self._rng = rng

        # Scent (scipy, aplicado inline)
        self._scent_sigma = 3.0
        try:
            from scipy.ndimage import gaussian_filter as _gf
            self._gauss    = _gf
            self._has_scipy = True
        except ImportError:
            self._has_scipy = False

        # Estado
        self._next_id   = 0
        self._organisms: list[OrgEntry] = []
        self._lifetimes: list[dict]     = []    # buffer para flush DB
        self._step      = 0
        self._t_start   = 0.0
        self._gen_max   = 0
        self._nasc_desde_log  = 0
        self._mortes_desde_log = 0
        self._comida_desde_log = 0

        # Flag de paragem
        self._stop_event = threading.Event()

        # Métricas expostas ao trainer
        self.stats: dict = {}   # actualizadas a cada LOG_STEPS

    # -------------------------------------------------------------------------
    # Thread principal
    # -------------------------------------------------------------------------

    def run(self) -> None:
        self._t_start = time.monotonic()

        # Spawnar população inicial
        for _ in range(self.config.n_org_inicial):
            self._spawn(geracao=0)

        _t_last_log  = self._t_start
        _t_last_snap = self._t_start

        try:
            while not self._stop_event.is_set():
                self._world_step()
                self._step += 1

                # Reprodução mid-life (a cada 10 steps)
                if self._step % 10 == 0:
                    self._check_mating()

                # Scent update inline
                if self._step % self.SCENT_STEPS == 0:
                    self._update_scent()
                    if self.config.usar_patches:
                        self.world.step_patches()

                # Flush lifetimes para DB
                if len(self._lifetimes) >= 20:
                    self._flush_lifetimes()

                # Log periódico — baseado em tempo
                now = time.monotonic()
                if now - _t_last_log >= self.LOG_INTERVAL_S:
                    self._log_periodic(dt_s=now - _t_last_log)
                    _t_last_log = now

                if now - _t_last_snap >= self.SNAPSHOT_INTERVAL_S:
                    self._snapshot()
                    _t_last_snap = now

        except Exception as exc:
            import traceback
            print(f"[World-{self.config.mundo_id}] ERRO: {exc}")
            traceback.print_exc()
        finally:
            self._flush_lifetimes()
            self._log_periodic(dt_s=time.monotonic() - _t_last_log)

    def stop(self) -> None:
        self._stop_event.set()

    def get_patch_positions(self) -> list[dict]:
        """Retorna posições e raio de todos os patches criados neste mundo.

        Chamar antes de start() — patches são gerados em AsyncWorld.__init__.
        """
        pm = self.world._patch_manager
        if pm is None:
            return []
        return [{"cx": int(p.cx), "cy": int(p.cy), "radius": int(p.radius)} for p in pm.patches]

    # -------------------------------------------------------------------------
    # Step único do mundo
    # -------------------------------------------------------------------------

    def _world_step(self) -> None:
        alive = [o for o in self._organisms if o.state.vivo]

        # Extinção → respawn mínimo
        if not alive:
            for _ in range(max(2, self.config.n_org_inicial // 2)):
                self._spawn(geracao=0)
            return

        # ------------------------------------------------------------------
        # Fase 1 — Observar + KAN + codificar (CPU, sequential)
        # Recolhe obs e prepara inputs para todos os organismos antes de
        # tocar na GPU — separa env state collection de inferência.
        # ------------------------------------------------------------------
        phase1: list[tuple] = []  # (org, obs, x [T,1,45])
        for org in alive:
            obs = self.world.get_observation(org.state)
            stats_kan = {
                "comida_vista_50steps":  len(org.janela_comida),
                "distancia_media_vista": 0.5,
                "steps_sem_comer":       min(org.steps_sem_comer, 100),
            }
            estado_kan = observacao_para_estado_kan(obs, stats_kan, device=self.device)
            org.kan.actualizar_snn(org.snn, estado_kan)
            x = encode_full_state(org.state, obs, T=T_STEPS, device=self.device)
            phase1.append((org, obs, x))

        # ------------------------------------------------------------------
        # Fase 2 — Inferência GPU em batch único para todos os organismos
        # batched_forward_multi_snn: [N,T,45] → spike_counts [N,4],
        #   h1 [T,N,64], h2 [T,N,32], out [T,N,4]
        # Substitui N chamadas bloqueantes ao InferenceServer por 1 chamada.
        # ------------------------------------------------------------------
        snns   = [org.snn for org, _, _ in phase1]
        xs     = [x for _, _, x in phase1]
        x_batch = torch.stack([x.squeeze(1) for x in xs], dim=0)  # [N, T, 45]
        spike_counts_b, h1_b, h2_b, out_b = batched_forward_multi_snn(
            snns, x_batch, self.device
        )

        # ------------------------------------------------------------------
        # Fase 3 — Acção + ambiente (CPU, sequential — side-effects)
        # ------------------------------------------------------------------
        phase3: list[tuple] = []  # (org, x, accao, reward, idx)
        for i, (org, obs, x) in enumerate(phase1):
            if random.random() < org.eps:
                accao = random.randint(0, 3)
            else:
                accao = int(spike_counts_b[i].argmax().item())
            reward = self._apply_action(org, accao)
            self._comida_desde_log += (1 if reward >= 1.0 else 0)
            phase3.append((org, x, accao, reward, i))

        # ------------------------------------------------------------------
        # Fase 4 — R-STDP (CPU, sequential) + estado + detecção de morte
        # ------------------------------------------------------------------
        dead_this_step = []
        for org, x, accao, reward, i in phase3:
            if org.state.vivo:
                org.learner.step_with_precomputed_spikes(
                    x,
                    h1_b[:, i:i+1, :],
                    h2_b[:, i:i+1, :],
                    out_b[:, i:i+1, :],
                    reward, accao=accao,
                )
            if org.state.visual_actual is not None:
                org.state.last_visual = org.state.visual_actual.clone()
            org.state.last_action  = accao
            org.state.idade       += 1
            org.lifetime_steps    += 1
            org.eps                = max(_EPS_FLOOR, org.eps * _EPS_DECAY)
            if org.mating_cooldown > 0:
                org.mating_cooldown -= 1
            if org.state.energia <= 0 or org.state.idade >= self.config.vida_max:
                dead_this_step.append(org)

        for org in dead_this_step:
            self._handle_death(org)

    # -------------------------------------------------------------------------
    # Acção
    # -------------------------------------------------------------------------

    def _apply_action(self, org: OrgEntry, accao: int) -> float:
        state = org.state
        if accao == 0:  # avancar
            state.energia -= _ENERGIA_AVANCAR
            dx, dy = _TRIG_LUT.get(int(state.angulo) % 360, (0, 0))
            self.world.move_organism(state, dx, dy)
            comeu = self.world.try_eat(state.x, state.y)
            if comeu:
                state.energia = min(state.energia_max, state.energia + _ENERGIA_COMIDA)
                org.lifetime_food  += 1
                org.steps_sem_comer = 0
                org.janela_comida.append(1)
                return 1.0
            else:
                org.steps_sem_comer += 1
                return -0.01
        elif accao in (1, 2):  # virar
            state.angulo  = (state.angulo + _DELTA_ANGULO[accao]) % 360
            state.energia -= _ENERGIA_VIRAR
            org.steps_sem_comer += 1
            return 0.0
        else:  # parar
            state.energia -= _ENERGIA_PARAR
            org.steps_sem_comer += 1
            return -0.02

    # -------------------------------------------------------------------------
    # Morte + reposição
    # -------------------------------------------------------------------------

    def _handle_death(self, org: OrgEntry) -> None:
        self._log_death(org)
        # Reposição: spawn filho dos dois melhores pais vivos, perto do local de morte
        morte_xy = (org.state.x, org.state.y)
        self._spawn_replacement(near_xy=morte_xy)

    def _log_death(self, org: OrgEntry, causa_override: Optional[str] = None) -> None:
        """Regista a morte na tabela lifetimes e marca org.state.vivo=False."""
        elapsed = time.monotonic() - self._t_start
        if causa_override is not None:
            causa = causa_override
        else:
            causa = "fome" if org.state.energia <= 0 else "senescencia"
        food_rate = org.lifetime_food / max(1, org.state.idade)
        self._lifetimes.append({
            "simulacao_id":      self.simulacao_id,
            "mundo_id":          self.config.mundo_id,
            "org_id":            org.org_id,
            "geracao":           org.geracao,
            "x_nasc":            org.x_nasc,
            "y_nasc":            org.y_nasc,
            "x_morte":           org.state.x,
            "y_morte":           org.state.y,
            "idade":             org.state.idade,
            "comida_total":      org.lifetime_food,
            "food_rate":         food_rate,
            "nasc_em_s":         org.nasc_em_s,
            "mort_em_s":         elapsed,
            "causa_morte":       causa,
            "n_filhos":          org.n_filhos,
            "is_cross_territory": org.is_cross_territory,
        })
        org.state.vivo = False
        self.world.deregister_organism(org.org_id)
        self._mortes_desde_log += 1

    def _spawn_replacement(self, near_xy: Optional[tuple[int, int]] = None) -> None:
        alive = [o for o in self._organisms if o.state.vivo]
        if len(alive) < 2:
            self._spawn(geracao=0)
            return

        patches = (
            self.world._patch_manager.patches
            if self.world._patch_manager is not None else None
        )

        # Seleccionar pais do mesmo território que o local de morte
        if patches is not None and near_xy is not None:
            mort_terr = _nearest_patch_id(near_xy[0], near_xy[1], patches)
            cands_terr = [o for o in alive
                          if _nearest_patch_id(o.state.x, o.state.y, patches) == mort_terr]
            candidatos = cands_terr if len(cands_terr) >= 2 else alive
        else:
            candidatos = alive

        candidatos = sorted(candidatos, key=lambda o: o.lifetime_food, reverse=True)
        pai_a = pai_b = None
        for a in candidatos:
            for b in candidatos:
                if a is b:
                    continue
                if cosine_sim_snn(a.snn, b.snn) < INBREEDING_COS_THR:
                    pai_a, pai_b = a, b
                    break
            if pai_a:
                break
        if pai_a is None:
            pai_a, pai_b = random.sample(alive, 2)

        nova_g = max(pai_a.geracao, pai_b.geracao) + 1
        self._gen_max = max(self._gen_max, nova_g)
        self._spawn(geracao=nova_g, pai_a=pai_a, pai_b=pai_b, near_xy=near_xy)
        pai_a.n_filhos += 1
        pai_b.n_filhos += 1

    # -------------------------------------------------------------------------
    # Reprodução sexual mid-life por proximidade
    # -------------------------------------------------------------------------

    def _check_mating(self) -> None:
        alive = [o for o in self._organisms if o.state.vivo]
        at_cap = len(alive) >= self.config.max_pop

        patches = (
            self.world._patch_manager.patches
            if self.world._patch_manager is not None else None
        )

        # Identificar alfa por território usando TAXA de comida (food/idade) — evita
        # enviesamento para organismos mais velhos que apenas comeram mais por terem vivido mais.
        alpha_per_terr: dict[int, OrgEntry] = {}
        if patches is not None:
            for o in alive:
                t = _nearest_patch_id(o.state.x, o.state.y, patches)
                rate_o = o.lifetime_food / max(1, o.state.idade)
                cur = alpha_per_terr.get(t)
                if cur is None:
                    alpha_per_terr[t] = o
                else:
                    rate_cur = cur.lifetime_food / max(1, cur.state.idade)
                    if rate_o > rate_cur:
                        alpha_per_terr[t] = o

        for i, org_a in enumerate(alive):
            terr_a = _nearest_patch_id(org_a.state.x, org_a.state.y, patches) if patches else None
            is_alpha = patches is not None and alpha_per_terr.get(terr_a) is org_a

            if org_a.mating_cooldown > 0 and not is_alpha:
                continue
            ea = org_a.state.energia / org_a.state.energia_max
            if ea < _MATING_MIN_ENERGY_NORM or org_a.state.idade < _MATING_MIN_AGE:
                continue

            mated = False

            # --- Passo 1: acasalamento intra-território (só quando pop não está no limite) ---
            if not at_cap:
                for org_b in alive[i + 1:]:
                    if org_b.mating_cooldown > 0:
                        continue
                    eb = org_b.state.energia / org_b.state.energia_max
                    if eb < _MATING_MIN_ENERGY_NORM or org_b.state.idade < _MATING_MIN_AGE:
                        continue
                    if cosine_sim_snn(org_a.snn, org_b.snn) >= INBREEDING_COS_THR:
                        continue

                    if patches is not None:
                        terr_b = _nearest_patch_id(org_b.state.x, org_b.state.y, patches)
                        if terr_a != terr_b:
                            continue

                    dist = abs(org_a.state.x - org_b.state.x) + abs(org_a.state.y - org_b.state.y)
                    if dist > _MATING_RADIUS and not is_alpha:
                        continue

                    nova_g = max(org_a.geracao, org_b.geracao) + 1
                    self._gen_max = max(self._gen_max, nova_g)
                    mid_xy = (
                        (org_a.state.x + org_b.state.x) // 2,
                        (org_a.state.y + org_b.state.y) // 2,
                    )
                    self._spawn(geracao=nova_g, pai_a=org_a, pai_b=org_b, near_xy=mid_xy)
                    org_a.n_filhos += 1
                    org_b.n_filhos += 1
                    if not is_alpha:
                        org_a.mating_cooldown = _MATING_COOLDOWN
                    org_b.mating_cooldown = _MATING_COOLDOWN
                    at_cap = len([o for o in self._organisms if o.state.vivo]) >= self.config.max_pop
                    mated = True
                    break

            # --- Passo 2: excursão cross-territorial do alfa ---
            # Funciona MESMO quando pop está no limite: o organismo mais fraco do território
            # alvo é deslocado (causa_morte="deslocamento") para dar lugar ao filho híbrido.
            # Isto garante dinâmicas tribais activas durante toda a simulação.
            if (not mated and is_alpha and patches is not None
                    and len(patches) > 1
                    and random.random() < _ALPHA_EXCURSION_PROB):

                rate_a = org_a.lifetime_food / max(1, org_a.state.idade)

                for org_b in alive:
                    if org_b is org_a or org_b.mating_cooldown > 0:
                        continue
                    eb = org_b.state.energia / org_b.state.energia_max
                    if eb < _MATING_MIN_ENERGY_NORM or org_b.state.idade < _MATING_MIN_AGE:
                        continue
                    terr_b = _nearest_patch_id(org_b.state.x, org_b.state.y, patches)
                    if terr_b == terr_a:
                        continue
                    if cosine_sim_snn(org_a.snn, org_b.snn) >= INBREEDING_COS_THR:
                        continue

                    # Conflito com o alfa defensor do território B (usando food RATE)
                    alpha_b = alpha_per_terr.get(terr_b)
                    if alpha_b is not None and alpha_b is not org_b:
                        rate_b = alpha_b.lifetime_food / max(1, alpha_b.state.idade)
                        if rate_a >= rate_b:
                            alpha_b.state.energia = max(
                                1.0, alpha_b.state.energia - _CONFLICT_ENERGY_LOSS
                            )
                        else:
                            org_a.state.energia = max(
                                1.0, org_a.state.energia - _CONFLICT_ENERGY_LOSS
                            )
                            break  # excursão falhada

                    # Se pop no limite: deslocar organismo mais fraco (energia mínima) do terr B
                    if at_cap:
                        orgs_terr_b = [
                            o for o in alive
                            if o is not org_b
                            and _nearest_patch_id(o.state.x, o.state.y, patches) == terr_b
                        ]
                        if not orgs_terr_b:
                            continue  # não há quem deslocar — tentar outra fêmea
                        weakest = min(orgs_terr_b, key=lambda o: o.state.energia)
                        # Log da morte por deslocamento (sem _spawn_replacement — substituímos já)
                        self._log_death(weakest, causa_override="deslocamento")

                    # Acasalamento cross-territorial bem-sucedido
                    nova_g = max(org_a.geracao, org_b.geracao) + 1
                    self._gen_max = max(self._gen_max, nova_g)
                    mid_xy = (
                        (org_a.state.x + org_b.state.x) // 2,
                        (org_a.state.y + org_b.state.y) // 2,
                    )
                    self._spawn(
                        geracao=nova_g, pai_a=org_a, pai_b=org_b,
                        near_xy=mid_xy, is_cross_territory=True,
                    )
                    org_a.n_filhos += 1
                    org_b.n_filhos += 1
                    org_b.mating_cooldown = _MATING_COOLDOWN
                    # Alfa não entra em cooldown — pode tentar novamente no próximo check
                    break

    # -------------------------------------------------------------------------
    # Spawn
    # -------------------------------------------------------------------------

    def _load_base_snn(self, snn: object) -> None:
        """Carrega pesos base (ex: v80p) com padding para feromona (42→45 inputs)."""
        path = self.config.base_snn_path
        if not path:
            return
        try:
            sd = torch.load(path, map_location=self.device)
            # Compatibilidade: fc1.weight pode ter shape [64,42] se pré-feromona
            if "fc1.weight" in sd:
                w = sd["fc1.weight"]
                if w.shape[1] == 42:
                    pad = torch.zeros(w.shape[0], 3, device=self.device)
                    sd["fc1.weight"] = torch.cat([w, pad], dim=1)
            snn.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[World-{self.config.mundo_id}] base_snn load erro: {e}")

    def _spawn(
        self,
        geracao: int = 0,
        pai_a: Optional[OrgEntry] = None,
        pai_b: Optional[OrgEntry] = None,
        near_xy: Optional[tuple[int, int]] = None,
        is_cross_territory: bool = False,
    ) -> OrgEntry:
        self._next_id += 1
        org_id = self._next_id

        # Criar SNN/KAN/learner
        if pai_a is not None and pai_b is not None:
            g_a = get_genome(pai_a.snn, pai_a.kan)
            g_b = get_genome(pai_b.snn, pai_b.kan)
            g_f = crossover(g_a, g_b, self._rng)
            g_f = mutate(g_f, MUTACAO_SIGMA_DEFAULT, self._rng)
            snn_f = OrganismoSNN(device=self.device).to(self.device)
            kan_f = KANPolicy()
            set_genome(snn_f, kan_f, g_f, self.device)
            learner_f = RSTDPLearner(snn_f, device=self.device)
            herdar_epigenetica(learner_f, pai_a.learner, pai_b.learner)
        else:
            snn_f   = OrganismoSNN(device=self.device).to(self.device)
            kan_f   = KANPolicy()
            learner_f = RSTDPLearner(snn_f, device=self.device)
            # Carregar pesos base + mutação leve para diversidade inicial
            self._load_base_snn(snn_f)
            if self.config.base_snn_path:
                g = get_genome(snn_f, kan_f)
                g = mutate(g, MUTACAO_SIGMA_DEFAULT * 0.3, self._rng)
                set_genome(snn_f, kan_f, g, self.device)

        state = self.world.register_organism(
            org_id,
            energia_max=self.config.energia_max,
            geracao=geracao,
            near_xy=near_xy,
            near_radius=80,
        )
        learner_f.reset_traces()

        elapsed = time.monotonic() - self._t_start
        org = OrgEntry(
            org_id       = org_id,
            state        = state,
            snn          = snn_f,
            kan          = kan_f,
            learner      = learner_f,
            geracao      = geracao,
            eps          = _EPS_INICIO,
            x_nasc       = state.x,
            y_nasc       = state.y,
            nasc_em_s    = elapsed,
            is_cross_territory = is_cross_territory,
        )
        self._organisms.append(org)
        self._nasc_desde_log += 1
        return org

    # -------------------------------------------------------------------------
    # Scent inline
    # -------------------------------------------------------------------------

    def _update_scent(self) -> None:
        if self._has_scipy:
            blurred = self._gauss(self.world.food_grid, sigma=self._scent_sigma)
        else:
            food = self.world.food_grid
            blurred = food.copy()
            blurred[1:-1, 1:-1] += 0.1 * (
                food[:-2, 1:-1] + food[2:, 1:-1] +
                food[1:-1, :-2] + food[1:-1, 2:]
            )
        mx = blurred.max()
        if mx > 1e-9:
            blurred /= mx
        self.world.scent_grid[:] = blurred

    # -------------------------------------------------------------------------
    # DB helpers
    # -------------------------------------------------------------------------

    def _flush_lifetimes(self) -> None:
        if not self._db_writer or not self._lifetimes:
            return
        rows = self._lifetimes[:]
        self._lifetimes.clear()
        self._db_writer.enqueue_many("""
            INSERT INTO organismos_lifetimes
                (simulacao_id, mundo_id, org_id, geracao,
                 x_nasc, y_nasc, x_morte, y_morte,
                 idade, comida_total, food_rate, nasc_em_s, mort_em_s,
                 causa_morte, n_filhos, is_cross_territory)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, [(
            r["simulacao_id"], r["mundo_id"], r["org_id"], r["geracao"],
            r["x_nasc"], r["y_nasc"], r["x_morte"], r["y_morte"],
            r["idade"], r["comida_total"], r.get("food_rate", 0.0),
            r["nasc_em_s"], r["mort_em_s"],
            r["causa_morte"], r["n_filhos"], r.get("is_cross_territory", False),
        ) for r in rows])

    def _log_periodic(self, dt_s: float = 30.0) -> None:
        alive   = [o for o in self._organisms if o.state.vivo]
        elapsed = time.monotonic() - self._t_start

        cv = 0.0
        gen_med = 0.0
        if len(alive) > 1:
            try:
                div     = calcular_diversidade([o.snn for o in alive])
                cv      = div["cv_fc1"]
                gen_med = float(np.mean([o.geracao for o in alive]))
            except Exception:
                pass

        # Comida/s (usando dt real entre logs)
        comida_rate = self._comida_desde_log / max(dt_s, 0.001)

        self.stats = {
            "mundo_id":    self.config.mundo_id,
            "n_patches":   self.config.n_patches,
            "food_pct":    self.config.food_pct,
            "n_vivos":     len(alive),
            "gen_max":     self._gen_max,
            "gen_media":   gen_med,
            "cv_fc1":      cv,
            "comida_rate": comida_rate,
            "nasc":        self._nasc_desde_log,
            "mortes":      self._mortes_desde_log,
            "step":        self._step,
            "elapsed_s":   elapsed,
        }

        if self._db_writer:
            self._db_writer.execute_one("""
                INSERT INTO mundos_log
                    (simulacao_id, mundo_id, t_s, step,
                     n_vivos, comida_rate_s, gen_max, gen_media,
                     cv_fc1, n_nascimentos, n_mortes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                self.simulacao_id, self.config.mundo_id,
                elapsed, self._step,
                len(alive), comida_rate, self._gen_max, gen_med,
                cv, self._nasc_desde_log, self._mortes_desde_log,
            ))

        self._nasc_desde_log   = 0
        self._mortes_desde_log = 0
        self._comida_desde_log = 0

    def _snapshot(self) -> None:
        alive   = [o for o in self._organisms if o.state.vivo]
        elapsed = time.monotonic() - self._t_start

        if not self._db_writer or not alive:
            return

        rows = [(
            self.simulacao_id, self.config.mundo_id,
            elapsed, self._step,
            o.org_id, o.state.x, o.state.y,
            o.geracao,
            float(o.state.energia / o.state.energia_max),
            o.lifetime_food,
        ) for o in alive]
        self._db_writer.enqueue_many("""
            INSERT INTO posicoes_snapshot
                (simulacao_id, mundo_id, t_s, step,
                 org_id, x, y, geracao, energia_norm, comida_acc)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, rows)

        # Actualizar estados dos patches em mundos_config.patches_json
        pm = self.world._patch_manager
        if pm is not None:
            patches_state = [
                {"cx": int(p.cx), "cy": int(p.cy), "radius": int(p.radius), "state": p.state}
                for p in pm.patches
            ]
            self._db_writer.execute_one(
                "UPDATE mundos_config SET patches_json = %s::jsonb"
                " WHERE simulacao_id = %s AND mundo_id = %s",
                (json.dumps(patches_state), self.simulacao_id, self.config.mundo_id),
            )

    def _log_config_to_db(self) -> None:
        if not self._db_writer:
            return
        self._db_writer.execute_one("""
            INSERT INTO mundos_config
                (simulacao_id, mundo_id, world_size, n_food, food_pct,
                 n_org_inicial, vida_max, max_pop, seed)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (simulacao_id, mundo_id) DO NOTHING
        """, (
            self.simulacao_id, self.config.mundo_id,
            self.config.world_size, self.config.n_food,
            self.config.food_pct, self.config.n_org_inicial,
            self.config.vida_max, self.config.max_pop, self.config.seed,
        ))
