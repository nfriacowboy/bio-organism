"""
Trainer assíncrono — população de organismos autónomos sem tick global.

Cada organismo é uma thread independente. Não há sincronização entre eles
excepto via o mundo partilhado (food_grid, scent_grid) e o servidor GPU.

Variáveis de ambiente:
  SNN_N_ORGANISMOS     (int,   default 20)
  SNN_DEVICE           (str,   default auto)
  SNN_USAR_PATCHES     (bool,  default 1)
  SNN_MAX_TEMPO_S      (float, default 3600) — duração máxima em segundos
  SNN_LOG_INTERVAL_S   (float, default 30)
  SNN_INFERENCE_BATCH  (int,   default 64)
  SNN_TREINO_NOTA      (str,   default "async_v1")
  SNN_SNAPSHOT_INTERVAL (int,  default 5)  — snapshots a cada N log intervals
  DATABASE_URL         (str)
"""
from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from snn import OrganismoSNN
from snn.network import T_STEPS
from snn.kan_policy import KANPolicy
from snn.learning import RSTDPLearner
from world.world_async import AsyncWorld, ScentUpdater
from training.inference_server import InferenceServer
from training.organism_worker import OrganismoWorker
from world.reproduction import (
    get_genome, crossover, mutate, set_genome, herdar_epigenetica,
    calcular_diversidade,
    cosine_sim_snn,
    INBREEDING_COS_THR,
    MUTACAO_SIGMA_DEFAULT,
)

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

N_ORGANISMOS      = int(os.getenv("SNN_N_ORGANISMOS",    "20"))
DEVICE_STR        = os.getenv("SNN_DEVICE",
                               "cuda" if torch.cuda.is_available() else "cpu")
USAR_PATCHES      = os.getenv("SNN_USAR_PATCHES",   "1").strip() in ("1","true","yes")
MAX_TEMPO_S       = float(os.getenv("SNN_MAX_TEMPO_S",     "3600"))
LOG_INTERVAL_S    = float(os.getenv("SNN_LOG_INTERVAL_S",  "30"))
MAX_BATCH         = int(os.getenv("SNN_INFERENCE_BATCH",   "64"))
TREINO_NOTA       = os.getenv("SNN_TREINO_NOTA",           "async_v1")
DATABASE_URL      = os.getenv("DATABASE_URL",              "")
FASE_INICIAL      = int(os.getenv("SNN_FASE_INICIAL",      "1"))
SNAPSHOT_INTERVAL = int(os.getenv("SNN_SNAPSHOT_INTERVAL", "5"))  # log intervals


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _criar_organismo(device: torch.device) -> tuple:
    snn     = OrganismoSNN(device=device).to(device)
    kan     = KANPolicy()
    learner = RSTDPLearner(snn, device=device)
    return snn, kan, learner


def _spawn_filho_de(
    pai_a: OrganismoWorker,
    pai_b: OrganismoWorker,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple:
    """Cria SNN/KAN/learner do filho via crossover + mutação (sem side-effects de Organismo)."""
    g_a = get_genome(pai_a.snn, pai_a.kan)
    g_b = get_genome(pai_b.snn, pai_b.kan)
    g_f = crossover(g_a, g_b, rng)
    g_f = mutate(g_f, MUTACAO_SIGMA_DEFAULT, rng)

    snn_f = OrganismoSNN(device=device).to(device)
    kan_f = KANPolicy()
    set_genome(snn_f, kan_f, g_f, device)

    learner_f = RSTDPLearner(snn_f, device=device)
    herdar_epigenetica(learner_f, pai_a.learner, pai_b.learner)

    return snn_f, kan_f, learner_f


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db_connect(url: str):
    """Tenta conectar ao PostgreSQL. Devolve conn ou None."""
    if not url:
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"[async] DB falhou (sem logging): {e}")
        return None


def _db_log_intervalo(conn, treino_id: str, ep: int, elapsed: float,
                      vivos: list, lifetimes: deque,
                      geracao_max: int, nascimentos: int,
                      cv: float, gen_media: float, fase: int) -> None:
    """Insere linha em episodios_populacao_log."""
    if conn is None:
        return
    try:
        if lifetimes:
            comidas = [l["comida"] for l in lifetimes]
            idades  = [l["idade"]  for l in lifetimes]
            comida_media = float(np.mean(comidas))
            comida_max   = int(max(comidas))
            vida_media   = float(np.mean(idades))
            vida_max     = int(max(idades))
        else:
            comida_media = comida_max = vida_media = vida_max = None

        import psycopg2  # import guardado por conn não-None
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO episodios_populacao_log
                    (treino_id, episodio, fase, n_organismos,
                     comida_media, comida_max, vida_media, vida_max,
                     elapsed_s, geracao_max, n_nascimentos, cv_fc1, geracao_media)
                VALUES (%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s,%s)
            """, (
                treino_id, ep, fase, len(vivos),
                comida_media, comida_max, vida_media, vida_max,
                elapsed, geracao_max, nascimentos, cv, gen_media,
            ))
    except Exception as e:
        print(f"[async] DB log erro: {e}")


def _db_snapshot(conn, treino_id: str, ep: int, vivos: list) -> None:
    """Insere snapshot de posições em snapshots_log."""
    if conn is None or not vivos:
        return
    try:
        rows = []
        for w in vivos:
            rows.append((
                treino_id, ep, w.org_id,
                w.state.x, w.state.y, float(w.state.angulo),
                w.geracao, w.state.idade, w._lifetime_food, True,
            ))
        import psycopg2
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO snapshots_log
                    (treino_id, episodio, organismo_idx, x, y, angulo,
                     geracao, idade, comida_lifetime, vivo)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, rows)
    except Exception as e:
        print(f"[async] DB snapshot erro: {e}")


# ---------------------------------------------------------------------------
# Trainer principal
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(DEVICE_STR)
    rng    = np.random.default_rng()

    treino_id = datetime.now(timezone.utc).isoformat()

    print(f"[async] device={device}  N={N_ORGANISMOS}  patches={USAR_PATCHES}")
    print(f"[async] max_tempo={MAX_TEMPO_S:.0f}s  batch_max={MAX_BATCH}")
    print(f"[async] treino_id={treino_id}")

    # --- DB ---
    db_conn = _db_connect(DATABASE_URL)
    if db_conn:
        print(f"[async] DB conectado — {DATABASE_URL[:50]}...")

    # --- Mundo ---
    world = AsyncWorld(fase=FASE_INICIAL, usar_patches=USAR_PATCHES, rng=rng)

    # --- Scent (background) ---
    scent = ScentUpdater(world, update_interval_s=0.2, sigma=3.0)
    scent.start()

    # --- Servidor de inferência GPU ---
    server = InferenceServer(device=device, max_batch=MAX_BATCH, timeout_ms=2.0)
    server.start()

    # --- Registo da população ---
    populacao:     dict[int, OrganismoWorker] = {}
    _pop_lock      = threading.RLock()
    _next_id_ctr   = [0]
    _geracao_max   = [0]
    _lifetimes:    deque = deque(maxlen=200)
    _nascimentos_desde_log = [0]   # reset a cada logging interval

    def _next_id() -> int:
        _next_id_ctr[0] += 1
        return _next_id_ctr[0]

    def _registar_worker(worker: OrganismoWorker) -> None:
        with _pop_lock:
            populacao[worker.org_id] = worker

    def _remover_worker(org_id: int) -> None:
        with _pop_lock:
            populacao.pop(org_id, None)

    def _spawn_organismo(
        geracao:  int    = 0,
        snn=None, kan=None, learner=None,
        eh_filho: bool = False,
    ) -> OrganismoWorker:
        if snn is None:
            snn, kan, learner = _criar_organismo(device)

        worker = OrganismoWorker(
            org_id   = _next_id(),
            world    = world,
            server   = server,
            snn      = snn,
            kan      = kan,
            learner  = learner,
            device   = device,
            geracao  = geracao,
        )
        if eh_filho:
            with _pop_lock:
                _nascimentos_desde_log[0] += 1

        _registar_worker(worker)

        def _on_death(w: OrganismoWorker) -> None:
            try:
                _lifetimes.append(w.stats)
                _remover_worker(w.org_id)
                _substituir_morto(w)
            except Exception as exc:
                import traceback
                print(f"[async] ERRO em _on_death org={w.org_id}: {exc}")
                traceback.print_exc()

        worker.on_death(_on_death)
        worker.start()
        return worker

    def _substituir_morto(morto: OrganismoWorker) -> None:
        """Quando um organismo morre, spawna um filho de dois progenitores vivos."""
        with _pop_lock:
            vivos = list(populacao.values())

        if len(vivos) < 2:
            _spawn_organismo(geracao=0, eh_filho=True)
            return

        random.shuffle(vivos)
        pai_a = pai_b = None
        for a in vivos:
            for b in vivos:
                if a is b:
                    continue
                if cosine_sim_snn(a.snn, b.snn) < INBREEDING_COS_THR:
                    pai_a, pai_b = a, b
                    break
            if pai_a:
                break

        if pai_a is None:
            pai_a, pai_b = random.sample(vivos, 2)

        nova_g = max(pai_a.geracao, pai_b.geracao) + 1
        _geracao_max[0] = max(_geracao_max[0], nova_g)

        snn_f, kan_f, learner_f = _spawn_filho_de(pai_a, pai_b, device, rng)
        _spawn_organismo(geracao=nova_g, snn=snn_f, kan=kan_f, learner=learner_f,
                         eh_filho=True)

    # --- Spawnar população inicial ---
    for _ in range(N_ORGANISMOS):
        _spawn_organismo(geracao=0)

    print(f"[async] {N_ORGANISMOS} organismos iniciados")

    # --- Loop de monitorização ---
    t_start      = time.monotonic()
    t_last_log   = t_start
    _ep_counter  = [0]

    try:
        while time.monotonic() - t_start < MAX_TEMPO_S:
            now     = time.monotonic()
            elapsed = now - t_start

            if now - t_last_log >= LOG_INTERVAL_S:
                with _pop_lock:
                    n_vivos  = len(populacao)
                    vivos    = list(populacao.values())
                    nasc     = _nascimentos_desde_log[0]
                    _nascimentos_desde_log[0] = 0   # reset

                div  = {"cv_fc1": 0.0}
                cv   = 0.0
                gen_media = 0.0

                if vivos:
                    idades   = [w.state.idade    for w in vivos]
                    geracoes = [w.geracao         for w in vivos]
                    gen_media = float(np.mean(geracoes))

                    snns = [w.snn for w in vivos]
                    if len(snns) > 1:
                        div = calcular_diversidade(snns)
                    cv = div["cv_fc1"]

                if _lifetimes:
                    lt_comida  = [l["comida"] for l in _lifetimes]
                    lt_idade   = [l["idade"]  for l in _lifetimes]
                    med_comida = np.mean(lt_comida)
                    med_idade  = np.mean(lt_idade)
                else:
                    med_comida = med_idade = 0.0

                _ep_counter[0] += 1
                ep = _ep_counter[0]

                print(
                    f"[{elapsed:6.0f}s] vivos={n_vivos:3d} "
                    f"idade_med={np.mean(idades) if vivos else 0:5.0f} "
                    f"comida_ep={med_comida:5.1f} "
                    f"vida_med={med_idade:5.0f} "
                    f"gen_max={_geracao_max[0]:3d} "
                    f"nasc={nasc:3d} "
                    f"cv={cv:.3f} "
                    f"food_grid={world.n_food:4d} "
                    f"batch_avg={server.avg_batch_size:.1f}"
                )

                # DB: métricas de população
                _db_log_intervalo(
                    db_conn, treino_id, ep, elapsed,
                    vivos, _lifetimes,
                    _geracao_max[0], nasc, cv, gen_media, FASE_INICIAL,
                )

                # DB: snapshot tribal (a cada SNAPSHOT_INTERVAL intervals)
                if ep % SNAPSHOT_INTERVAL == 0:
                    _db_snapshot(db_conn, treino_id, ep, vivos)

                t_last_log = now

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[async] Interrompido pelo utilizador")

    finally:
        server.stop()
        scent.stop()

        print(f"\n[async] Terminado após {time.monotonic()-t_start:.0f}s")
        print(f"[async] Total batches GPU: {server.n_batches}  "
              f"requests: {server.n_requests}  "
              f"avg_batch: {server.avg_batch_size:.1f}")

        if _lifetimes:
            all_comida = [l["comida"] for l in _lifetimes]
            all_idade  = [l["idade"]  for l in _lifetimes]
            print(f"[async] Lifetimes registadas: {len(_lifetimes)}")
            print(f"[async] Comida/lifetime: media={np.mean(all_comida):.1f} "
                  f"max={max(all_comida)}")
            print(f"[async] Vida/lifetime:   media={np.mean(all_idade):.0f} "
                  f"max={max(all_idade)}")

        if db_conn:
            try:
                db_conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
