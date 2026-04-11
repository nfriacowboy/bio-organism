"""
Trainer multi-mundo — 100 mundos paralelos com nº de patches variável.

Design:
- N_MUNDOS mundos (default 100), divididos em grupos por nº de food patches
- 1 InferenceServer GPU partilhado → batching cross-mundo eficiente
- N_MUNDOS threads (1 por mundo) + 1 thread de monitorização
- DB: tabelas 007_multi_world (simulacoes, mundos_config, organismos_lifetimes,
       mundos_log, posicoes_snapshot)

Patches de comida (variável científica):
  10 níveis × 10 réplicas = 100 mundos
  n_patches: [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

Hipótese a testar:
  Poucos patches → poucos territórios → competição forte → tribos separadas
  Muitos patches → territórios distribuídos → mistura → sem estrutura tribal clara

Pesos base: v80p (olfato 1/d² treinado, fase 3, 7.6× random) → organismos navegam
  para patches por olfato → filhos nascem perto de patch → tribo = lineagem territorial

Variáveis de ambiente:
  SNN_N_MUNDOS         (int,   default 100)
  SNN_WORLD_SIZE       (int,   default 500)
  SNN_N_ORG_PER_WORLD  (int,   default 10)
  SNN_VIDA_MAX         (int,   default 100)
  SNN_MAX_POP          (int,   default 25)
  SNN_PATCH_RADIUS     (int,   default 40)
  SNN_BASE_SNN_PATH    (str,   default "")  — path snn_ep*.pt de treino base
  SNN_DEVICE           (str,   default auto)
  SNN_MAX_TEMPO_S      (float, default 7200)
  SNN_LOG_INTERVAL_S   (float, default 120)
  DATABASE_URL         (str)
  SNN_TREINO_NOTA      (str)
"""
from __future__ import annotations

import os
import time
import json
import threading
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch

from training.world_runner import WorldRunner, WorldConfig
from training.inference_server import InferenceServer
from training.db_writer import DBWriter

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

N_MUNDOS        = int(os.getenv("SNN_N_MUNDOS",          "20"))
WORLD_SIZE      = int(os.getenv("SNN_WORLD_SIZE",         "200"))
N_ORG_PER_WORLD = int(os.getenv("SNN_N_ORG_PER_WORLD",   "10"))
VIDA_MAX        = int(os.getenv("SNN_VIDA_MAX",           "300"))
MAX_POP         = int(os.getenv("SNN_MAX_POP",            "25"))
PATCH_RADIUS    = int(os.getenv("SNN_PATCH_RADIUS",       "40"))
DEVICE_STR      = os.getenv("SNN_DEVICE",
                             "cuda" if torch.cuda.is_available() else "cpu")
MAX_TEMPO_S     = float(os.getenv("SNN_MAX_TEMPO_S",      "7200"))
LOG_INTERVAL_S  = float(os.getenv("SNN_LOG_INTERVAL_S",   "60"))
MAX_BATCH       = int(os.getenv("SNN_INFERENCE_BATCH",    "256"))
DATABASE_URL    = os.getenv("DATABASE_URL", "")
TREINO_NOTA     = os.getenv("SNN_TREINO_NOTA",            "multi_world_patches_sweep")
BASE_SNN_PATH   = os.getenv("SNN_BASE_SNN_PATH",          "")  # ex: mlartifacts/.../snn_ep2000.pt
FIXED_N_PATCHES = int(os.getenv("SNN_FIXED_N_PATCHES",   "0"))  # >0 → todos os mundos com este valor (modo validação)


# ---------------------------------------------------------------------------
# Geração de configurações de mundos
# ---------------------------------------------------------------------------

def _gerar_configs(
    n_mundos:       int,
    world_size:     int,
    n_org:          int,
    vida_max:       int,
    max_pop:        int,
    patch_radius:   int,
    base_snn_path:  str,
    fixed_n_patches: int = 0,
) -> list[WorldConfig]:
    """
    Distribui N mundos por nº de food patches.

    Modo validação (fixed_n_patches > 0): todos os mundos com o mesmo n_patches.
    Modo sweep (fixed_n_patches == 0): 4 níveis × (n_mundos//4) réplicas.
      Níveis: [2, 4, 6, 10]
    """
    if fixed_n_patches > 0:
        # Validação — todos os mundos com n_patches fixo
        levels_reps = [(fixed_n_patches, n_mundos)]
    else:
        # Sweep — 4 níveis distribuídos por n_mundos
        n_patches_levels = [2, 3, 4, 6]
        n_levels      = len(n_patches_levels)
        replicas_base = n_mundos // n_levels
        extras        = n_mundos  % n_levels
        levels_reps   = [
            (n_patches_levels[i], replicas_base + (1 if i < extras else 0))
            for i in range(n_levels)
        ]

    configs  = []
    mundo_id = 0
    for n_patches, reps in levels_reps:
        for rep in range(reps):
            configs.append(WorldConfig(
                mundo_id      = mundo_id,
                world_size    = world_size,
                n_patches     = n_patches,
                patch_radius  = patch_radius,
                usar_patches  = True,
                n_org_inicial = n_org,
                vida_max      = vida_max,
                max_pop       = max_pop,
                seed          = mundo_id * 1000 + rep,
                base_snn_path = base_snn_path,
            ))
            mundo_id += 1

    return configs[:n_mundos]


# ---------------------------------------------------------------------------
# DB helpers do trainer (simulacoes + mundos_config)
# ---------------------------------------------------------------------------

def _db_regista_simulacao(
    writer: DBWriter, simulacao_id: str, nota: str, n_mundos: int, configs: list[WorldConfig]
) -> None:
    config_json = json.dumps({
        "nota":             nota,
        "world_size":       WORLD_SIZE,
        "n_org_per_world":  N_ORG_PER_WORLD,
        "vida_max":         VIDA_MAX,
        "max_pop":          MAX_POP,
        "patch_radius":     PATCH_RADIUS,
        "device":           DEVICE_STR,
        "base_snn_path":    BASE_SNN_PATH,
        "n_patches_levels": sorted(set(c.n_patches for c in configs)),
        "sweep_var":        "n_patches",
    })
    writer.execute_one("""
        INSERT INTO simulacoes (simulacao_id, nota, n_mundos, config_json)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (simulacao_id) DO NOTHING
    """, (simulacao_id, nota, n_mundos, config_json))

    # n_food reused to store n_patches (patches sweep; food count controlled by PatchManager)
    writer.enqueue_many("""
        INSERT INTO mundos_config
            (simulacao_id, mundo_id, world_size, n_food, food_pct,
             n_org_inicial, vida_max, max_pop, seed)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (simulacao_id, mundo_id) DO NOTHING
    """, [(
        simulacao_id, c.mundo_id, c.world_size,
        c.n_patches,   # n_food = n_patches (proxy para queries)
        c.food_pct,    # estimativa baseada em n_patches × πr²
        c.n_org_inicial, c.vida_max, c.max_pop, c.seed,
    ) for c in configs])


# ---------------------------------------------------------------------------
# Thread de monitorização
# ---------------------------------------------------------------------------

def _monitor(
    runners:        list[WorldRunner],
    simulacao_id:   str,
    max_tempo_s:    float,
    log_interval_s: float,
) -> None:
    t_start    = time.monotonic()
    t_last_log = t_start

    while time.monotonic() - t_start < max_tempo_s:
        time.sleep(1.0)
        now     = time.monotonic()
        elapsed = now - t_start

        if now - t_last_log >= log_interval_s:
            # Agregar stats de todos os mundos
            all_stats = [r.stats for r in runners if r.stats]

            if all_stats:
                n_vivos_total = sum(s["n_vivos"]    for s in all_stats)
                gen_max_geral = max(s["gen_max"]     for s in all_stats)
                cv_media      = np.mean([s["cv_fc1"] for s in all_stats])
                comida_total  = sum(s.get("comida_rate", 0) for s in all_stats)
                mundos_vivos  = sum(1 for r in runners if r.is_alive())

                # Distribuição por nº de patches
                by_patches: dict[int, list] = {}
                for s in all_stats:
                    np_ = s.get("n_patches", 0)
                    by_patches.setdefault(np_, []).append(s)

                print(
                    f"\n[{elapsed:6.0f}s] {mundos_vivos}/{len(runners)} mundos vivos"
                    f"  orgs={n_vivos_total}"
                    f"  gen_max={gen_max_geral}"
                    f"  cv_med={cv_media:.3f}"
                    f"  comida_rate={comida_total:.1f}/s"
                )
                print(
                    f"  {'n_patches':>10}  {'mundos':>6}  "
                    f"{'orgs_med':>8}  {'gen_max':>7}  {'cv':>6}  {'food/s':>7}"
                )
                for np_ in sorted(by_patches):
                    sl   = by_patches[np_]
                    print(
                        f"  {np_:>10d}  {len(sl):>6d}  "
                        f"{np.mean([s['n_vivos'] for s in sl]):>8.1f}  "
                        f"{max(s['gen_max'] for s in sl):>7d}  "
                        f"{np.mean([s['cv_fc1'] for s in sl]):>6.3f}  "
                        f"{sum(s.get('comida_rate',0) for s in sl):>7.2f}"
                    )

            t_last_log = now

    print(f"\n[multi] Monitorização terminada após {time.monotonic()-t_start:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device       = torch.device(DEVICE_STR)
    simulacao_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 65)
    print("  bio-organism — Simulação Multi-Mundo (N-Patches Sweep)")
    print("=" * 65)
    print(f"  simulacao_id : {simulacao_id}")
    print(f"  N_MUNDOS     : {N_MUNDOS}")
    print(f"  WORLD_SIZE   : {WORLD_SIZE}×{WORLD_SIZE}")
    print(f"  N_ORG/MUNDO  : {N_ORG_PER_WORLD}")
    print(f"  VIDA_MAX     : {VIDA_MAX}")
    print(f"  PATCH_RADIUS : {PATCH_RADIUS}")
    print(f"  BASE_SNN     : {BASE_SNN_PATH or '(aleatório)'}")
    print(f"  DEVICE       : {device}")
    print(f"  MAX_TEMPO    : {MAX_TEMPO_S:.0f}s")
    print(f"  DATABASE     : {'sim' if DATABASE_URL else 'desactivado'}")
    print("=" * 65)

    # Gerar configurações
    configs = _gerar_configs(
        N_MUNDOS, WORLD_SIZE, N_ORG_PER_WORLD, VIDA_MAX, MAX_POP,
        PATCH_RADIUS, BASE_SNN_PATH, fixed_n_patches=FIXED_N_PATCHES,
    )
    patch_counts = Counter(c.n_patches for c in configs)
    print(f"  N_patches    : {dict(sorted(patch_counts.items()))}")
    print(f"  Réplicas     : {dict(sorted(Counter(v for v in patch_counts.values()).items()))}")
    print()

    # DB — único writer partilhado por todos os WorldRunners
    db_writer = DBWriter(DATABASE_URL)
    db_active = db_writer.start()
    if db_active:
        _db_regista_simulacao(db_writer, simulacao_id, TREINO_NOTA, N_MUNDOS, configs)
    else:
        print("[multi] DB desactivada — a correr sem persistência")

    # InferenceServer partilhado
    server = InferenceServer(device=device, max_batch=MAX_BATCH, timeout_ms=2.0)
    server.start()

    # Spawnar worlds
    runners = []
    for cfg in configs:
        r = WorldRunner(
            config       = cfg,
            server       = server,
            device       = device,
            simulacao_id = simulacao_id,
            db_writer    = db_writer if db_active else None,
        )
        runners.append(r)

    # Iniciar todos
    t_start = time.monotonic()

    # Gravar posições dos patches na BD (antes de iniciar threads, patches prontos no __init__)
    if db_active:
        db_writer.enqueue_many(
            "UPDATE mundos_config SET patches_json = %s::jsonb "
            "WHERE simulacao_id = %s AND mundo_id = %s",
            [(json.dumps(r.get_patch_positions()), simulacao_id, r.config.mundo_id)
             for r in runners],
        )

    for r in runners:
        r.start()

    print(f"[multi] {len(runners)} WorldRunners iniciados")

    # Thread de monitorização
    monitor_t = threading.Thread(
        target=_monitor,
        args=(runners, simulacao_id, MAX_TEMPO_S, LOG_INTERVAL_S),
        daemon=True,
        name="Monitor",
    )
    monitor_t.start()

    # Esperar monitorização (ou Ctrl+C)
    try:
        monitor_t.join()
    except KeyboardInterrupt:
        print("\n[multi] Ctrl+C — a parar mundos...")

    # Parar todos os runners e aguardar conclusão
    for r in runners:
        r.stop()
    for r in runners:
        r.join(timeout=5.0)

    # Sumário final
    elapsed = time.monotonic() - t_start
    all_stats = [r.stats for r in runners if r.stats]

    print(f"\n[multi] === SUMÁRIO FINAL ===")
    print(f"  Duração total: {elapsed:.0f}s")
    print(f"  Mundos completados: {sum(1 for r in runners if r.stats)}/{len(runners)}")

    if all_stats:
        print(f"\n  Por nº de patches:")
        print(f"  {'n_patches':>10}  {'n_mundos':>8}  {'gen_max_max':>11}  "
              f"{'cv_fc1_med':>10}  {'orgs_med':>8}")

        by_patches: dict[int, list] = {}
        for s in all_stats:
            np_ = s.get("n_patches", 0)
            by_patches.setdefault(np_, []).append(s)

        for np_ in sorted(by_patches):
            sl = by_patches[np_]
            print(
                f"  {np_:>10d}  {len(sl):>8d}  "
                f"{max(s['gen_max'] for s in sl):>11d}  "
                f"{np.mean([s['cv_fc1'] for s in sl]):>10.4f}  "
                f"{np.mean([s['n_vivos'] for s in sl]):>8.1f}"
            )

    print(f"\n  simulacao_id: {simulacao_id}")
    print(f"  GPU batches: {server.n_batches}  requests: {server.n_requests}  "
          f"avg_batch: {server.avg_batch_size:.1f}")

    server.stop()
    db_writer.stop()


if __name__ == "__main__":
    main()
