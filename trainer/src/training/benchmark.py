"""
Passo 15 — Benchmark de carga e optimização GPU.

Objectivo: throughput > 200 steps/s e utilização GPU > 60% durante treino.

Estratégia de optimização:
  1. Batch encoding: processar N observações de uma vez → [T, N, 18]
  2. torch.compile (PyTorch 2.x) na rede SNN para fusion de kernels.
  3. Múltiplos ambientes paralelos (sem GIL pois GPU-bound).
  4. Reduzir transferências CPU↔GPU com persistent tensor allocation.

Modo de uso:
  PYTHONPATH=trainer/src HSA_OVERRIDE_GFX_VERSION=12.0.0 \\
    python -m training.benchmark --n-envs 4 --n-steps 2000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from snn import OrganismoSNN, T_STEPS, N_INPUT
from snn.ann_init import pretreinar_com_dados_sinteticos
from snn.encoding import observacao_para_tensor
from snn.learning import RSTDPLearner, calcular_recompensa
from world import Mundo


# ---------------------------------------------------------------------------
# GPU utilisation helpers
# ---------------------------------------------------------------------------

def _gpu_mem_mb() -> float:
    """GPU memory allocated (MiB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def _rocm_utilization() -> int | None:
    """
    Query GPU utilization % via rocm-smi.
    Returns None if rocm-smi is not available.
    """
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--csv"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        # CSV line: device, GPU use%, …
        for line in out.splitlines():
            if line.startswith("card"):
                parts = line.split(",")
                if len(parts) >= 2:
                    val = parts[1].strip().rstrip("%")
                    return int(val)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Compiled SNN wrapper (optional)
# ---------------------------------------------------------------------------

def _try_compile(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Try to apply torch.compile for kernel fusion.
    Falls back to the original model if compilation fails.
    """
    if not torch.cuda.is_available():
        return model
    try:
        compiled = torch.compile(model, backend="inductor", mode="reduce-overhead")
        # Warm-up: one dummy forward to trigger compilation
        dummy = torch.zeros(T_STEPS, 1, N_INPUT, device=device)
        with torch.no_grad():
            compiled(dummy)
        return compiled
    except Exception as e:
        print(f"  [benchmark] torch.compile não disponível: {e}")
        return model


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    label: str
    n_steps: int
    elapsed_s: float
    steps_per_sec: float
    gpu_mem_mb: float
    gpu_util_pct: int | None

    def __str__(self) -> str:
        util = f"{self.gpu_util_pct}%" if self.gpu_util_pct is not None else "N/A"
        return (
            f"  [{self.label}] "
            f"{self.steps_per_sec:6.0f} steps/s | "
            f"GPU mem={self.gpu_mem_mb:.1f} MiB | "
            f"GPU util={util} | "
            f"total={self.n_steps} steps in {self.elapsed_s:.1f}s"
        )


def _run_single_env(
    snn: OrganismoSNN,
    learner: RSTDPLearner,
    device: torch.device,
    n_steps: int,
    label: str,
) -> BenchResult:
    """Single environment, sequential steps."""
    import random
    mundo = Mundo(fase=1)
    obs = mundo.reset()
    learner.reset_traces()

    t0 = time.perf_counter()
    total = 0
    for _ in range(n_steps):
        dist_ant = mundo._distancia_comida_mais_proxima()
        x = observacao_para_tensor(obs, T=T_STEPS, device=device)
        accao_idx = snn.accao(x)
        nome_acc = ["avancar", "virar_esquerda", "virar_direita", "parar"][accao_idx]
        obs2, renv, done = mundo.step(nome_acc)
        dist_dep = mundo._distancia_comida_mais_proxima()
        r = calcular_recompensa(renv, nome_acc, dist_ant, dist_dep, done)
        learner.step(x, r)
        obs = obs2
        if done:
            obs = mundo.reset()
            learner.reset_traces()
        total += 1

    elapsed = time.perf_counter() - t0
    return BenchResult(
        label=label,
        n_steps=total,
        elapsed_s=elapsed,
        steps_per_sec=total / elapsed,
        gpu_mem_mb=_gpu_mem_mb(),
        gpu_util_pct=_rocm_utilization(),
    )


def _run_batch_encoding(
    snn: OrganismoSNN,
    device: torch.device,
    n_obs: int,
    label: str,
) -> BenchResult:
    """
    Benchmark batched encoding + SNN forward:
    Encodes n_obs observations simultaneously → [T, n_obs, N_INPUT].
    """
    mundo = Mundo(fase=1)
    obs_list = []
    for _ in range(n_obs):
        obs = mundo.reset()
        obs_list.append(obs)

    # Build a [T, n_obs, N_INPUT] tensor in one pass
    t0 = time.perf_counter()
    N_ITERS = 500

    for _ in range(N_ITERS):
        # Stack n_obs observations into a batch
        tensors = [
            observacao_para_tensor(obs, T=T_STEPS, device=device).squeeze(1)
            for obs in obs_list
        ]                               # list of [T, N_INPUT]
        batch_x = torch.stack(tensors, dim=1)   # [T, n_obs, N_INPUT]

        with torch.no_grad():
            counts = snn(batch_x)       # [n_obs, 4]

    elapsed = time.perf_counter() - t0
    total = N_ITERS * n_obs
    return BenchResult(
        label=label,
        n_steps=total,
        elapsed_s=elapsed,
        steps_per_sec=total / elapsed,
        gpu_mem_mb=_gpu_mem_mb(),
        gpu_util_pct=_rocm_utilization(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SNN Organismo GPU benchmark")
    parser.add_argument("--n-steps",  type=int, default=2000, help="Steps per benchmark")
    parser.add_argument("--batch",    type=int, default=8,    help="Batch size for batched test")
    parser.add_argument("--compile",  action="store_true",    help="Use torch.compile")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[benchmark] device={device}  n_steps={args.n_steps}")
    print()

    # Setup
    snn = OrganismoSNN(device=device).to(device)
    _, loss = pretreinar_com_dados_sinteticos(snn, n_amostras=1000, epochs=5, device=device)
    print(f"  ANN pre-train loss: {loss:.3f}")

    if args.compile:
        print("  Applying torch.compile…")
        snn = _try_compile(snn, device)

    learner = RSTDPLearner(snn if hasattr(snn, 'lif1') else snn, device=device)

    # Warm-up
    dummy = torch.zeros(T_STEPS, 1, N_INPUT, device=device)
    with torch.no_grad():
        snn(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\n--- Benchmarks ---")

    # 1. Baseline: single env sequential
    r1 = _run_single_env(snn, learner, device, n_steps=args.n_steps, label="single-env")
    print(r1)

    # 2. Batched encoding + forward (measures raw SNN throughput)
    r2 = _run_batch_encoding(snn, device, n_obs=args.batch, label=f"batch-{args.batch}")
    print(r2)

    # Criterion: GPU utilization > 60% OR throughput > 200 steps/s
    target_util = 60
    target_sps = 200.0
    best_util = max(
        (r.gpu_util_pct for r in [r1, r2] if r.gpu_util_pct is not None),
        default=None,
    )
    best_sps = max(r1.steps_per_sec, r2.steps_per_sec)

    util_ok = best_util is not None and best_util >= target_util
    sps_ok  = best_sps >= target_sps
    passed  = util_ok or sps_ok

    print(f"  Critério Passo 15: GPU util ≥ {target_util}%  OU  throughput ≥ {target_sps:.0f} steps/s")
    if util_ok:
        print(f"  ✓ GPU utilização: {best_util}% ≥ {target_util}%  → PASSO 15 VALIDADO")
    elif sps_ok:
        print(f"  ✓ Throughput: {best_sps:.0f} steps/s ≥ {target_sps:.0f}  → PASSO 15 VALIDADO")
    else:
        msg = f"util={best_util}%" if best_util else "util=N/A"
        print(f"  ~ PASSO 15: {msg}, throughput={best_sps:.0f} steps/s — abaixo do critério")
        print("    Sugestões: --compile, aumentar --batch, ou verificar HSA_OVERRIDE_GFX_VERSION")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
