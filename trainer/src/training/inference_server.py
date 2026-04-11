"""
Servidor de inferência GPU batched para população assíncrona.

Cada organismo (thread) submete um pedido (snn, x) e bloqueia.
O servidor acumula pedidos até ao batch máximo ou timeout, depois
dispara uma única chamada GPU para todos — N×speedup vs N chamadas sequenciais.

Thread-safety: queue.Queue é thread-safe por design.
GPU: todos os organismos vivos partilham UMA chamada batched_forward_multi_snn.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import torch

from snn.network import batched_forward_multi_snn


# ---------------------------------------------------------------------------
# Resultado de inferência
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Resultado devolvido ao organismo pelo servidor."""
    action:       int            # índice da acção (winner-take-all)
    spike_counts: torch.Tensor   # [N_OUTPUT] — contagem de spikes
    h1:           torch.Tensor   # [T, 1, N_H1]  — activações H1 para R-STDP
    h2:           torch.Tensor   # [T, 1, N_H2]  — activações H2 para R-STDP
    out:          torch.Tensor   # [T, 1, N_OUTPUT] — activações output para R-STDP


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------

class InferenceServer:
    """
    Servidor GPU de inferência batched.

    Cada organismo chama request(snn, x) que bloqueia até receber resultado.
    O servidor corre numa thread dedicada e processa batches quando:
      - Acumulou max_batch pedidos, OU
      - Passou timeout_ms desde o primeiro pedido no batch actual.

    Com N=50 organismos e timeout=2ms:
      - Batch típico: 40-50 organismos por chamada
      - GPU speedup: ~40-50× vs sequential
    """

    def __init__(
        self,
        device:     torch.device,
        max_batch:  int   = 64,
        timeout_ms: float = 2.0,
    ):
        self.device    = device
        self.max_batch = max_batch
        self.timeout   = timeout_ms / 1000.0

        self._queue:   queue.Queue = queue.Queue()
        self._running: bool        = False
        self._thread   = threading.Thread(
            target=self._serve, daemon=True, name="InferenceServer"
        )

        # Métricas
        self.n_batches     = 0
        self.n_requests    = 0
        self.avg_batch_size = 0.0

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def request(self, snn, x: torch.Tensor) -> InferenceResult:
        """
        Submete pedido de inferência. Bloqueia até resultado estar disponível.

        snn : OrganismoSNN com os seus próprios pesos
        x   : [T, 1, N_INPUT] tensor de spikes
        Devolve InferenceResult com acção e activações para R-STDP.
        """
        evt           = threading.Event()
        result_holder = [None]
        self._queue.put((snn, x, evt, result_holder))
        evt.wait()
        return result_holder[0]

    # --- Loop do servidor ---

    def _serve(self) -> None:
        while self._running:
            batch = self._collect_batch()
            if batch:
                self._process_batch(batch)

    def _collect_batch(self) -> list:
        """
        Colecta pedidos durante no máximo timeout ou até max_batch.
        Bloqueia até pelo menos 1 pedido chegar.
        """
        batch = []
        try:
            # Bloqueante para o 1º item (evita busy-wait)
            first = self._queue.get(timeout=0.005)
            batch.append(first)
        except queue.Empty:
            return batch

        # Drena a queue até ao limite sem bloquear
        deadline = time.monotonic() + self.timeout
        while len(batch) < self.max_batch and time.monotonic() < deadline:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break

        return batch

    @torch.no_grad()
    def _process_batch(self, batch: list) -> None:
        """Executa batched forward e responde a cada organismo."""
        snns = [item[0] for item in batch]
        xs   = [item[1] for item in batch]   # list of [T, 1, N_INPUT]

        # Stack: [N, T, N_INPUT]
        x_batch = torch.stack([x.squeeze(1) for x in xs], dim=0)

        spike_counts, h1_b, h2_b, out_b = batched_forward_multi_snn(
            snns, x_batch, self.device
        )

        # Distribuir resultados
        for j, (snn, x, evt, result_holder) in enumerate(batch):
            result_holder[0] = InferenceResult(
                action       = int(spike_counts[j].argmax().item()),
                spike_counts = spike_counts[j].cpu(),
                h1  = h1_b[:,  j:j+1, :],   # [T, 1, N_H1]
                h2  = h2_b[:,  j:j+1, :],   # [T, 1, N_H2]
                out = out_b[:, j:j+1, :],   # [T, 1, N_OUTPUT]
            )
            evt.set()

        # Métricas
        self.n_batches  += 1
        self.n_requests += len(batch)
        self.avg_batch_size = self.n_requests / self.n_batches
