"""
Aprendizagem R-STDP (Reward-modulated STDP) básica.
Passo 6: sem NACA nem AGMP — apenas R-STDP para validar que o organismo aprende.

A eligibility trace é calculada por correlação pré-pós spike dentro de cada step (T=8 timesteps).
A actualização de pesos ocorre no fim de cada step do mundo, modulada pelo sinal de recompensa.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass, field

from .network import OrganismoSNN, N_H1, N_H2, N_INPUT, N_OUTPUT


@dataclass
class ConfigRSTDP:
    lr_snn: float = 1e-3            # taxa de aprendizagem base
    tau_trace: float = 20.0         # constante de tempo da eligibility trace (ms ~ timesteps)
    clip_w: float = 1.0             # clipping máximo dos pesos
    grad_clip: float = 0.5          # clipping do delta_w por step


class EligibilityTrace:
    """
    Mantém a eligibility trace de uma ligação (pre → post).
    trace = correlação temporal filtrada entre spikes pre e post.
    """

    def __init__(self, shape: tuple[int, int], tau: float = 20.0, device: torch.device | None = None):
        self.tau = tau
        self.device = device or torch.device("cpu")
        self.trace = torch.zeros(shape, device=self.device)

    def actualizar(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """
        pre_spikes:  [T, batch, n_pre]
        post_spikes: [T, batch, n_post]
        Correlação 0-lag somada ao longo dos T timesteps.
        trace[post, pre] += sum_t(post[t] ⊗ pre[t]) / T
        """
        T = pre_spikes.shape[0]
        # Médias sobre batch e T para obter correlação [n_post, n_pre]
        pre_mean  = pre_spikes.mean(dim=(0, 1))    # [n_pre]
        post_mean = post_spikes.mean(dim=(0, 1))   # [n_post]
        corr = torch.outer(post_mean, pre_mean)    # [n_post, n_pre]
        # Decaimento exponencial + acumulação
        decay = torch.exp(torch.tensor(-1.0 / self.tau))
        self.trace = decay * self.trace + corr

    def reset(self) -> None:
        self.trace.zero_()


class RSTDPLearner:
    """
    Aprende os pesos da SNN via R-STDP básico (sem NACA/AGMP).

    Mantém eligibility traces para as 3 camadas de pesos:
      fc1 (input→H1), fc2 (H1→H2), fc3 (H2→output)

    A actualização ocorre no fim de cada step do mundo:
      delta_W = lr * reward * eligibility_trace
    """

    def __init__(self, snn: OrganismoSNN, cfg: ConfigRSTDP | None = None, device: torch.device | None = None):
        self.snn = snn
        self.cfg = cfg or ConfigRSTDP()
        self.device = device or torch.device("cpu")

        # Eligibility traces para cada camada
        self.et_fc1 = EligibilityTrace((N_H1, N_INPUT), self.cfg.tau_trace, self.device)
        self.et_fc2 = EligibilityTrace((N_H2, N_H1),   self.cfg.tau_trace, self.device)
        self.et_fc3 = EligibilityTrace((N_OUTPUT, N_H2), self.cfg.tau_trace, self.device)

        # Recompensa acumulada (janela exponencial para suavizar)
        self._reward_ema = 0.0
        self._reward_tau = 10.0   # suavização do sinal de recompensa

        # Spikes guardados no último forward (injectados pelo hook)
        self._last_input: torch.Tensor | None = None
        self._last_h1: torch.Tensor | None = None
        self._last_h2: torch.Tensor | None = None
        self._last_out: torch.Tensor | None = None

        # Instalar hooks de forward para capturar spikes intermédios
        self._instalar_hooks()

    # -------------------------------------------------------------------
    def _instalar_hooks(self) -> None:
        """Hooks que gravam spikes de cada camada durante o forward."""

        def hook_h1(module, input, output):
            # output: spike do lif1, shape [batch, N_H1] — mas chamado T vezes
            # Queremos acumular ao longo dos T timesteps
            if not hasattr(self, "_h1_buffer"):
                self._h1_buffer = []
            self._h1_buffer.append(output.detach().unsqueeze(0))  # [1, batch, N_H1]

        def hook_h2(module, input, output):
            if not hasattr(self, "_h2_buffer"):
                self._h2_buffer = []
            self._h2_buffer.append(output.detach().unsqueeze(0))

        def hook_out(module, input, output):
            if not hasattr(self, "_out_buffer"):
                self._out_buffer = []
            self._out_buffer.append(output.detach().unsqueeze(0))

        self.snn.lif1.register_forward_hook(hook_h1)
        self.snn.lif2.register_forward_hook(hook_h2)
        self.snn.lif3.register_forward_hook(hook_out)

    # -------------------------------------------------------------------
    def _colectar_spikes(self, x_input: torch.Tensor) -> None:
        """Chama o forward e recolhe os buffers de spikes."""
        self._h1_buffer = []
        self._h2_buffer = []
        self._out_buffer = []
        _ = self.snn(x_input)   # popula os buffers via hooks

        T = len(self._h1_buffer)
        if T == 0:
            return

        self._last_input = x_input.detach()                           # [T, batch, N_INPUT]
        self._last_h1    = torch.cat(self._h1_buffer, dim=0)          # [T, batch, N_H1]
        self._last_h2    = torch.cat(self._h2_buffer, dim=0)          # [T, batch, N_H2]
        self._last_out   = torch.cat(self._out_buffer, dim=0)         # [T, batch, N_OUTPUT]

    # -------------------------------------------------------------------
    def step(self, x_input: torch.Tensor, recompensa: float) -> torch.Tensor:
        """
        Executa um step de aprendizagem:
        1. Forward pass (recolhe spikes)
        2. Actualiza eligibility traces
        3. Actualiza pesos com R-STDP
        Devolve spike_counts [batch, N_OUTPUT].
        """
        self._colectar_spikes(x_input)

        if self._last_input is None:
            return torch.zeros(x_input.shape[1], N_OUTPUT, device=self.device)

        # Actualizar eligibility traces
        self.et_fc1.actualizar(self._last_input, self._last_h1)
        self.et_fc2.actualizar(self._last_h1,   self._last_h2)
        self.et_fc3.actualizar(self._last_h2,   self._last_out)

        # Suavizar recompensa (EMA)
        self._reward_ema += (recompensa - self._reward_ema) / self._reward_tau

        # Actualizar pesos se há sinal de aprendizagem
        self._actualizar_pesos(self._reward_ema)

        # Reconstruir counts do último forward
        return self._last_out.sum(dim=0)   # [batch, N_OUTPUT]

    # -------------------------------------------------------------------
    @torch.no_grad()
    def _actualizar_pesos(self, recompensa: float) -> None:
        lr = self.cfg.lr_snn
        clip_dw = self.cfg.grad_clip
        clip_w  = self.cfg.clip_w

        for camada, trace in [
            (self.snn.fc1, self.et_fc1.trace),
            (self.snn.fc2, self.et_fc2.trace),
            (self.snn.fc3, self.et_fc3.trace),
        ]:
            if camada.weight.shape != trace.shape:
                continue
            delta_w = lr * recompensa * trace
            delta_w = delta_w.clamp(-clip_dw, clip_dw)
            camada.weight.add_(delta_w)
            camada.weight.clamp_(-clip_w, clip_w)

    # -------------------------------------------------------------------
    def reset_traces(self) -> None:
        """Chamar no início de cada episódio."""
        self.et_fc1.reset()
        self.et_fc2.reset()
        self.et_fc3.reset()
        self._reward_ema = 0.0


# ---------------------------------------------------------------------------
# Sinal de recompensa composto (reward shaping)
# ---------------------------------------------------------------------------

def calcular_recompensa(
    recompensa_ambiente: float,
    accao: str,
    dist_comida_antes: float | None,
    dist_comida_depois: float | None,
    morreu: bool,
) -> float:
    """
    Combina o sinal do ambiente com reward shaping adicional.
    O ambiente (Mundo.step) já inclui: +1.0 comer, -5.0 morte, bónus proximidade.
    Aqui adicionamos apenas o custo de movimento e bónus de aproximação direcional.
    """
    r = recompensa_ambiente
    r += -0.01   # custo de movimento por step (step de aprendizagem)

    # Bónus de aproximação: recompensa por reduzir distância à comida mais próxima
    if dist_comida_antes is not None and dist_comida_depois is not None:
        aproximacao = dist_comida_antes - dist_comida_depois
        if aproximacao > 0:
            r += 0.05 * min(aproximacao / 5.0, 1.0)

    return r
