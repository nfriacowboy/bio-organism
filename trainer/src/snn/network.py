"""
Arquitectura SNN do Organismo.

Topologia:
  Input  18  (15 visão + 3 estado: energia_norm, tempo_vida_norm, bias=1.0)
  H1     64  LIF feedforward
  H2     32  RLIF recorrente (ligações laterais)
  Output  4  LIF winner-take-all → avancar | virar_esquerda | virar_direita | parar

Params base: tau=20ms, threshold=1.0, soft-reset, FastSigmoid(slope=25), T=8
"""
from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# Dimensões fixas da rede
N_INPUT   = 18   # 15 visão + 3 estado
N_H1      = 64
N_H2      = 32
N_OUTPUT  = 4
T_STEPS   = 8    # timesteps por step do mundo


def _make_lif(tau: float = 20.0, threshold: float = 1.0) -> neuron.LIFNode:
    return neuron.LIFNode(
        tau=tau,
        v_threshold=threshold,
        v_reset=None,                     # soft-reset: v -= threshold após spike
        surrogate_function=surrogate.Sigmoid(alpha=25.0),
        detach_reset=True,
    )


class OrganismoSNN(nn.Module):
    """
    SNN motor/reactiva do organismo.
    Passo 4: forward pass sem aprendizagem (pesos aleatórios).
    A aprendizagem (R-STDP) é adicionada no Passo 6.
    """

    def __init__(
        self,
        tau: float = 5.0,          # tau=5 com T=8: membrana carrega em ~6 steps
        threshold: float = 0.3,    # limiar ajustado para inputs rate-coded
        t_steps: int = T_STEPS,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.t_steps = t_steps
        self.device = device or torch.device("cpu")

        # --- Camada 1: feedforward ---
        self.fc1  = layer.Linear(N_INPUT, N_H1)
        self.lif1 = _make_lif(tau, threshold)

        # --- Camada 2: recorrente (RLIF) ---
        self.fc2     = layer.Linear(N_H1, N_H2)
        self.fc2_rec = layer.Linear(N_H2, N_H2, bias=False)   # ligações laterais
        self.lif2    = _make_lif(tau, threshold)

        # --- Camada de output ---
        self.fc3  = layer.Linear(N_H2, N_OUTPUT)
        self.lif3 = _make_lif(tau, threshold)

        # Bias extra modulado pela KAN (injectado externamente)
        self.bias_explorar  = 0.0
        self.bias_exploitar = 0.0

        # Inicialização de pesos (Xavier uniform)
        self._init_weights()

        functional.set_step_mode(self, step_mode="s")

    # -------------------------------------------------------------------
    def _init_weights(self) -> None:
        # Pesos positivos com escala suficiente para gerar spikes logo no início.
        # Com threshold=0.3 e ~3 inputs activos por timestep precisamos:
        # 3 * avg_weight > 0.3  →  avg_weight > 0.1  →  [0.05, 0.25]
        nn.init.uniform_(self.fc1.weight, 0.05, 0.25)
        nn.init.constant_(self.fc1.bias, 0.05)

        nn.init.uniform_(self.fc2.weight, 0.03, 0.18)
        nn.init.constant_(self.fc2.bias, 0.02)

        nn.init.uniform_(self.fc3.weight, -0.15, 0.15)
        nn.init.zeros_(self.fc3.bias)

        # Recorrência inicializada pequena para estabilidade
        nn.init.uniform_(self.fc2_rec.weight, -0.05, 0.05)

    # -------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, batch, N_INPUT] ou [T, N_INPUT] (sem batch dim)
        Devolve spike_counts: [batch, N_OUTPUT] — número de spikes por neurónio output.
        """
        # Normalizar para [T, batch, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)   # [T, 1, N_INPUT]

        T, B, _ = x.shape
        assert T == self.t_steps, f"Esperava T={self.t_steps}, recebeu T={T}"

        # Reset estados dos neurónios no início de cada step do mundo
        functional.reset_net(self)

        spike_out_accum = torch.zeros(B, N_OUTPUT, device=self.device)
        h2_spikes = torch.zeros(B, N_H2, device=self.device)

        # Rastreamento de taxas de spikes para AGMP (acessível externamente)
        self._spike_rates_h1: list[torch.Tensor] = []
        self._spike_rates_h2: list[torch.Tensor] = []

        for t in range(T):
            xt = x[t]                                            # [B, N_INPUT]

            # H1
            h1 = self.lif1(self.fc1(xt))                        # [B, N_H1]

            # H2 com recorrência
            h2_in = self.fc2(h1) + self.fc2_rec(h2_spikes)
            h2 = self.lif2(h2_in)                               # [B, N_H2]
            h2_spikes = h2.detach()                             # detach para estabilidade

            # Output com bias KAN
            out_in = self.fc3(h2)
            # Bias explorar → neurónios 0 (avancar) e 2 (virar_direita)
            # Bias exploitar → neurónio 0 (avancar)
            out_in[:, 0] += self.bias_exploitar * 0.2
            out_in[:, 1] += self.bias_explorar * 0.2
            out_in[:, 2] += self.bias_explorar * 0.2
            out = self.lif3(out_in)                             # [B, N_OUTPUT]

            spike_out_accum += out

            # Guardar para AGMP
            self._spike_rates_h1.append(h1.detach().mean().item())
            self._spike_rates_h2.append(h2.detach().mean().item())

        return spike_out_accum   # [B, N_OUTPUT] → spike counts por neuronío

    # -------------------------------------------------------------------
    def accao(self, x: torch.Tensor) -> int:
        """
        Winner-take-all: devolve o índice da acção com mais spikes.
        x: input para um único step (sem batch dim).
        """
        with torch.no_grad():
            counts = self.forward(x)   # [1, N_OUTPUT]
        return int(counts.squeeze(0).argmax().item())

    # -------------------------------------------------------------------
    @property
    def taxa_spikes_media(self) -> float:
        """Taxa de spikes média nas camadas ocultas (para AGMP)."""
        if not self._spike_rates_h1:
            return 0.0
        return (sum(self._spike_rates_h1) + sum(self._spike_rates_h2)) / (
            len(self._spike_rates_h1) + len(self._spike_rates_h2)
        )

    # -------------------------------------------------------------------
    def replay_update(self, spike_patterns: dict, recompensa: float) -> None:
        """
        Replay: reforça padrões de spikes guardados no ExperienceReplay.

        Chamado por ExperienceReplay.consolidar() durante a fase de "sono"
        (entre episódios). A implementação efectiva do update de pesos é
        feita pelo RSTDPLearner via o ciclo de consolidação — este método
        serve como ponto de extensão para actualizações futuras directamente
        na rede (ex: Hebbian offline sobre os padrões arquivados).

        spike_patterns: dict com chaves 'h1', 'h2', 'out' (tensores de spikes).
        recompensa:     sinal de recompensa associado à experiência.
        """
        pass   # stub — full weight update delegated to RSTDPLearner.consolidar_replay()


# ---------------------------------------------------------------------------
# ANN-to-SNN conversion helper (spec §2.2 NOTA IMP.)
# ---------------------------------------------------------------------------

class _ANNEquivalente(nn.Module):
    """MLP equivalente para pré-treino (5 epochs backprop) antes de converter para SNN."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUT, N_H1), nn.ReLU(),
            nn.Linear(N_H1, N_H2),   nn.ReLU(),
            nn.Linear(N_H2, N_OUTPUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def converter_ann_para_snn(snn: OrganismoSNN, ann: _ANNEquivalente) -> None:
    """
    Copia os pesos do MLP pré-treinado para a SNN.
    Mapeia: ann.net[0]→fc1, ann.net[2]→fc2, ann.net[4]→fc3
    """
    with torch.no_grad():
        snn.fc1.weight.copy_(ann.net[0].weight)
        snn.fc1.bias.copy_(ann.net[0].bias)
        snn.fc2.weight.copy_(ann.net[2].weight)
        snn.fc2.bias.copy_(ann.net[2].bias)
        snn.fc3.weight.copy_(ann.net[4].weight)
        snn.fc3.bias.copy_(ann.net[4].bias)


def pretreinar_ann_e_converter(
    snn: OrganismoSNN,
    dados_x: torch.Tensor,
    dados_y: torch.Tensor,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> _ANNEquivalente:
    """
    Pré-treina o MLP equivalente com backprop e converte os pesos para a SNN.
    dados_x: [N, N_INPUT], dados_y: [N] (índices de acção)
    Devolve o ANN treinado.
    """
    dev = device or torch.device("cpu")
    ann = _ANNEquivalente().to(dev)
    dados_x = dados_x.to(dev)
    dados_y = dados_y.to(dev)

    optimizador = torch.optim.Adam(ann.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    ann.train()
    for ep in range(epochs):
        optimizador.zero_grad()
        logits = ann(dados_x)
        loss = criterio(logits, dados_y)
        loss.backward()
        optimizador.step()

    converter_ann_para_snn(snn, ann)
    return ann
