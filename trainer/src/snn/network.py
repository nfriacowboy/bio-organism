"""
Arquitectura SNN do Organismo.

Topologia v4.0 (loop sensoriomotor + olfacto + feromonas):
  Input  45  (15 visão + 3 estado + 4 eferente + 15 flow + 2 prop + 3 olfato + 3 feromona)
  H1     64  LIF feedforward
  H2     32  RLIF recorrente (ligações laterais)
  Output  4  LIF winner-take-all → avancar | virar_esquerda | virar_direita | parar

Params base: tau=5ms, threshold=0.01 (v20), soft-reset, FastSigmoid(slope=25), T=8
"""
from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# Dimensões fixas da rede
N_INPUT   = 45   # v4.0: 15 visual + 3 estado + 4 eferente + 15 flow + 2 prop + 3 olfato + 3 feromona
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
        threshold: float = 0.01,   # v20: 0.01 — pesos ANN init geram potencial ~0.01-0.03/step; 0.15 causava 100% neurónios mortos
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
        # Com threshold=0.01 e inputs normalizados [0,1], potencial médio >> 0.01 → spikes garantidos.
        nn.init.uniform_(self.fc1.weight, 0.05, 0.25)
        nn.init.constant_(self.fc1.bias, 0.05)

        nn.init.uniform_(self.fc2.weight, 0.03, 0.18)
        nn.init.constant_(self.fc2.bias, 0.02)

        # Output: pesos positivos iniciais para garantir que os neurónios de output
        # disparam desde o início, permitindo que R-STDP aprenda diferenças entre acções.
        nn.init.uniform_(self.fc3.weight, 0.02, 0.10)
        nn.init.zeros_(self.fc3.bias)

        # Recorrência inicializada pequena para estabilidade
        nn.init.uniform_(self.fc2_rec.weight, -0.05, 0.05)

    # -------------------------------------------------------------------
    def set_threshold(self, threshold: float) -> None:
        """Actualiza v_threshold em todas as camadas LIF (lif1, lif2, lif3)."""
        self.lif1.v_threshold = threshold
        self.lif2.v_threshold = threshold
        self.lif3.v_threshold = threshold

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


# ---------------------------------------------------------------------------
# Batched forward para N organismos com pesos diferentes (GPU efficiency)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batched_forward_multi_snn(
    all_snns: list["OrganismoSNN"],
    x_batch: torch.Tensor,    # [N, T, N_INPUT]
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass batched para N organismos com pesos divergentes (evolução).

    Em vez de chamar snn.forward() N vezes sequencialmente (batch=1 cada),
    faz UMA passagem GPU com todos os N organismos em simultâneo usando
    bmm (batch matrix multiply) com os pesos empilhados [N, out, in].

    Com N=50 e GPU: ~5ms totais vs ~250ms sequencial → ~50× speedup.

    Parâmetros:
        all_snns : lista de N OrganismoSNN (cada um com pesos próprios)
        x_batch  : [N, T, N_INPUT] — inputs codificados por organismo
        device   : torch.device (cuda para speedup real)

    Retorna:
        spike_counts : [N, N_OUTPUT]    — contagem de spikes (para winner-take-all)
        h1_spikes    : [T, N, N_H1]    — activações H1 para R-STDP
        h2_spikes    : [T, N, N_H2]    — activações H2 para R-STDP
        out_spikes   : [T, N, N_OUTPUT] — activações output para R-STDP
    """
    N = len(all_snns)
    T = x_batch.shape[1]

    # Stack pesos de todos os organismos: [N, out_dim, in_dim]
    fc1_w  = torch.stack([s.fc1.weight      for s in all_snns]).to(device)  # [N,64,45]
    fc1_b  = torch.stack([s.fc1.bias        for s in all_snns]).to(device)  # [N,64]
    fc2_w  = torch.stack([s.fc2.weight      for s in all_snns]).to(device)  # [N,32,64]
    fc2_b  = torch.stack([s.fc2.bias        for s in all_snns]).to(device)  # [N,32]
    fc2r_w = torch.stack([s.fc2_rec.weight  for s in all_snns]).to(device)  # [N,32,32]
    fc3_w  = torch.stack([s.fc3.weight      for s in all_snns]).to(device)  # [N,4,32]
    fc3_b  = torch.stack([s.fc3.bias        for s in all_snns]).to(device)  # [N,4]

    # Biases KAN por organismo (injectados no output pelo trainer)
    bias_expl = torch.tensor([s.bias_explorar  for s in all_snns], device=device)  # [N]
    bias_expt = torch.tensor([s.bias_exploitar for s in all_snns], device=device)  # [N]

    # Parâmetros LIF (tau/threshold iguais em todos os organismos por design)
    tau       = float(all_snns[0].lif1.tau)
    threshold = float(all_snns[0].lif1.v_threshold)
    decay     = 1.0 - 1.0 / tau   # SpikingJelly: H = v*(1-1/tau) + X*(1/tau)
    inp_scale = 1.0 / tau

    # Estados LIF inicializados a zero [N, H] — equivalente a functional.reset_net()
    v1    = torch.zeros(N, N_H1,     device=device)
    v2    = torch.zeros(N, N_H2,     device=device)
    v3    = torch.zeros(N, N_OUTPUT, device=device)
    h2_sp = torch.zeros(N, N_H2,    device=device)  # spikes recorrentes H2(t-1)

    spike_out_accum = torch.zeros(N, N_OUTPUT, device=device)
    h1_list:  list[torch.Tensor] = []
    h2_list:  list[torch.Tensor] = []
    out_list: list[torch.Tensor] = []

    xb = x_batch.to(device)  # [N, T, 45]

    for t in range(T):
        xt = xb[:, t, :]   # [N, 45]

        # H1: bmm(fc1_w [N,64,45], xt [N,45,1]) → [N,64] + bias
        h1_pre = torch.bmm(fc1_w, xt.unsqueeze(-1)).squeeze(-1) + fc1_b
        v1 = v1 * decay + h1_pre * inp_scale
        s1 = (v1 >= threshold).float()
        v1 = v1 - s1 * threshold   # soft-reset

        # H2: linear + recorrência + LIF
        h2_pre = torch.bmm(fc2_w,  s1.unsqueeze(-1)).squeeze(-1) + fc2_b
        h2_rec = torch.bmm(fc2r_w, h2_sp.unsqueeze(-1)).squeeze(-1)
        v2 = v2 * decay + (h2_pre + h2_rec) * inp_scale
        s2 = (v2 >= threshold).float()
        v2 = v2 - s2 * threshold
        h2_sp = s2.detach()

        # Output: linear + biases KAN + LIF
        out_pre = torch.bmm(fc3_w, s2.unsqueeze(-1)).squeeze(-1) + fc3_b  # [N,4]
        out_pre[:, 0] += bias_expt * 0.2
        out_pre[:, 1] += bias_expl * 0.2
        out_pre[:, 2] += bias_expl * 0.2
        v3 = v3 * decay + out_pre * inp_scale
        s3 = (v3 >= threshold).float()
        v3 = v3 - s3 * threshold

        spike_out_accum += s3
        h1_list.append(s1.unsqueeze(0))    # [1, N, 64]
        h2_list.append(s2.unsqueeze(0))    # [1, N, 32]
        out_list.append(s3.unsqueeze(0))   # [1, N, 4]

    h1_spikes  = torch.cat(h1_list,  dim=0)   # [T, N, 64]
    h2_spikes  = torch.cat(h2_list,  dim=0)   # [T, N, 32]
    out_spikes = torch.cat(out_list, dim=0)   # [T, N, 4]

    # Popular _spike_rates em cada SNN para AGMP.taxa_spikes_media
    # 2 reduções GPU + 1 CPU transfer cada (vs 160 .item() calls individuais)
    h1_rates = h1_spikes.mean(dim=2).cpu()   # [T, N]
    h2_rates = h2_spikes.mean(dim=2).cpu()   # [T, N]
    for i, snn in enumerate(all_snns):
        snn._spike_rates_h1 = h1_rates[:, i].tolist()
        snn._spike_rates_h2 = h2_rates[:, i].tolist()

    return spike_out_accum, h1_spikes, h2_spikes, out_spikes


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
