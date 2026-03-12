"""
Script de validação do Passo 6: R-STDP básico.
Treina o organismo na Fase 1 até atingir ≥200 steps de vida média em 10 episódios consecutivos.
Usa epsilon-greedy para exploração.
"""
from __future__ import annotations

import sys
import random
import torch
import numpy as np

sys.path.insert(0, "trainer/src")

from world import Mundo, ACCOES, ACCAO_IDX
from snn.network import OrganismoSNN, T_STEPS
from snn.encoding import observacao_para_tensor
from snn.learning import RSTDPLearner, ConfigRSTDP, calcular_recompensa
from snn.ann_init import pretreinar_com_dados_sinteticos

SEED_BASE      = 42
MAX_EPISODIOS  = 600
META_VIDA      = 200.0
JANELA         = 10
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Epsilon-greedy: começa em 0.4, decai para 0.05
EPSILON_INI    = 0.4
EPSILON_FIM    = 0.05
EPSILON_DECAY  = 0.98   # multiplicativo por episódio


def escolher_accao(snn: OrganismoSNN, x: torch.Tensor, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(len(ACCOES))
    with torch.no_grad():
        counts = snn(x)
    return int(counts.squeeze(0).argmax().item())


def treinar() -> bool:
    random.seed(SEED_BASE)
    torch.manual_seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    snn = OrganismoSNN(device=DEVICE).to(DEVICE)
    cfg = ConfigRSTDP(lr_snn=2e-3, tau_trace=20.0, clip_w=2.0, grad_clip=0.3)
    learner = RSTDPLearner(snn, cfg, DEVICE)

    # ANN-to-SNN initialization (spec §2.2)
    print("Pré-treino ANN→SNN com dados sintéticos...")
    _, loss_final = pretreinar_com_dados_sinteticos(snn, n_amostras=5000, epochs=5, device=DEVICE)
    print(f"  Loss final ANN: {loss_final:.4f}\n")

    historico_vida: list[float] = []
    historico_comida: list[float] = []
    meta_atingida = False
    epsilon = EPSILON_INI

    print(f"Device: {DEVICE} | LR={cfg.lr_snn} | tau_trace={cfg.tau_trace}")
    print(f"Meta: {META_VIDA} steps médios em {JANELA} episódios consecutivos")
    print(f"Exploração: epsilon={EPSILON_INI}→{EPSILON_FIM} (decay={EPSILON_DECAY})\n")

    for ep in range(1, MAX_EPISODIOS + 1):
        seed_ep = SEED_BASE + ep
        mundo = Mundo(fase=1, seed=seed_ep)
        obs = mundo.reset()

        learner.reset_traces()
        steps_vivos = 0
        tempo_vida_norm = 0.0

        while mundo.organismo.vivo:
            x = observacao_para_tensor(obs, T=T_STEPS, device=DEVICE,
                                       tempo_vida_norm=tempo_vida_norm)

            dist_antes = mundo._distancia_comida_mais_proxima()
            accao_idx  = escolher_accao(snn, x, epsilon)
            accao      = ACCOES[accao_idx]

            obs_novo, reward_amb, terminado = mundo.step(accao)
            dist_depois = mundo._distancia_comida_mais_proxima()

            recompensa = calcular_recompensa(
                recompensa_ambiente=reward_amb,
                accao=accao,
                dist_comida_antes=dist_antes,
                dist_comida_depois=dist_depois,
                morreu=terminado,
            )

            learner.step(x, recompensa)

            steps_vivos += 1
            tempo_vida_norm = min(steps_vivos / 2000.0, 1.0)
            obs = obs_novo

            if terminado:
                break

        epsilon = max(EPSILON_FIM, epsilon * EPSILON_DECAY)

        historico_vida.append(float(steps_vivos))
        historico_vida = historico_vida[-JANELA:]
        historico_comida.append(float(mundo.comida_comida))
        historico_comida = historico_comida[-JANELA:]

        media = sum(historico_vida) / len(historico_vida)
        media_comida = sum(historico_comida) / len(historico_comida)

        if ep % 20 == 0 or ep <= 5:
            taxa = snn.taxa_spikes_media
            print(f"  Ep {ep:4d} | steps={steps_vivos:5d} | "
                  f"média={media:6.1f} | comida={mundo.comida_comida} "
                  f"(med={media_comida:.1f}) | ε={epsilon:.3f} | spikes={taxa:.3f}")

        if len(historico_vida) == JANELA and media >= META_VIDA:
            print(f"\n✓ META ATINGIDA no episódio {ep}!")
            print(f"  Média últimos {JANELA} episódios: {media:.1f} steps")
            print(f"  Comida média: {media_comida:.1f} peças/episódio")
            meta_atingida = True
            break

    if not meta_atingida:
        media_final = sum(historico_vida) / len(historico_vida)
        print(f"\n✗ Meta não atingida após {MAX_EPISODIOS} episódios.")
        print(f"  Melhor média recente: {media_final:.1f} steps (meta: {META_VIDA})")

    return meta_atingida


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    ok = treinar()
    sys.exit(0 if ok else 1)
