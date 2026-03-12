"""
Passo 14 — Trainer principal com Curriculum Learning (Fases 1→2→3).

Loop de treino:
  1. Pré-treino ANN-to-SNN com dados sintéticos.
  2. Episódios de treino com R-STDP + NACA + AGMP + KANPolicy.
  3. CurriculumManager avança fase quando threshold é atingido.
  4. Logs por step → PostgreSQL (StepLogger).
  5. Métricas por episódio → MLflow (MLflowTracker).
  6. Checkpoints periódicos a cada SAVE_INTERVAL episódios.

Variáveis de ambiente (obrigatórias quando DB/MLflow activos):
  DATABASE_URL, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

Variáveis opcionais:
  SNN_N_AMOSTRAS      (int, default 5000)   — amostras ANN pre-train
  SNN_N_EPISODIOS     (int, default 500)    — episódios de treino total
  SNN_SAVE_INTERVAL   (int, default 50)     — checkpoint a cada N ep
  SNN_LOG_STEPS       (bool, default 0)     — 1 = gravar steps_log no DB
  SNN_TREINO_NOTA     (str, default "")     — nota livre
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone

import torch

from snn import OrganismoSNN, T_STEPS
from snn.ann_init import pretreinar_com_dados_sinteticos
from snn.encoding import observacao_para_tensor
from snn.kan_policy import KANPolicy, observacao_para_estado_kan
from snn.learning import RSTDPLearner, calcular_recompensa
from world import Mundo
from world.curriculum import CurriculumManager, FASES

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------

N_AMOSTRAS    = int(os.getenv("SNN_N_AMOSTRAS",    "5000"))
N_EPISODIOS   = int(os.getenv("SNN_N_EPISODIOS",   "500"))
SAVE_INTERVAL = int(os.getenv("SNN_SAVE_INTERVAL", "50"))
LOG_STEPS     = os.getenv("SNN_LOG_STEPS", "0").strip() in ("1", "true", "yes")
TREINO_NOTA   = os.getenv("SNN_TREINO_NOTA", "")

# Epsilon-greedy: decai ao longo de todos os episódios
_EPS_INICIO = 0.40
_EPS_FIM    = 0.05
_EPS_DECAY  = 0.98

# Max steps por fase (mundo maior → mais steps permitidos)
_MAX_STEPS_POR_FASE = {1: 2000, 2: 5000, 3: 10000}

# Estatísticas para KANPolicy
_JANELA_COMIDA   = 50   # steps para contar comida_vista_50steps
_STEPS_SEM_COMER_MAX = 50.0


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _epsilon(ep: int) -> float:
    return max(_EPS_FIM, _EPS_INICIO * (_EPS_DECAY ** ep))


def _eficiencia(comida: int, energia_gasta: float) -> float:
    if energia_gasta <= 0:
        return 0.0
    return comida / max(energia_gasta, 1.0)


# ---------------------------------------------------------------------------
# Loop de episódio
# ---------------------------------------------------------------------------

def _correr_episodio(
    mundo: Mundo,
    snn: OrganismoSNN,
    kan: KANPolicy,
    learner: RSTDPLearner,
    ep: int,
    device: torch.device,
    step_logger=None,
    episodio_global: int = 0,
) -> dict:
    """
    Corre um episódio completo.
    Devolve métricas: tempo_vida, comida, eficiencia, recompensa_total,
                      taxa_inferencia, novidade_media, fase_curriculum.
    """
    obs      = mundo.reset()
    learner.reset_traces()

    eps          = _epsilon(ep)
    max_steps    = _MAX_STEPS_POR_FASE.get(mundo.fase, 2000)
    acc_reward   = 0.0
    acc_novidade = 0.0
    n_inferencia = 0
    comida       = 0
    steps_sem_comer = 0
    comida_vista_janela: list[int] = []

    for s in range(max_steps):
        dist_ant = mundo._distancia_comida_mais_proxima()

        # Encode obs → SNN input
        energia_norm = obs["energia_norm"]
        x = observacao_para_tensor(obs, T=T_STEPS, device=device)

        # KAN actualiza bias_explorar/bias_exploitar a cada N=10 steps
        stats_kan = {
            "comida_vista_50steps": len(comida_vista_janela),
            "distancia_media_vista": (dist_ant / max(mundo.cfg.largura, 1)) if dist_ant else 0.5,
            "steps_sem_comer":       min(steps_sem_comer, _STEPS_SEM_COMER_MAX),
        }
        estado_kan = observacao_para_estado_kan(obs, stats_kan, device=device)
        kan.actualizar_snn(snn, estado_kan)

        # Acção: epsilon-greedy
        if random.random() < eps:
            accao_idx = random.randint(0, 3)
        else:
            accao_idx = snn.accao(x)
        nome_acc = ["avancar", "virar_esquerda", "virar_direita", "parar"][accao_idx]

        obs2, renv, done = mundo.step(nome_acc)
        dist_dep = mundo._distancia_comida_mais_proxima()
        r = calcular_recompensa(renv, nome_acc, dist_ant, dist_dep, done)

        # R-STDP step
        learner.step(x, r, estado=obs, accao=accao_idx, proximo_estado=obs2)
        modo     = learner.modo
        novidade = learner.agmp.novidade

        acc_reward   += r
        acc_novidade += novidade
        if modo == "inferencia":
            n_inferencia += 1

        comeu = renv > 0.5
        if comeu:
            comida += 1
            steps_sem_comer = 0
            comida_vista_janela.append(s)
        else:
            steps_sem_comer += 1
        # manter janela dos últimos _JANELA_COMIDA items
        comida_vista_janela = [si for si in comida_vista_janela if s - si < _JANELA_COMIDA]

        # Log ao PostgreSQL (opcional)
        if step_logger is not None:
            obs_log = dict(obs)
            obs_log["comeu"] = comeu
            try:
                step_logger.log_step(
                    episodio=episodio_global,
                    step=s,
                    obs=obs_log,
                    accao=nome_acc,
                    modo=modo,
                    novidade=novidade,
                    recompensa=r,
                    bias_explorar=float(snn.bias_explorar),
                    bias_exploitar=float(snn.bias_exploitar),
                )
            except Exception:
                pass   # não interromper treino por falha de log

        obs = obs2
        if done:
            break

    n_steps = s + 1
    efic    = _eficiencia(comida, mundo.energia_gasta)
    taxa_inf = n_inferencia / max(n_steps, 1)
    nov_med  = acc_novidade / max(n_steps, 1)

    return {
        "tempo_vida":      n_steps,
        "comida":          comida,
        "eficiencia":      efic,
        "recompensa_total": acc_reward,
        "taxa_inferencia": taxa_inf,
        "novidade_media":  nov_med,
        "fase_curriculum": mundo.fase,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trainer] device={device}  n_episodios={N_EPISODIOS}  fase_inicial=1")

    # --- Inicializar SNN + KAN + R-STDP ---
    snn = OrganismoSNN(device=device).to(device)
    _ann, ann_loss = pretreinar_com_dados_sinteticos(snn, n_amostras=N_AMOSTRAS, device=device)
    print(f"[trainer] ANN pre-train loss: {ann_loss:.3f}")

    kan     = KANPolicy(device=device)
    learner = RSTDPLearner(snn, device=device)

    # --- Curriculum ---
    curriculum = CurriculumManager(fase_inicial=1)

    # --- DB / MLflow (opcionais — falha silenciosa se não configurados) ---
    treino_id     = datetime.now(timezone.utc).isoformat()
    treino_manager = None
    mlflow_tracker = None
    step_logger    = None

    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            from training.logger import TreinoManager, StepLogger
            treino_manager = TreinoManager(database_url=database_url)
            # create_treino generates its own id and returns it
            treino_id = treino_manager.create_treino(
                parametros={"n_amostras": N_AMOSTRAS, "n_episodios": N_EPISODIOS},
                nota=TREINO_NOTA,
            )
            if LOG_STEPS:
                step_logger = StepLogger(treino_id=treino_id, database_url=database_url)
            print(f"[trainer] DB: treino criado id={treino_id}")
        except Exception as e:
            print(f"[trainer] DB indisponível: {e}")
            treino_manager = None

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        try:
            from training.mlflow_tracker import MLflowTracker
            mlflow_tracker = MLflowTracker(
                treino_id=treino_id,
                parametros={"n_amostras": N_AMOSTRAS, "n_episodios": N_EPISODIOS},
            )
            print(f"[trainer] MLflow: run_id={mlflow_tracker.mlflow_run_id}")
        except Exception as e:
            print(f"[trainer] MLflow indisponível: {e}")
            mlflow_tracker = None

    # --- Loop de treino ---
    mundo = Mundo(fase=curriculum.fase_actual)
    print(f"\n[trainer] Início — fase {curriculum.fase_actual} "
          f"({mundo.cfg.largura}×{mundo.cfg.altura})")

    try:
        for ep in range(1, N_EPISODIOS + 1):
            t0 = time.monotonic()
            metricas = _correr_episodio(
                mundo=mundo,
                snn=snn,
                kan=kan,
                learner=learner,
                ep=ep,
                device=device,
                step_logger=step_logger,
                episodio_global=ep,
            )
            dt = time.monotonic() - t0

            # Consolidar replay entre episódios
            learner.consolidar_replay(batch_size=32)

            # Curriculum: registar e eventualmente avançar fase
            avancou = curriculum.registar_episodio(
                tempo_vida=metricas["tempo_vida"],
                eficiencia=metricas["eficiencia"],
            )
            if avancou:
                print(f"\n[trainer] *** FASE {curriculum.fase_actual} ***  ep={ep}")
                mundo = Mundo(fase=curriculum.fase_actual)

            metricas["fase_curriculum"] = curriculum.fase_actual

            # Logging periódico na consola
            if ep % 10 == 0 or avancou:
                print(
                    f"  ep {ep:4d}: steps={metricas['tempo_vida']:5d}  "
                    f"comida={metricas['comida']:3d}  "
                    f"efic={metricas['eficiencia']:.3f}  "
                    f"modo={learner.modo}  "
                    f"fase={curriculum.fase_actual}  "
                    f"dt={dt:.1f}s"
                )

            # MLflow
            if mlflow_tracker is not None:
                try:
                    mlflow_tracker.log_episodio(ep, metricas)
                except Exception:
                    pass

            # DB: actualizar metricas_finais
            if treino_manager is not None and ep % 50 == 0:
                try:
                    treino_manager.update_metricas(treino_id, metricas)
                except Exception:
                    pass

            # Checkpoint periódico
            if ep % SAVE_INTERVAL == 0:
                if mlflow_tracker is not None:
                    try:
                        mlflow_tracker.log_modelo(
                            snn_state_dict=snn.state_dict(),
                            kan_state_dict=kan.kan.state_dict(),
                            episodio=ep,
                        )
                        print(f"  [checkpoint] ep={ep} guardado")
                    except Exception as e:
                        print(f"  [checkpoint] falha: {e}")

    finally:
        # Fechar loggers
        if step_logger is not None:
            try:
                step_logger.close()
            except Exception:
                pass

        if treino_manager is not None:
            try:
                treino_manager.update_estado(treino_id, "concluido")
            except Exception:
                pass

        if mlflow_tracker is not None:
            try:
                mlflow_tracker.log_modelo(
                    snn_state_dict=snn.state_dict(),
                    kan_state_dict=kan.kan.state_dict(),
                    episodio=N_EPISODIOS,
                )
                mlflow_tracker.end()
            except Exception:
                pass

    print(f"\n[trainer] Treino concluído. Fase final: {curriculum.fase_actual}")


if __name__ == "__main__":
    main()
