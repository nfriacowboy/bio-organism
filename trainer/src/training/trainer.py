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
  SNN_N_AMOSTRAS      (int, default 5000)   — amostras ANN pre-train; 0 desactiva o pre-train
  SNN_N_EPISODIOS     (int, default 500)    — episódios de treino total
  SNN_SAVE_INTERVAL   (int, default 50)     — checkpoint a cada N ep
  SNN_LOG_STEPS       (bool, default 0)     — 1 = gravar steps_log no DB
  SNN_TREINO_NOTA     (str, default "")     — nota livre
  SNN_DEVICE          (str, default "cpu")  — device de treino ("cpu" recomendado para batch=1);
                                              GPU é 3× mais lento para redes pequenas (batch=1)
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone

import torch

from snn import OrganismoSNN, T_STEPS
from snn.ann_init import pretreinar_com_dados_sinteticos
from snn.encoding import encode_full_state
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
EXPORT_DIR    = os.getenv("SNN_EXPORT_DIR", "exports")
# Resume: SNN_RESUME_TREINO_ID + SNN_RESUME_EP + SNN_RESUME_FASE
# Se RESUME_EP > 0, carrega weights do checkpoint e retoma daí.
RESUME_TREINO_ID = os.getenv("SNN_RESUME_TREINO_ID", "")
RESUME_EP        = int(os.getenv("SNN_RESUME_EP",   "0"))
RESUME_FASE      = int(os.getenv("SNN_RESUME_FASE", "3"))
RESUME_MODO      = os.getenv("SNN_RESUME_MODO", "inferencia")  # 'aprendizagem'|'inferencia'

# v4.0: gate de novelty para modo inferência (CV de recompensa por episódio)
_NOVELTY_CV_THR = float(os.getenv("SNN_NOVELTY_CV_THR",   "0.20"))
_NOVELTY_WINDOW = int(os.getenv("SNN_NOVELTY_WINDOW",      "20"))
_NOVELTY_CONSEC = int(os.getenv("SNN_NOVELTY_CONSEC",      "15"))
_NOVELTY_MIN_EP = int(os.getenv("SNN_NOVELTY_MIN_EP_FASE", "50"))
# SNN_NOVELTY_CV_THR   : CV < threshold → candidato a inferência (default 0.20)
# SNN_NOVELTY_WINDOW   : janela de episódios para calcular CV (default 20)
# SNN_NOVELTY_CONSEC   : eps consecutivos com CV baixo antes de entrar em inferência (default 15)
# SNN_NOVELTY_MIN_EP_FASE: eps mínimos em fase antes de poder entrar em inferência (default 50)

# Epsilon-greedy: decai ao longo de todos os episódios
_EPS_INICIO = 0.90   # v38: alto inicial — SNN não domina ep1; p_av≈33%
_EPS_FIM    = 0.15   # v22: floor 15% (era 5%) — evita colapso avançar após AGMP freezes
_EPS_DECAY  = 0.990  # v33: ligeiramente mais rápido (era 0.995) — ep200→15%

# Max steps por fase (mundo maior → mais steps permitidos)
_MAX_STEPS_POR_FASE = {1: 2000, 2: 5000, 3: 10000}

# Estatísticas para KANPolicy
_JANELA_COMIDA   = 50   # steps para contar comida_vista_50steps
_STEPS_SEM_COMER_MAX = 50.0
_N_JANELA_PERSEGUICAO = 20   # steps para converter detecção curta em consumo


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _epsilon(ep: int) -> float:
    return max(_EPS_FIM, _EPS_INICIO * (_EPS_DECAY ** ep))


def _calcular_modo(
    reward_hist: list[float],
    consec: int,
    modo_actual: str,
    ep_na_fase: int,
) -> tuple[str, int]:
    """
    Calcula o modo de plasticidade com base no CV (coeficiente de variação)
    da recompensa por episódio. v4.0.

    Modo 'aprendizagem': R-STDP activo, epsilon normal.
    Modo 'inferencia'  : R-STDP congelado, epsilon=floor.

    Transição para inferência: CV < _NOVELTY_CV_THR por _NOVELTY_CONSEC eps consecutivos.
    Retorno a aprendizagem: imediato quando CV >= _NOVELTY_CV_THR.
    Guardas: nunca inferência antes de ep_na_fase >= _NOVELTY_MIN_EP;
             nunca inferência com recompensa média ≤ 0 (convergência em falha).
    """
    if ep_na_fase < _NOVELTY_MIN_EP or len(reward_hist) < _NOVELTY_WINDOW:
        return "aprendizagem", 0
    mu = sum(reward_hist) / len(reward_hist)
    if mu <= 0.0:
        return "aprendizagem", 0   # convergência em falha — não é inferência real
    var = sum((r - mu) ** 2 for r in reward_hist) / len(reward_hist)
    cv = var ** 0.5 / abs(mu)
    if cv >= _NOVELTY_CV_THR:
        return "aprendizagem", 0
    new_consec = consec + 1
    if new_consec >= _NOVELTY_CONSEC:
        return "inferencia", new_consec
    return modo_actual, new_consec


def _eficiencia(comida: int, num_comida_disponivel: int) -> float:
    """Fracção de comida disponível que foi consumida (0.0–1.0)."""
    if num_comida_disponivel <= 0:
        return 0.0
    return comida / num_comida_disponivel


# Limiar mínimo de intensidade de visão para considerar comida "visível"
_VISAO_THRESHOLD = 0.05

# Ângulos por índice de direcção (mesma ordem do layout de input SNN)
_ANGULOS_VISAO = ["-15.0", "-7.5", "0.0", "7.5", "15.0"]

def _accao_esperada_visao(obs: dict) -> int | None:
    """
    Dado o estado de visão do organismo, devolve a acção correcta para se
    aproximar da comida mais próxima visível, ou None se não há comida visível.

    O dict obs["visao"] tem o formato:
      { "-15.0": {"zona": "curta"|"media"|"longa", "tipo": 1|str, "dist": int}, ... }

    Só considera comida (tipo==1) em zona 'curta' ou 'media'.
    Escolhe o raio com menor distância → acção correcta:
      índices 0,1 (-15°/-7.5°) → virar_esquerda (1)
      índice 2    (0°)          → avancar        (0)
      índices 3,4 (+7.5°/+15°) → virar_direita  (2)
    """
    visao = obs.get("visao")
    if not visao or not isinstance(visao, dict):
        return None

    melhor_idx  = None
    melhor_dist = float("inf")

    for i, ang in enumerate(_ANGULOS_VISAO):
        ray = visao.get(ang, {})
        if ray.get("tipo") != 1:          # não é comida
            continue
        if ray.get("zona", "longa") == "longa":  # demasiado distante / ruído
            continue
        dist = float(ray.get("dist", 999))
        if dist < melhor_dist:
            melhor_dist = dist
            melhor_idx  = i

    if melhor_idx is None:
        return None   # sem comida visível em curta/media

    if melhor_idx <= 1:
        return 1   # virar_esquerda  (-15° ou -7.5°)
    elif melhor_idx == 2:
        return 0   # avancar         (0°)
    else:
        return 2   # virar_direita   (+7.5° ou +15°)


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
    eps_override: float | None = None,
    avancar_bonus: float = 0.0,         # v27: bónus avançar quando anti-avançar collapse detectado
    turn_bonus: float = 0.30,           # v45: constante (sem TURN-OFF)
    wrong_advance_penalty: float = 0.0, # v40: penalizar avançar errado quando p_av>0.35
    wrong_turn_penalty: float = 0.0,    # v55: penalizar viragem na direcção errada
    biased_eps_prob: float = 0.0,       # v56: P(acção_correcta | food_visible & epsilon)
    food_scale: float = 1.0,            # v80i: escalar food reward (ex: 0.10 durante RECOVERY-LOW)
) -> dict:
    """
    Corre um episódio completo.
    Devolve métricas: tempo_vida, comida, eficiencia, recompensa_total,
                      taxa_inferencia, novidade_media, fase_curriculum.
    """
    obs      = mundo.reset()
    learner.reset_traces()

    eps          = eps_override if eps_override is not None else _epsilon(ep)
    max_steps    = _MAX_STEPS_POR_FASE.get(mundo.fase, 2000)
    acc_reward   = 0.0
    acc_novidade = 0.0
    acc_bias_explorar   = 0.0
    acc_bias_exploitar  = 0.0
    n_inferencia = 0
    comida       = 0
    steps_sem_comer = 0
    comida_vista_janela: list[int] = []
    n_vis    = 0   # steps com comida visível nos sensores
    n_correct = 0  # steps com acção correcta quando comida visível
    n_acc    = [0, 0, 0, 0]    # contagem de acções: avancar, virar_esq, virar_dir, parar
    janelas_perseguicao: list[dict] = []   # janelas activas de perseguição
    janelas_fechadas:    list[dict] = []   # janelas expiradas (para taxa_perseguicao)

    for s in range(max_steps):
        dist_ant = mundo._distancia_comida_mais_proxima()

        # Encode obs → SNN input (v2.0: usa organismo.visual_actual, last_visual, last_action)
        energia_norm = obs["energia_norm"]
        x = encode_full_state(mundo.organismo, obs, T=T_STEPS, device=device)
        # v4.0: normalização relativa já aplicada em olfato.py (calcular_olfato) — sem processamento extra aqui.

        # KAN actualiza bias_explorar/bias_exploitar a cada N=10 steps
        stats_kan = {
            "comida_vista_50steps": len(comida_vista_janela),
            "distancia_media_vista": (dist_ant / max(mundo.cfg.largura, 1)) if dist_ant else 0.5,
            "steps_sem_comer":       min(steps_sem_comer, _STEPS_SEM_COMER_MAX),
        }
        estado_kan = observacao_para_estado_kan(obs, stats_kan, device=device)
        kan.actualizar_snn(snn, estado_kan)

        # Acção esperada (para métricas de precisão direcional)
        accao_esp = _accao_esperada_visao(obs)

        # Acção: epsilon-greedy (v56: biased quando comida visível)
        _is_biased = False
        _is_snn = False
        if random.random() < eps:
            if biased_eps_prob > 0.0 and accao_esp is not None and random.random() < biased_eps_prob:
                accao_idx = accao_esp   # biased: escolher acção correcta quando food visible
                _is_biased = True
            elif mundo.fase >= 3:
                # v72: olfato-biased eps em fase esparsa — C. elegans regula pirouettes
                # pelo gradiente químico; sem reward extra → R-STDP aprende via food reward natural.
                _olf = obs.get("olfato", {})
                _oe, _of, _od = _olf.get("esq", 0.0), _olf.get("frente", 0.0), _olf.get("dir", 0.0)
                if _oe + _of + _od >= 0.05:
                    _dom = max(("esq", _oe), ("frente", _of), ("dir", _od), key=lambda t: t[1])[0]
                    accao_idx = 0 if _dom == "frente" else (2 if _dom == "dir" else 1)
                else:
                    accao_idx = random.randint(0, 3)
            else:
                accao_idx = random.randint(0, 3)
        else:
            accao_idx = snn.accao(x)
            _is_snn = True
        # v75: pirouette induction (mecanismo C. elegans gradiente descendente/lateral).
        # Quando SNN escolhe avançar mas olfato dominante não está à frente,
        # sobrepõe probabilisticamente para virar em direcção ao gradiente.
        # P(override) = (olf_nonfron / olf_total) × 0.5 — SNN mantém 50%+ controlo.
        if mundo.fase >= 3 and _is_snn and accao_idx == 0:
            _olf_p = obs.get("olfato", {})
            _oe_p = _olf_p.get("esq", 0.0)
            _of_p = _olf_p.get("frente", 0.0)
            _od_p = _olf_p.get("dir", 0.0)
            _tot_p = _oe_p + _of_p + _od_p
            if _tot_p >= 0.05:
                _nonfron_p = _oe_p + _od_p
                if random.random() < (_nonfron_p / _tot_p) * 0.5:
                    accao_idx = 1 if _oe_p >= _od_p else 2
        # v76: pirouette suppression (mecanismo C. elegans gradiente ascendente/frontal).
        # Quando SNN escolhe virar mas olfato frontal é dominante (>50% total),
        # sobrepõe probabilisticamente para avançar em direcção à comida.
        # P(override) = 0.50 — suprimir pirouette quando gradiente aponta para a frente.
        # R-STDP vê food reward após advance olfato-guiado → aprende olfato_frente→avancar.
        if mundo.fase >= 3 and _is_snn and accao_idx in (1, 2):
            _olf_s = obs.get("olfato", {})
            _of_s = _olf_s.get("frente", 0.0)
            _tot_s = sum(_olf_s.values())
            if _tot_s >= 0.05 and _of_s / _tot_s > 0.50:
                if random.random() < 0.50:
                    accao_idx = 0
        nome_acc = ["avancar", "virar_esquerda", "virar_direita", "parar"][accao_idx]
        n_acc[accao_idx] += 1

        # Precisão direcional: quando há comida visível, foi a acção correcta?
        if accao_esp is not None:
            n_vis += 1
            if accao_idx == accao_esp:
                n_correct += 1

        obs2, renv, done = mundo.step(nome_acc)
        dist_dep = mundo._distancia_comida_mais_proxima()
        # v80i: food×food_scale preserva ANN init; R-STDP aprende via turn_bonus (imediato).
        # wrong_adv=0.30 previne advance-always; RECOVERY-LOW threshold=0.06 (não 0.10)
        # para permitir fases naturais de baixo pav (prec alto ~0.43) sem disrupção.
        _renv_rstdp = renv * food_scale
        r = calcular_recompensa(_renv_rstdp, nome_acc, dist_ant, dist_dep, done,
                                accao_idx=accao_idx, accao_esperada=accao_esp,
                                turn_bonus=(0.0 if _is_biased else turn_bonus),
                                avancar_unconditional=0.0,
                                wrong_advance_penalty=wrong_advance_penalty,
                                wrong_turn_penalty=wrong_turn_penalty)
        if avancar_bonus != 0.0 and accao_idx == 0:   # v79: aceita negativos (cap directo)
            r += avancar_bonus
        # v61: C. elegans starvation dispersal — só fase 2 (em fase 1 prejudica prec; em fase 3
        # domina reward (0.20×~460 steps = 92/ep >> food 1/ep) → R-STDP aprende advance-always)
        _starv_bonus = 0.20 if (energia_norm < 0.3 and mundo.fase == 2) else 0.0
        if _starv_bonus > 0.0 and accao_idx == 0:
            r += _starv_bonus
        # v80p: olfato-conditional bonus em fase 3 — reforça directamente olfato→accao (proporcional
        # à força do sector). Substitui starvation_bonus (dominava reward em fase esparsa).
        # bonus = 0.020 × sector_strength: when olfato_frente=0.6 and organism advances → +0.012;
        # when olfato_esq=0.6 and organism turns left → +0.012. Naturally balanced (no deadlock).
        if mundo.fase >= 3:
            _olf_f3 = obs.get("olfato", {})
            _olf_total_f3 = _olf_f3.get("esq", 0.0) + _olf_f3.get("frente", 0.0) + _olf_f3.get("dir", 0.0)
            if _olf_total_f3 > 0.05:
                _olf_sector = {"avancar": _olf_f3.get("frente", 0.0),
                               "virar_esquerda": _olf_f3.get("esq", 0.0),
                               "virar_direita": _olf_f3.get("dir", 0.0)}.get(nome_acc, 0.0)
                r += 0.020 * _olf_sector

        # v79: advance_bonus passado via avancar_bonus no outer loop (cap directo, ver lá).

        # R-STDP step
        learner.step(x, r, estado=obs, accao=accao_idx, proximo_estado=obs2)
        modo     = learner.modo
        novidade = learner.agmp.novidade

        acc_reward   += r
        acc_novidade += novidade
        acc_bias_explorar  += float(snn.bias_explorar)
        acc_bias_exploitar += float(snn.bias_exploitar)
        if modo == "inferencia":
            n_inferencia += 1

        comeu = renv > 0.5
        if comeu:
            comida += 1
            steps_sem_comer = 0
            comida_vista_janela.append(s)
            # taxa_perseguicao: marcar todas as janelas activas como sucesso
            for j in janelas_perseguicao:
                j["comeu"] = True
        else:
            steps_sem_comer += 1
        # manter janela dos últimos _JANELA_COMIDA items
        comida_vista_janela = [si for si in comida_vista_janela if s - si < _JANELA_COMIDA]

        # taxa_perseguicao: abrir janela quando comida detectada em zona curta
        visao = obs.get("visao", {})
        comida_curta = any(
            r.get("tipo") == 1 and r.get("zona") == "curta"
            for r in visao.values()
        )
        if comida_curta:
            janelas_perseguicao.append({"inicio": s, "comeu": False})
        # fechar janelas expiradas
        expiradas = [j for j in janelas_perseguicao if s - j["inicio"] >= _N_JANELA_PERSEGUICAO]
        janelas_fechadas.extend(expiradas)
        janelas_perseguicao = [j for j in janelas_perseguicao if s - j["inicio"] < _N_JANELA_PERSEGUICAO]

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

    n_steps  = s + 1
    efic     = _eficiencia(comida, mundo.cfg.num_comida)
    taxa_inf = n_inferencia / max(n_steps, 1)
    nov_med  = acc_novidade / max(n_steps, 1)
    precisao = n_correct / n_vis if n_vis > 0 else 0.0
    # taxa_perseguicao: fechar janelas ainda activas no fim do episódio
    janelas_fechadas.extend(janelas_perseguicao)
    n_janelas = len(janelas_fechadas)
    taxa_perseguicao = sum(j["comeu"] for j in janelas_fechadas) / n_janelas if n_janelas > 0 else 0.0
    p_avancar = n_acc[0] / max(n_steps, 1)

    return {
        "tempo_vida":           n_steps,
        "comida":               comida,
        "eficiencia":           efic,
        "recompensa_total":     acc_reward,
        "taxa_inferencia":      taxa_inf,
        "novidade_media":       nov_med,
        "bias_explorar_medio":  acc_bias_explorar  / max(n_steps, 1),
        "bias_exploitar_medio": acc_bias_exploitar / max(n_steps, 1),
        "fase_curriculum":      mundo.fase,
        "morreu":               not mundo.organismo.vivo,
        "precisao_direcional":  precisao,
        "steps_com_visao":      n_vis,
        "taxa_perseguicao":     taxa_perseguicao,
        "seed":                 mundo.seed,
        "p_avancar":            p_avancar,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _device_override = os.getenv("SNN_DEVICE", "cpu").strip().lower()
    device = torch.device(_device_override)
    _fase_inicial = RESUME_FASE if RESUME_EP > 0 else 1
    print(f"[trainer] device={device}  n_episodios={N_EPISODIOS}  fase_inicial={_fase_inicial}")

    # --- Inicializar SNN + KAN + R-STDP ---
    snn = OrganismoSNN(device=device).to(device)
    if RESUME_EP > 0 and RESUME_TREINO_ID:
        # Carregar weights do checkpoint (resume)
        _art = os.path.join(os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlartifacts"), RESUME_TREINO_ID)
        _snn_path = os.path.join(_art, f"snn_ep{RESUME_EP}.pt")
        _kan_path = os.path.join(_art, f"kan_ep{RESUME_EP}.pt")
        _state = torch.load(_snn_path, map_location=device)
        # Compatibilidade 42→45 inputs (Fase C feromona)
        if "fc1.weight" in _state and _state["fc1.weight"].shape[1] == 42:
            _w = _state["fc1.weight"]
            _state["fc1.weight"] = torch.cat([_w, torch.zeros(_w.shape[0], 3, dtype=_w.dtype)], dim=1)
        snn.load_state_dict(_state, strict=True)
        print(f"[trainer] Resume: SNN carregado de {_snn_path}")
        kan = KANPolicy(device=device)
        kan.kan.load_state_dict(torch.load(_kan_path, map_location=device))
        print(f"[trainer] Resume: KAN carregado de {_kan_path}")
    else:
        if N_AMOSTRAS > 0:
            _ann, ann_loss = pretreinar_com_dados_sinteticos(snn, n_amostras=N_AMOSTRAS, device=device)
            print(f"[trainer] ANN pre-train loss: {ann_loss:.3f}")
        else:
            print("[trainer] ANN pre-train desactivado (SNN_N_AMOSTRAS=0) — usando _init_weights")
        kan = KANPolicy(device=device)

    learner = RSTDPLearner(snn, device=device)

    # --- Curriculum ---
    curriculum = CurriculumManager(fase_inicial=_fase_inicial)

    # --- DB / MLflow (opcionais — falha silenciosa se não configurados) ---    treino_id     = datetime.now(timezone.utc).isoformat()
    treino_manager  = None
    mlflow_tracker  = None
    step_logger     = None
    episodio_logger = None

    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            from training.logger import TreinoManager, StepLogger, EpisodioLogger
            treino_manager = TreinoManager(database_url=database_url)
            # create_treino generates its own id and returns it
            treino_id = treino_manager.create_treino(
                parametros={"n_amostras": N_AMOSTRAS, "n_episodios": N_EPISODIOS},
                nota=TREINO_NOTA,
            )
            episodio_logger = EpisodioLogger(treino_id=treino_id, database_url=database_url)
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

    # Checkpoint ep=0 (pesos iniciais) — necessário para evolucao_pesos.csv no export
    # Ignorado em resume (pesos já guardados no treino original)
    if mlflow_tracker is not None and RESUME_EP == 0:
        try:
            mlflow_tracker.log_modelo(
                snn_state_dict=snn.state_dict(),
                kan_state_dict=kan.kan.state_dict(),
                episodio=0,
            )
        except Exception:
            pass

    # --- Loop de treino ---
    mundo = Mundo(fase=curriculum.fase_actual)
    print(f"\n[trainer] Início — fase {curriculum.fase_actual} "
          f"({mundo.cfg.largura}×{mundo.cfg.altura})")

    historico_p_avancar: list[float] = []   # rolling window para alerta v19
    _recovery_counter  = 0   # v19: epsilon recovery após colapso avançar (p_av>85%)
    _pirouette_counter = 0   # v4.0: recovery dedicado para pirouette deadlock (p_av<0.10)
    _ep_fase_inicio    = RESUME_EP if RESUME_EP > 0 else 1   # v62: epsilon por-fase
    # v4.0: gate de novelty (CV de recompensa)
    _reward_hist_cv: list[float] = []
    _consec_low_cv:  int = 0
    _modo: str = RESUME_MODO if RESUME_EP > 0 else "aprendizagem"
    if RESUME_EP > 0:
        learner.set_modo_externo(inferencia=(_modo == "inferencia"))
        print(f"[trainer] Resume ep={RESUME_EP} fase={_fase_inicial} modo={_modo}")

    try:
        for ep in range(RESUME_EP + 1, N_EPISODIOS + 1):
            t0 = time.monotonic()
            learner.set_episode(ep)   # lambda_naca + LR warmup
            _ep_rel = ep - _ep_fase_inicio + 1   # v62: eps por-fase (reinicia a 0.90 ao mudar fase)
            # v4.0: inferência → epsilon=floor; RECOVERY sobrepõe se activo
            if _modo == "inferencia":
                _eps_ov = _EPS_FIM
            elif _pirouette_counter > 0:
                _eps_ov = 0.60   # v80k: eps alto para escapar pirouette (natural ~0.15 insuficiente)
            elif _recovery_counter > 0:
                _eps_ov = 0.50   # v4.0: eps 0.50 para injetar turns quando p_av demasiado alto
            else:
                _eps_ov = _epsilon(_ep_rel)
            # v80i reward shaping: food×0.10 + wrong_adv=0.30 incondicional.
            # RECOVERY-LOW threshold=0.06 permite baixo pav (prec~0.43) sem captura.
            _is_fase1 = (mundo.fase == 1)
            _p_av_roll10 = (
                sum(historico_p_avancar[-10:]) / len(historico_p_avancar[-10:])
                if len(historico_p_avancar) >= 1 else 0.0
            )
            # v80i: wrong_adv=0.30 incondicional — evita advance-always sem causar pirouette
            _wrong_adv = 0.30 if _is_fase1 else 0.0
            # v80k: avancar_bonus=+0.05 durante RECOVERY-LOW (foi 0.20 em v80j → overshooting)
            # +0.05 suficiente para bootstrap sem causar R-STDP "advance always good"
            _adv_bonus_f3 = 0.05 if (_is_fase1 and _pirouette_counter > 0) else 0.0
            # turn_bonus: só fase1, e só quando p_av>=0.06 (alinhado com RECOVERY-LOW thr)
            # pav=0.06-0.10 é zona natural onde prec pica ~0.43 → turn_bonus deve estar activo
            _turn_bonus  = 0.30 if (_is_fase1 and _p_av_roll10 >= 0.06) else 0.0
            metricas = _correr_episodio(
                mundo=mundo,
                snn=snn,
                kan=kan,
                learner=learner,
                ep=ep,
                device=device,
                step_logger=step_logger,
                episodio_global=ep,
                eps_override=_eps_ov,
                avancar_bonus=_adv_bonus_f3,
                turn_bonus=_turn_bonus,
                wrong_advance_penalty=_wrong_adv,
                wrong_turn_penalty=0.0,
                biased_eps_prob=0.0,
                food_scale=0.10 if _is_fase1 else 1.0,
            )
            if _recovery_counter > 0:
                _recovery_counter -= 1
            if _pirouette_counter > 0:
                _pirouette_counter -= 1
            dt = time.monotonic() - t0

            # Consolidar replay entre episódios
            learner.consolidar_replay(batch_size=32)

            # v80k: calcular modo INF com base em CV de precisão (não recompensa)
            # recompensa_total ≈ 0 ou negativa (food×0.10 + wrong_adv) → mu≤0 → cv_c=0 sempre.
            # prec (0.38-0.47) é sempre positivo → mu>0 → CV=std/mean ≈ 0.09 → INF após 15 eps.
            _reward_hist_cv.append(metricas["precisao_direcional"])
            _reward_hist_cv = _reward_hist_cv[-_NOVELTY_WINDOW:]
            _modo, _consec_low_cv = _calcular_modo(
                _reward_hist_cv, _consec_low_cv, _modo, curriculum._episodios_na_fase
            )
            learner.set_modo_externo(inferencia=(_modo == "inferencia"))
            metricas["modo_plasticidade"] = _modo

            # Curriculum: registar e eventualmente avançar fase
            avancou = curriculum.registar_episodio(
                tempo_vida=metricas["tempo_vida"],
                eficiencia=metricas["eficiencia"],
                precisao=metricas["precisao_direcional"],
                em_inferencia=(_modo == "inferencia"),   # v4.0
            )
            if avancou:
                print(f"\n[trainer] *** FASE {curriculum.fase_actual} ***  ep={ep}")
                mundo = Mundo(fase=curriculum.fase_actual)
                # v80n: NÃO reset eps na transição f1→f2.
                # Em v80m, reset _ep_fase_inicio→eps=0.90 em fase2 destruiu directional learning:
                # eps=0.90 + food×1.0 + R-STDP unfrozen → advance-all overwrite (pav=0.88-0.91).
                # Fase 2 herda eps≈0.15 (INF floor) da fase 1 → R-STDP refinamento, não sobrescrita.
                # Só resetar eps para fase>=3 (entrada em mundo radicalmente diferente).
                if curriculum.fase_actual >= 3:
                    _ep_fase_inicio = ep + 1
                # v4.0: nova fase = nova distribuição, resetar gate de novelty
                _reward_hist_cv.clear()
                _consec_low_cv = 0
                _modo = "aprendizagem"
                _pirouette_counter = 0
                learner.set_modo_externo(inferencia=False)

            metricas["fase_curriculum"] = curriculum.fase_actual

            # Logging periódico na consola
            if ep % 10 == 0 or avancou:
                print(
                    f"  ep {ep:4d}: steps={metricas['tempo_vida']:5d}  "
                    f"comida={metricas['comida']:3d}  "
                    f"prec={metricas['precisao_direcional']:.3f}  "
                    f"pers={metricas['taxa_perseguicao']:.3f}  "
                    f"vis={metricas['steps_com_visao']:4d}  "
                    f"pav={metricas.get('p_avancar', 0.0):.2f}  "
                    f"modo={'INF' if _modo == 'inferencia' else 'APR'}  "
                    f"cv_c={_consec_low_cv:2d}  "
                    f"fase={curriculum.fase_actual}  "
                    f"dt={dt:.1f}s"
                )

            # Alerta de colapso de distribuição de acções (v19: epsilon recovery)
            historico_p_avancar.append(metricas.get("p_avancar", 0.0))
            if ep > 50:
                janela = historico_p_avancar[-10:]
                p_av_media = sum(janela) / len(janela)
                # v65: RECOVERY alto-p_av só em fase 1 — em fase 2+ p_av=0.82 É correcto
                # v80i: com food×0.10+wrong_adv, pav≈0.07-0.25; 0.65 é safety net
                if _is_fase1 and p_av_media > 0.65 and _recovery_counter == 0 and not avancou:
                    _recovery_counter = 15
                    print(
                        f"  [RECOVERY-HIGH] ep {ep}: p_avancar={p_av_media:.1%}"
                        f" — eps=0.50 por 15 eps"
                    )
                # v4.0: RECOVERY baixo-p_av — pirouette deadlock (p_av<0.10 em fase1)
                # v80i: threshold 0.10→0.06 — permite pav=0.07-0.09 (prec alto ~0.43)
                # sem disrupção; só true pirouette (<0.06) activa recovery
                elif _is_fase1 and p_av_media < 0.06 and _pirouette_counter == 0 and _modo != "inferencia" and not avancou:
                    _pirouette_counter = 15
                    print(
                        f"  [RECOVERY-LOW] ep {ep}: p_avancar={p_av_media:.1%}"
                        f" — pirouette deadlock, eps=0.60 por 15 eps"
                    )

            # Calibração AGMP no fim do episódio
            calibracao = learner.fim_episodio(ep)
            if calibracao is not None:
                # Guardar threshold calibrado nas métricas do treino no DB
                if treino_manager is not None:
                    try:
                        treino_manager.update_metricas(treino_id, calibracao)
                    except Exception:
                        pass

            # MLflow
            if mlflow_tracker is not None:
                try:
                    mlflow_tracker.log_episodio(ep, metricas)
                except Exception:
                    pass

            # DB: episodios_log (single source of truth)
            if episodio_logger is not None:
                try:
                    episodio_logger.log_episodio(ep, metricas)
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

        if episodio_logger is not None:
            try:
                episodio_logger.close()
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

    # --- Export automático ---
    if database_url:
        try:
            from export.run import run_export
            print(f"\n[trainer] A exportar resultados para '{EXPORT_DIR}/' ...")
            run_export(treino_id=treino_id, output_dir=EXPORT_DIR)
        except Exception as e:
            print(f"[trainer] Export falhou (não crítico): {e}")


if __name__ == "__main__":
    main()
