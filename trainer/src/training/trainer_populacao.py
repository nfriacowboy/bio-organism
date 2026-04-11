"""
Trainer de População — Mundo v2.0 + Fase A (Reprodução).

Corre N organismos simultâneos num mundo com patches dinâmicos de comida.
Cada organismo tem SNN + KAN + RSTDPLearner independentes, inicializados
a partir dos pesos do organismo base (base_v1).

Fase A: reprodução contínua — quando um organismo morre, substitui-o por
um filho de dois progenitores vivos (crossover + mutação).

Variáveis de ambiente:
  SNN_N_ORGANISMOS     (int, default 10) — tamanho da população
  SNN_N_EPISODIOS      (int, default 500)
  SNN_SAVE_INTERVAL    (int, default 50)
  SNN_DEVICE           (str, default "cpu")
  SNN_BASE_SNN_PATH    (str) — path para snn_base_v1.pt
  SNN_BASE_KAN_PATH    (str) — path para kan_base_v1.pt
  SNN_MUTACAO_SIGMA    (float, default 0.01) — desvio padrão da mutação
  SNN_USAR_PATCHES     (bool, default 1) — patches dinâmicos de comida
  SNN_USAR_PREDADOR    (bool, default 0) — activar Fase C.2 predador
  SNN_REPRODUCAO       (bool, default 1) — activar Fase A reprodução
  SNN_TREINO_NOTA      (str, default "")
  SNN_FASE_INICIAL     (int, default 1)  — fase de início do curriculum (1|2|3)
  DATABASE_URL         (str) — URL PostgreSQL para logging de métricas
  SNN_SNAPSHOT_INTERVAL (int, default 10) — intervalo de episódios para snapshot de posições
"""
from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch

from snn import OrganismoSNN, T_STEPS
from snn.network import batched_forward_multi_snn
from snn.encoding import encode_full_state
from snn.kan_policy import KANPolicy, observacao_para_estado_kan
from snn.learning import RSTDPLearner
from world import Mundo
from world.curriculum import CurriculumManager
from world.environment import Organismo
from world.reproduction import (
    podem_reproduzir, executar_reproducao, calcular_diversidade,
    cosine_sim_snn, INBREEDING_COS_THR,
    MUTACAO_SIGMA_DEFAULT,
)

# ---------------------------------------------------------------------------
# Parâmetros via env vars
# ---------------------------------------------------------------------------

N_ORGANISMOS   = int(os.getenv("SNN_N_ORGANISMOS",   "10"))
N_EPISODIOS    = int(os.getenv("SNN_N_EPISODIOS",     "500"))
SAVE_INTERVAL  = int(os.getenv("SNN_SAVE_INTERVAL",  "50"))
DEVICE_STR     = os.getenv("SNN_DEVICE", "cpu")

_DEFAULT_BASE  = os.path.join(os.path.dirname(__file__), "..", "..", "mlartifacts", "base_v1")
BASE_SNN_PATH  = os.getenv("SNN_BASE_SNN_PATH", os.path.join(_DEFAULT_BASE, "snn_base_v1.pt"))
BASE_KAN_PATH  = os.getenv("SNN_BASE_KAN_PATH", os.path.join(_DEFAULT_BASE, "kan_base_v1.pt"))

MUTACAO_SIGMA  = float(os.getenv("SNN_MUTACAO_SIGMA",  str(MUTACAO_SIGMA_DEFAULT)))
USAR_PATCHES   = os.getenv("SNN_USAR_PATCHES",   "1").strip() in ("1", "true", "yes")
USAR_PREDADOR  = os.getenv("SNN_USAR_PREDADOR",  "0").strip() in ("1", "true", "yes")
USAR_REPRODUCAO = os.getenv("SNN_REPRODUCAO",    "1").strip() in ("1", "true", "yes")
TREINO_NOTA       = os.getenv("SNN_TREINO_NOTA", "")
DATABASE_URL      = os.getenv("DATABASE_URL", "")
SNAPSHOT_INTERVAL = int(os.getenv("SNN_SNAPSHOT_INTERVAL", "10"))
FASE_INICIAL      = int(os.getenv("SNN_FASE_INICIAL", "1"))

# Epsilon-greedy (igual ao trainer.py)
_EPS_INICIO = 0.90
_EPS_FIM    = 0.15
_EPS_DECAY  = 0.990

# Max steps por fase
_MAX_STEPS_POR_FASE = {1: 2000, 2: 5000, 3: 10000}

# Estatísticas KAN
_STEPS_SEM_COMER_MAX = 50.0
_JANELA_COMIDA       = 50

# Diversidade calculada a cada N episódios (custosa para N=10+)
_DIVERSIDADE_INTERVALO = 10


# ---------------------------------------------------------------------------
# DB helpers (opcionais — activos apenas se DATABASE_URL estiver definido)
# ---------------------------------------------------------------------------

def _db_connect():
    """Abre conexão psycopg2; devolve None se DATABASE_URL não está definido."""
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        print(f"[DB] Conexão estabelecida", flush=True)
        return conn
    except Exception as e:
        print(f"[DB] Conexão falhou: {e}", flush=True)
        return None


def _log_episodio_pop(conn, treino_id: str, ep: int, fase: int,
                      pop: dict, n_nasc: int, div_cv: float,
                      geracao_max: int, elapsed: float) -> None:
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO episodios_populacao_log
                       (treino_id, episodio, fase, n_organismos,
                        comida_media, comida_total, comida_max,
                        vida_media, vida_max, eps_valor, elapsed_s,
                        geracao_max, n_nascimentos, cv_fc1, geracao_media)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (treino_id, ep, fase, N_ORGANISMOS,
                 pop["comida_media"], pop["comida_total"], pop["comida_max"],
                 pop["vida_media"], pop["vida_max"],
                 _epsilon(ep), elapsed,
                 geracao_max, n_nasc, div_cv,
                 pop.get("geracao_media", 0.0)),
            )
        conn.commit()
    except Exception as e:
        print(f"[DB] log_episodio falhou: {e}", flush=True)
        try:
            conn.rollback()
        except Exception:
            pass


def _log_snapshot(conn, treino_id: str, ep: int, mundo) -> None:
    """Regista posições de todos os organismos para visualização tribal."""
    if conn is None:
        return
    try:
        import psycopg2.extras
        rows = [
            (treino_id, ep, i,
             org.x, org.y, org.angulo,
             org.geracao, org.idade,
             org.comida_lifetime, org.vivo)
            for i, org in enumerate(mundo.organismos)
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO snapshots_log
                       (treino_id, episodio, organismo_idx, x, y,
                        angulo, geracao, idade, comida_lifetime, vivo)
                   VALUES %s""",
                rows,
            )
        conn.commit()
    except Exception as e:
        print(f"[DB] log_snapshot falhou: {e}", flush=True)
        try:
            conn.rollback()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _epsilon(ep: int) -> float:
    return max(_EPS_FIM, _EPS_INICIO * (_EPS_DECAY ** ep))


def _criar_organismo(device: torch.device) -> tuple[OrganismoSNN, KANPolicy, RSTDPLearner]:
    """Cria SNN + KAN + Learner inicializados a partir do base_v1.

    Suporta carregamento de checkpoints com N_INPUT=42 (base_v1) no SNN com N_INPUT=45 (Fase C):
    os 3 novos inputs de feromona são inicializados a zero (o organismo parte sem resposta à feromona
    e aprende gradualmente via R-STDP).
    """
    snn = OrganismoSNN(device=device).to(device)
    if os.path.exists(BASE_SNN_PATH):
        state = torch.load(BASE_SNN_PATH, map_location=device, weights_only=True)
        # Compatibilidade: se fc1.weight tem shape [64, 42] (base_v1), expandir para [64, 45]
        if "fc1.weight" in state and state["fc1.weight"].shape[1] == 42:
            w = state["fc1.weight"]                           # [64, 42]
            pad = torch.zeros(w.shape[0], 3, dtype=w.dtype, device=w.device)
            state["fc1.weight"] = torch.cat([w, pad], dim=1)  # [64, 45]
        snn.load_state_dict(state, strict=True)
    kan = KANPolicy(device=device)
    if os.path.exists(BASE_KAN_PATH):
        kan.kan.load_state_dict(torch.load(BASE_KAN_PATH, map_location=device, weights_only=True))

    # Mutação inicial para diversificar população de partida (Fase A bootstrap)
    if MUTACAO_SIGMA > 0.0:
        rng_np = np.random.default_rng()
        with torch.no_grad():
            for p in snn.parameters():
                noise = torch.tensor(rng_np.normal(0.0, MUTACAO_SIGMA, p.shape), dtype=p.dtype)
                p.add_(noise.to(p.device))

    learner = RSTDPLearner(snn, device=device)
    return snn, kan, learner


def _guardar_checkpoints(
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]],
    ep: int,
    treino_id: str,
) -> None:
    pasta = os.path.join("checkpoints_pop", treino_id.replace(":", "_").replace("+", ""))
    os.makedirs(pasta, exist_ok=True)
    for i, (snn, kan, _) in enumerate(populacao):
        torch.save(snn.state_dict(), os.path.join(pasta, f"ep{ep:04d}_org{i:02d}_snn.pt"))
        torch.save(kan.kan.state_dict(), os.path.join(pasta, f"ep{ep:04d}_org{i:02d}_kan.pt"))


# ---------------------------------------------------------------------------
# Reprodução contínua (Fase A)
# ---------------------------------------------------------------------------

def _spawn_filho(
    mundo: Mundo,
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]],
    slot_morto: int,
    rng_np: np.random.Generator,
    device: torch.device,
    ep: int,
    geracao_max: int,
) -> int:
    """
    Tenta substituir o organismo morto no slot_morto por um filho de dois progenitores.

    Procura pares de organismos vivos que satisfaçam as condições de reprodução.
    Se não encontrar par compatível, usa clone mutado do melhor progenitor (fitness=comida_lifetime).

    Retorna a nova geração do filho criado (ou geracao_max se fallback).
    """
    vivos = [(i, mundo.organismos[i]) for i in range(len(mundo.organismos))
             if i != slot_morto and mundo.organismos[i].vivo]

    if len(vivos) < 2:
        # Fallback: sem progenitores suficientes — criar cópia mutada do base_v1
        return _spawn_base_v1(mundo, populacao, slot_morto, rng_np, device, geracao_max)

    # Procurar par compatível para reprodução
    par = None
    # Embaralhar para não favorecer sempre os mesmos
    indices_vivos = list(range(len(vivos)))
    rng_np.shuffle(indices_vivos)
    for ii in range(len(indices_vivos)):
        for jj in range(ii + 1, len(indices_vivos)):
            i_a = vivos[indices_vivos[ii]][0]
            i_b = vivos[indices_vivos[jj]][0]
            org_a, org_b = mundo.organismos[i_a], mundo.organismos[i_b]
            if podem_reproduzir(org_a, org_b):
                # Fase B.2 — anti-endogamia: rejeitar pares geneticamente demasiado similares
                snn_a_chk = populacao[i_a][0]
                snn_b_chk = populacao[i_b][0]
                if cosine_sim_snn(snn_a_chk, snn_b_chk) < INBREEDING_COS_THR:
                    par = (i_a, i_b)
                    break
        if par:
            break

    if par is None:
        # Sem par compatível: fallback para melhor organismo vivo + mutação
        if vivos:
            melhor_i = max(vivos, key=lambda t: t[1].comida_lifetime)[0]
            return _spawn_clone_mutado(mundo, populacao, slot_morto, melhor_i, rng_np, device, geracao_max)
        return _spawn_base_v1(mundo, populacao, slot_morto, rng_np, device, geracao_max)

    i_a, i_b = par
    org_a, org_b = mundo.organismos[i_a], mundo.organismos[i_b]
    snn_a, kan_a, learner_a = populacao[i_a]
    snn_b, kan_b, learner_b = populacao[i_b]

    nova_geracao = max(org_a.geracao, org_b.geracao) + 1
    snn_f, kan_f, learner_f = executar_reproducao(
        org_a, org_b, snn_a, kan_a, learner_a,
        snn_b, kan_b, learner_b,
        novo_id=slot_morto, nova_geracao=nova_geracao,
        device=device, rng=rng_np, sigma=MUTACAO_SIGMA,
    )

    # Criar novo Organismo no mundo em posição aleatória vazia
    _inserir_organismo_mundo(mundo, slot_morto, nova_geracao, rng_np)
    populacao[slot_morto] = (snn_f, kan_f, learner_f)
    return nova_geracao


def _spawn_base_v1(
    mundo: Mundo,
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]],
    slot: int,
    rng_np: np.random.Generator,
    device: torch.device,
    geracao_max: int,
) -> int:
    """Fallback: inserir clone base_v1 mutado."""
    snn, kan, learner = _criar_organismo(device)
    _inserir_organismo_mundo(mundo, slot, geracao_max, rng_np)
    populacao[slot] = (snn, kan, learner)
    return geracao_max


def _spawn_clone_mutado(
    mundo: Mundo,
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]],
    slot: int,
    src_i: int,
    rng_np: np.random.Generator,
    device: torch.device,
    geracao_max: int,
) -> int:
    """Fallback: clone mutado de um progenitor."""
    from world.reproduction import get_genome, set_genome, mutate
    snn_src, kan_src, learner_src = populacao[src_i]
    g = get_genome(snn_src, kan_src)
    g = mutate(g, MUTACAO_SIGMA * 2, rng_np)  # sigma duplo para manter diversidade

    snn_f = OrganismoSNN(device=device).to(device)
    kan_f = KANPolicy(device=device)
    set_genome(snn_f, kan_f, g, device)
    learner_f = RSTDPLearner(snn_f, device=device)
    learner_f._reward_baseline = learner_src._reward_baseline * 0.3

    nova_g = mundo.organismos[src_i].geracao + 1
    _inserir_organismo_mundo(mundo, slot, nova_g, rng_np)
    populacao[slot] = (snn_f, kan_f, learner_f)
    return nova_g


def _inserir_organismo_mundo(
    mundo: Mundo,
    slot: int,
    geracao: int,
    rng_np: np.random.Generator,
) -> None:
    """Substitui mundo.organismos[slot] por um novo Organismo em posição vazia aleatória."""
    h, w = mundo.grid.shape
    for _ in range(500):
        ox = int(rng_np.integers(0, w))
        oy = int(rng_np.integers(0, h))
        if mundo.grid[oy, ox] == 0:
            break
    oang = float(rng_np.integers(0, 24) * 15)
    mundo.organismos[slot] = Organismo(ox, oy, oang, id=slot, geracao=geracao)


def _renovacao_geracional(
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]],
    metricas_ep: list[dict],
    geracao_max: int,
    device: torch.device,
    rng_np: np.random.Generator,
) -> int:
    """
    Renovação geracional no fim de cada episódio (pressão evolutiva).

    Selecciona os top-50% organismos por fitness (comida + tempo_vida/200).
    Substitui os bottom-50% por filhos de pares aleatórios do top-50%.
    Os sobreviventes mantêm os seus pesos; apenas a população fraca é renovada.

    Retorna o novo geracao_max.
    """
    from world.reproduction import get_genome, crossover, mutate, set_genome, herdar_epigenetica

    n = len(populacao)
    k = max(2, n // 2)   # top-50%

    # Fitness de cada organismo no episódio
    fitness = [
        m.get("comida", 0) + m.get("tempo_vida", 0) / 200.0
        for m in metricas_ep
    ]
    ranking = sorted(range(n), key=lambda i: fitness[i], reverse=True)
    top_k = ranking[:k]
    bottom_k = ranking[k:]

    nova_geracao_max = geracao_max
    for slot in bottom_k:
        # Escolher dois pais aleatórios do top-k (com reposição se necessário)
        # Fase B.2: evitar endogamia — tentar pares com dist. genética suficiente
        i_a = i_b = top_k[0]
        if len(top_k) >= 2:
            for _ in range(10):  # até 10 tentativas para encontrar par não-endogâmico
                i_a_c, i_b_c = rng_np.choice(top_k, size=2, replace=False)
                if cosine_sim_snn(populacao[int(i_a_c)][0], populacao[int(i_b_c)][0]) < INBREEDING_COS_THR:
                    i_a, i_b = i_a_c, i_b_c
                    break
            else:
                # fallback: aceitar o último par mesmo que similar
                i_a, i_b = rng_np.choice(top_k, size=2, replace=False)

        # Crossover + mutação
        snn_a, kan_a, learner_a = populacao[int(i_a)]
        snn_b, kan_b, learner_b = populacao[int(i_b)]
        g_a = get_genome(snn_a, kan_a)
        g_b = get_genome(snn_b, kan_b)
        g_f = crossover(g_a, g_b, rng_np)
        g_f = mutate(g_f, MUTACAO_SIGMA, rng_np)

        snn_f = OrganismoSNN(device=device).to(device)
        kan_f = KANPolicy(device=device)
        set_genome(snn_f, kan_f, g_f, device)

        learner_f = RSTDPLearner(snn_f, device=device)
        herdar_epigenetica(learner_f, learner_a, learner_b)

        populacao[slot] = (snn_f, kan_f, learner_f)
        nova_geracao_max = max(nova_geracao_max, geracao_max + 1)

    return nova_geracao_max


# ---------------------------------------------------------------------------
# Loop de episódio
# ---------------------------------------------------------------------------

def _correr_episodio(
    mundo: Mundo,
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]],
    ep: int,
    device: torch.device,
    rng_np: np.random.Generator,
    geracao_max: int,
) -> tuple[list[dict], int, int]:
    """
    Corre 1 episódio completo para toda a população, com reprodução contínua.

    Retorna (metricas_ind, n_nascimentos, geracao_max_final).
    """
    eps = _epsilon(ep)
    max_steps = _MAX_STEPS_POR_FASE.get(mundo.fase, 2000)

    obs_lista = [mundo._observacao(org) for org in mundo.organismos]
    for _, _, learner in populacao:
        learner.reset_traces()

    metricas = [{"tempo_vida": 0, "comida": 0, "recompensa_total": 0.0,
                 "steps_sem_comer": 0, "janela_comida": [], "geracao": mundo.organismos[i].geracao}
                for i in range(len(populacao))]

    n_nascimentos = 0
    vivos_ep = [True] * len(populacao)  # rastrear quem nasceu neste episódio

    for _step in range(max_steps):
        if mundo.terminado:
            break

        # Recolher acções
        accoes_nomes: list[str] = []
        xs_pre: list[torch.Tensor | None] = []

        for i, (snn, kan, _) in enumerate(populacao):
            org = mundo.organismos[i]
            if not org.vivo:
                accoes_nomes.append("parar")
                xs_pre.append(None)
                continue

            obs = obs_lista[i]
            x = encode_full_state(org, obs, T=T_STEPS, device=device)
            xs_pre.append(x)

            stats_kan = {
                "comida_vista_50steps": len(metricas[i]["janela_comida"]),
                "distancia_media_vista": 0.5,
                "steps_sem_comer": min(metricas[i]["steps_sem_comer"], _STEPS_SEM_COMER_MAX),
            }
            estado_kan = observacao_para_estado_kan(obs, stats_kan, device=device)
            kan.actualizar_snn(snn, estado_kan)

            if random.random() < eps:
                accao_idx = random.randint(0, 3)
            else:
                accao_idx = snn.accao(x)

            accoes_nomes.append(["avancar", "virar_esquerda", "virar_direita", "parar"][accao_idx])

        # Step do mundo
        resultados = mundo.step_populacao(accoes_nomes)

        # Aprendizagem e reprodução
        for i, (snn, kan, learner) in enumerate(populacao):
            res = resultados[i]
            if xs_pre[i] is None:
                continue

            r = res["recompensa_base"]
            metricas[i]["recompensa_total"] += r
            metricas[i]["tempo_vida"] += 1

            if res["comeu"]:
                metricas[i]["comida"] += 1
                metricas[i]["steps_sem_comer"] = 0
                metricas[i]["janela_comida"].append(1)
            else:
                metricas[i]["steps_sem_comer"] += 1

            metricas[i]["janela_comida"] = metricas[i]["janela_comida"][-_JANELA_COMIDA:]

            if not res["morreu"]:
                learner.step(xs_pre[i], r)
                obs_lista[i] = res["obs"]
            elif vivos_ep[i] and USAR_REPRODUCAO:
                # Organismo morreu: substituir por filho
                nova_g = _spawn_filho(mundo, populacao, i, rng_np, device, ep, geracao_max)
                geracao_max = max(geracao_max, nova_g)
                metricas[i]["geracao"] = nova_g
                n_nascimentos += 1
                vivos_ep[i] = False  # evitar spawn múltiplo no mesmo slot no mesmo step
                obs_lista[i] = mundo._observacao(mundo.organismos[i])
                # Reset métricas do slot (novo organismo começa do zero)
                metricas[i] = {"tempo_vida": 0, "comida": 0, "recompensa_total": 0.0,
                               "steps_sem_comer": 0, "janela_comida": [], "geracao": nova_g}
                vivos_ep[i] = True  # novo organismo vivo

    # Limpar campos temporários
    for m in metricas:
        m.pop("steps_sem_comer", None)
        m.pop("janela_comida", None)

    return metricas, n_nascimentos, geracao_max


# ---------------------------------------------------------------------------
# Agregação de métricas de população
# ---------------------------------------------------------------------------

def _agregar_metricas(
    metricas_ind: list[dict],
    organismos: list[Organismo],
) -> dict:
    vivos = [m for m in metricas_ind if m["tempo_vida"] > 0]
    if not vivos:
        return {"comida_media": 0.0, "vida_media": 0.0, "comida_total": 0,
                "comida_max": 0, "vida_max": 0, "geracao_media": 0.0}
    comidas = [m["comida"] for m in vivos]
    vidas   = [m["tempo_vida"] for m in vivos]
    geracoes = [o.geracao for o in organismos]
    return {
        "comida_media": float(np.median(comidas)),
        "vida_media":   float(np.median(vidas)),
        "comida_total": int(sum(comidas)),
        "comida_max":   int(max(comidas)),
        "vida_max":     int(max(vidas)),
        "geracao_media": float(np.mean(geracoes)),
        "geracao_max":  int(max(geracoes)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(DEVICE_STR)
    rng_np = np.random.default_rng()
    treino_id = datetime.now(timezone.utc).isoformat()

    print("=== Trainer Populacao v2.0 + Fase A (Reproducao) ===")
    print(f"treino_id    : {treino_id}")
    print(f"n_organismos : {N_ORGANISMOS}")
    print(f"n_episodios  : {N_EPISODIOS}")
    print(f"device       : {DEVICE_STR}")
    print(f"fase_inicial : {FASE_INICIAL}")
    print(f"base_snn     : {BASE_SNN_PATH}  exists={os.path.exists(BASE_SNN_PATH)}")
    print(f"base_kan     : {BASE_KAN_PATH}  exists={os.path.exists(BASE_KAN_PATH)}")
    print(f"usar_patches : {USAR_PATCHES}")
    print(f"usar_predador: {USAR_PREDADOR}")
    print(f"reproducao   : {USAR_REPRODUCAO}")
    print(f"mutacao_σ    : {MUTACAO_SIGMA}")
    if TREINO_NOTA:
        print(f"nota         : {TREINO_NOTA}")
    print()

    # Criar população inicial
    populacao: list[tuple[OrganismoSNN, KANPolicy, RSTDPLearner]] = [
        _criar_organismo(device) for _ in range(N_ORGANISMOS)
    ]
    curriculum = CurriculumManager(fase_inicial=FASE_INICIAL)
    geracao_max = 0

    # DB (opcional)
    conn = _db_connect()
    _div_cv_cache: float = 0.0   # último cv calculado (atualizado a cada _DIVERSIDADE_INTERVALO eps)

    for ep in range(N_EPISODIOS):
        mundo = Mundo(
            fase=curriculum.fase_actual,
            n_organismos=N_ORGANISMOS,
            usar_patches=USAR_PATCHES,
            usar_predador=USAR_PREDADOR,
        )

        t0 = time.time()
        metricas_ind, n_nasc, geracao_max = _correr_episodio(
            mundo, populacao, ep, device, rng_np, geracao_max
        )
        elapsed = time.time() - t0

        pop = _agregar_metricas(metricas_ind, mundo.organismos)

        avancou = curriculum.registar_episodio(
            tempo_vida=int(pop["vida_media"]),
            eficiencia=pop["comida_media"] / max(1, pop["vida_media"]) * 1000,
            precisao=0.0,
        )

        # Renovação geracional (Fase A): substituir bottom-50% por filhos do top-50%
        if USAR_REPRODUCAO:
            geracao_max = _renovacao_geracional(
                populacao, metricas_ind, geracao_max, device, rng_np
            )

        # Diversidade genética (amostrada a cada _DIVERSIDADE_INTERVALO episódios)
        div_str = ""
        if ep % _DIVERSIDADE_INTERVALO == 0:
            snns = [populacao[i][0] for i in range(N_ORGANISMOS)]
            div = calcular_diversidade(snns)
            _div_cv_cache = div["cv_fc1"]
            div_str = f"  cv={div['cv_fc1']:.3f}  dL2={div['dist_media']:.2f}"

        # Patches info
        patch_str = ""
        if USAR_PATCHES and mundo.patch_manager is not None:
            s = mundo.patch_manager.estado()
            patch_str = f"  food={s['total_food']}"

        eps_val = _epsilon(ep)
        print(
            f"ep={ep:4d}  fase={curriculum.fase_actual}"
            f"  comida_med={pop['comida_media']:5.1f}"
            f"  comida_max={pop['comida_max']:3d}"
            f"  vida_med={pop['vida_media']:5.0f}"
            f"  gen={pop['geracao_max']:3d}"
            f"  nasc={n_nasc:2d}"
            f"  eps={eps_val:.2f}"
            f"  {elapsed:.1f}s"
            f"{div_str}{patch_str}",
            flush=True,
        )

        if avancou:
            print(f">>> AVANCOU para fase {curriculum.fase_actual} <<<", flush=True)

        # DB: log por episódio + snapshot periódico de posições
        _log_episodio_pop(conn, treino_id, ep, curriculum.fase_actual,
                          pop, n_nasc, _div_cv_cache, geracao_max, elapsed)
        if ep % SNAPSHOT_INTERVAL == 0:
            _log_snapshot(conn, treino_id, ep, mundo)

        if (ep + 1) % SAVE_INTERVAL == 0:
            _guardar_checkpoints(populacao, ep + 1, treino_id)
            print(f"    [checkpoint ep{ep+1} guardado]", flush=True)

    _guardar_checkpoints(populacao, N_EPISODIOS, treino_id)
    print(f"\nFim. checkpoints_pop/{treino_id.replace(':', '_').replace('+', '')}/")
    if conn is not None:
        conn.close()
    main()
