"""
Dashboard de monitorização tribal — bio-organism SNN

Corre com:
    DATABASE_URL="postgresql://..." streamlit run dashboard/dashboard.py

Ou define DATABASE_URL no sidebar da UI.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tribal Monitor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_conn(url: str):
    """Conexão persistente psycopg2 (cache de recurso — não recria em cada refresh)."""
    import psycopg2
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


def _query(conn, sql: str, params=None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=30)
def _load_treinos(url: str) -> list[str]:
    conn = _get_conn(url)
    df = _query(conn, """
        SELECT treino_id, MIN(created_at) AS inicio
        FROM episodios_populacao_log
        GROUP BY treino_id
        ORDER BY inicio DESC
        LIMIT 50
    """)
    return df["treino_id"].tolist() if not df.empty else []


@st.cache_data(ttl=15)
def _load_metricas(url: str, treino_id: str) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT episodio, fase, comida_media, comida_max, vida_media, vida_max,
               geracao_max, n_nascimentos, cv_fc1, geracao_media, n_organismos,
               elapsed_s
        FROM episodios_populacao_log
        WHERE treino_id = %s
        ORDER BY episodio
    """, (treino_id,))


@st.cache_data(ttl=15)
def _load_snapshot_episodes(url: str, treino_id: str) -> list[int]:
    conn = _get_conn(url)
    df = _query(conn, """
        SELECT DISTINCT episodio FROM snapshots_log
        WHERE treino_id = %s
        ORDER BY episodio
    """, (treino_id,))
    return df["episodio"].tolist() if not df.empty else []


@st.cache_data(ttl=15)
def _load_snapshot(url: str, treino_id: str, episodio: int) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT organismo_idx, x, y, angulo, geracao, idade, comida_lifetime, vivo
        FROM snapshots_log
        WHERE treino_id = %s AND episodio = %s
        ORDER BY organismo_idx
    """, (treino_id, episodio))


def _dbscan_tribes(df: pd.DataFrame, eps: float = 80.0, min_samples: int = 2) -> pd.Series:
    """Clusteriza organismos por proximidade espacial. Returns Series com label por linha."""
    try:
        from sklearn.cluster import DBSCAN
        import numpy as np
        coords = df[["x", "y"]].values.astype(float)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        return pd.Series(labels, index=df.index, name="tribe")
    except ImportError:
        # fallback: todos na mesma tribo se sklearn não disponível
        return pd.Series(0, index=df.index, name="tribe")


# ---------------------------------------------------------------------------
# Multi-mundo query helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _load_simulacoes(url: str) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT simulacao_id, nota, n_mundos
        FROM simulacoes ORDER BY simulacao_id DESC LIMIT 30
    """)


@st.cache_data(ttl=60)
def _load_mundos_config(url: str, simulacao_id: str) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT mundo_id, world_size, n_food AS n_patches, seed,
               COALESCE(patches_json::text, '[]') AS patches_json
        FROM mundos_config WHERE simulacao_id = %s ORDER BY mundo_id
    """, (simulacao_id,))


@st.cache_data(ttl=60)
def _load_posicoes(url: str, simulacao_id: str, mundo_id: int) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT t_s, org_id, x, y, geracao, energia_norm, comida_acc
        FROM posicoes_snapshot
        WHERE simulacao_id = %s AND mundo_id = %s
        ORDER BY t_s, org_id
    """, (simulacao_id, mundo_id))


@st.cache_data(ttl=60)
def _load_mundos_log(url: str, simulacao_id: str, mundo_id: int) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT t_s, n_vivos, comida_rate_s, gen_max, cv_fc1, n_nascimentos
        FROM mundos_log
        WHERE simulacao_id = %s AND mundo_id = %s
        ORDER BY t_s
    """, (simulacao_id, mundo_id))


@st.cache_data(ttl=60)
def _load_lifetimes_multi(url: str, simulacao_id: str, mundo_id: int) -> pd.DataFrame:
    conn = _get_conn(url)
    return _query(conn, """
        SELECT org_id, x_nasc, y_nasc, x_morte, y_morte,
               COALESCE(is_cross_territory, FALSE) AS is_cross_territory,
               COALESCE(food_rate, 0.0) AS food_rate,
               causa_morte, geracao
        FROM organismos_lifetimes
        WHERE simulacao_id = %s AND mundo_id = %s
    """, (simulacao_id, mundo_id))


def _tribos_por_frame(df_pos: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Atribui cluster DBSCAN a cada organismo em cada snapshot t_s."""
    import numpy as np
    from sklearn.cluster import DBSCAN as _DBSCAN
    frames = []
    for t_s_val, grp in df_pos.groupby("t_s"):
        xy = grp[["x", "y"]].values.astype(float)
        if len(xy) >= 2:
            labels = _DBSCAN(eps=eps, min_samples=2).fit_predict(xy)
        else:
            labels = np.full(len(xy), -1)
        grp = grp.copy()
        grp["tribo"] = [f"T{l}" if l >= 0 else "Isolado" for l in labels]
        n_c = int(max(labels) + 1) if len(labels) > 0 and int(max(labels)) >= 0 else 0
        grp["n_tribos"] = n_c
        frames.append(grp)
    return pd.concat(frames, ignore_index=True) if frames else df_pos.copy()


def _build_animated_fig_multi(
    df: pd.DataFrame,
    world_size: int,
    frame_ms: int,
    patches: list,
) -> go.Figure:
    """Figura Plotly animada com organismos por tribo + círculos de patch estáticos."""
    df = df.copy()
    df["t_s_label"] = df["t_s"].apply(lambda v: f"{v:.0f}s")
    df["size"] = (df["energia_norm"].clip(0, 1) * 12).clip(lower=3)

    fig = px.scatter(
        df, x="x", y="y",
        animation_frame="t_s_label",
        animation_group="org_id",
        color="tribo",
        size="size", size_max=14,
        hover_data=["org_id", "geracao", "energia_norm", "comida_acc"],
        range_x=[0, world_size], range_y=[world_size, 0],
        template="plotly_dark",
        labels={"x": "Coluna", "y": "Linha", "tribo": "Tribo"},
    )
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = frame_ms
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
    fig.update_layout(height=550, margin=dict(t=30, b=10))

    # Círculos de patch (estáticos — não se movem durante a simulação)
    for p in patches:
        r = p.get("radius", 40)
        fig.add_shape(
            type="circle",
            x0=p["cx"] - r, y0=p["cy"] - r,
            x1=p["cx"] + r, y1=p["cy"] + r,
            line=dict(color="rgba(0,220,100,0.35)", width=1),
            fillcolor="rgba(0,220,100,0.06)",
        )
    return fig


@st.cache_resource(show_spinner=False)
def _load_llm(model_id: str):
    """Carrega modelo HuggingFace na GPU (ou CPU). Cacheado na sessão — só carrega uma vez."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        )
        mdl.eval()
        return tok, mdl
    except Exception as e:
        return None, str(e)


def _compute_narrative_stats(
    df_pos: pd.DataFrame,
    df_log: pd.DataFrame,
    df_lt: pd.DataFrame,
    patches: list,
    world_size: int,
) -> dict:
    """Extrai estatísticas agregadas da simulação para gerar narrativa."""
    stats: dict = {}

    # --- Tempo e população ---
    stats["total_time"] = float(df_pos["t_s"].max()) if not df_pos.empty else 0
    stats["n_frames"] = int(df_pos["t_s"].nunique()) if not df_pos.empty else 0
    stats["n_organisms_seen"] = int(df_pos["org_id"].nunique()) if not df_pos.empty else 0
    stats["peak_pop"] = int(df_log["n_vivos"].max()) if not df_log.empty else 0
    stats["final_pop"] = int(df_log["n_vivos"].iloc[-1]) if not df_log.empty else 0
    stats["world_size"] = world_size
    stats["n_patches"] = len(patches)
    # Estado actual dos patches (growing/peak/depleting/recovering)
    _phase_counts: dict = {}
    for p in patches:
        if isinstance(p, dict):
            st = p.get("state", "unknown")
            _phase_counts[st] = _phase_counts.get(st, 0) + 1
    stats["patch_states"] = _phase_counts

    # --- Tribos ---
    tribe_counts_per_frame = []
    switch_rates = []
    prev_frame_tribes = {}
    for t_s, grp in df_pos.groupby("t_s"):
        tribes = grp["tribo"].value_counts()
        non_iso = {k: v for k, v in tribes.items() if k != "Isolado"}
        tribe_counts_per_frame.append(len(non_iso))
        # switch rate vs previous frame
        cur_tribes = dict(zip(grp["org_id"], grp["tribo"]))
        if prev_frame_tribes:
            common_ids = set(cur_tribes) & set(prev_frame_tribes)
            if common_ids:
                switched = sum(1 for oid in common_ids if cur_tribes[oid] != prev_frame_tribes[oid])
                switch_rates.append(switched / len(common_ids))
        prev_frame_tribes = cur_tribes

    stats["max_tribes"] = max(tribe_counts_per_frame) if tribe_counts_per_frame else 0
    stats["avg_tribes"] = float(sum(tribe_counts_per_frame) / max(1, len(tribe_counts_per_frame)))
    stats["pct_frames_multi_tribe"] = (
        100 * sum(1 for c in tribe_counts_per_frame if c > 1) / max(1, len(tribe_counts_per_frame))
    )
    stats["avg_switch_rate"] = float(sum(switch_rates) / max(1, len(switch_rates))) if switch_rates else 0

    # --- Energia / forragem ---
    if not df_pos.empty:
        stats["avg_energy"] = float(df_pos["energia_norm"].mean())
        early = df_pos[df_pos["t_s"] <= df_pos["t_s"].quantile(0.25)]["energia_norm"].mean()
        late = df_pos[df_pos["t_s"] >= df_pos["t_s"].quantile(0.75)]["energia_norm"].mean()
        stats["energy_early"] = float(early) if not pd.isna(early) else 0
        stats["energy_late"] = float(late) if not pd.isna(late) else 0
        stats["energy_trend"] = "improving" if late > early + 0.05 else ("declining" if late < early - 0.05 else "stable")
    else:
        stats.update(avg_energy=0, energy_early=0, energy_late=0, energy_trend="stable")

    # --- Cross-territory ---
    if not df_lt.empty:
        stats["n_cross"] = int(df_lt["is_cross_territory"].sum())
        stats["pct_cross"] = 100 * stats["n_cross"] / max(1, len(df_lt))
    else:
        stats["n_cross"] = 0
        stats["pct_cross"] = 0

    # --- Evolução ---
    if not df_log.empty:
        stats["gen_max"] = int(df_log["gen_max"].max())
        stats["gen_rate"] = stats["gen_max"] / max(1, stats["total_time"]) * 100  # gens per 100s
        stats["cv_final"] = float(df_log["cv_fc1"].iloc[-1]) if df_log["cv_fc1"].iloc[-1] is not None else 0
        stats["cv_peak"] = float(df_log["cv_fc1"].max())
        cv_first = df_log["cv_fc1"].iloc[0] if df_log["cv_fc1"].iloc[0] is not None else 0
        stats["diversity_trend"] = (
            "rising" if stats["cv_final"] > float(cv_first) + 0.05
            else ("falling" if stats["cv_final"] < float(cv_first) - 0.05 else "stable")
        )
    else:
        stats.update(gen_max=0, gen_rate=0, cv_final=0, cv_peak=0, diversity_trend="stable")

    return stats


def _generate_narrative(stats: dict, llm_model: str = "") -> str:
    """Gera narrativa HTML. Se llm_model definido, usa modelo local HuggingFace; senão regras."""

    def _rule_based() -> str:
        s = stats
        sections = []

        # --- Visão geral ---
        pop_change = ""
        if s["peak_pop"] and s["final_pop"]:
            ratio = s["final_pop"] / s["peak_pop"]
            if ratio < 0.5:
                pop_change = " The population collapsed to under half its peak — resources became too scarce or predation too severe."
            elif ratio > 0.9:
                pop_change = " A stable, healthy population was maintained throughout."
        sections.append({
            "icon": "🌍", "title": "Overview",
            "text": (
                f"The simulation ran for <b>{s['total_time']:.0f}s</b> across a "
                f"<b>{s['world_size']}×{s['world_size']}</b> world with "
                f"<b>{s['n_patches']}</b> food patches. "
                f"<b>{s['n_organisms_seen']}</b> unique organisms were observed; "
                f"peak population reached <b>{s['peak_pop']}</b>{pop_change}"
            )
        })

        # --- Tribos ---
        tribe_stability = (
            "highly fluid" if s["avg_switch_rate"] > 0.20
            else ("moderately dynamic" if s["avg_switch_rate"] > 0.08 else "remarkably stable")
        )
        tribe_explain = (
            "Patches of food act as gravity wells — organisms that converge on the same patch "
            "end up close together and DBSCAN groups them as a tribe. "
            if s["n_patches"] > 0 else
            "Organisms cluster naturally as they follow similar gradient cues from surrounding food. "
        )
        switch_explain = (
            f"With a {100*s['avg_switch_rate']:.0f}% inter-frame reassignment rate, "
            "tribes shifted boundaries constantly as organisms crossed patch territories. "
            if s["avg_switch_rate"] > 0.08 else
            f"Only {100*s['avg_switch_rate']:.0f}% of organisms changed tribe per frame, "
            "suggesting tight, persistent social clusters. "
        )
        multi_info = (
            f"<b>{s['max_tribes']}</b> concurrent tribes appeared at peak, present in "
            f"{s['pct_frames_multi_tribe']:.0f}% of frames. "
        ) if s["max_tribes"] > 1 else "A single dominant group controlled most of the territory. "
        sections.append({
            "icon": "🏘️", "title": "Why Tribes Form",
            "text": tribe_explain + multi_info + switch_explain +
            "An organism 'changes tribe' when its position shifts closer to a different "
            "cluster — usually after depleting local food and migrating toward a richer patch."
        })

        # --- Forragem ---
        energy_adj = {"improving": "rose", "declining": "fell", "stable": "remained stable"}[s["energy_trend"]]
        energy_interp = (
            "rising energy means the SNN is learning to navigate toward food more efficiently."
            if s["energy_trend"] == "improving" else
            ("falling energy signals resource depletion outpacing foraging skill — "
             "the environment may be too sparse for the current population." if s["energy_trend"] == "declining"
             else "steady energy suggests a balance between foraging skill and resource availability.")
        )
        cross_info = (
            f"<b>{s['pct_cross']:.1f}%</b> of organisms were born outside their parent's territory "
            "(cross-territory births), indicating active food-seeking migration. "
            if s["n_cross"] > 0 else
            "No cross-territory births were detected — organisms tended to reproduce locally. "
        )
        phase_info = ""
        if s.get("patch_states"):
            _ph = s["patch_states"]
            _parts = []
            if _ph.get("growing"):   _parts.append(f"{_ph['growing']} a crescer 🟢")
            if _ph.get("peak"):      _parts.append(f"{_ph['peak']} em pico 🟡")
            if _ph.get("depleting"): _parts.append(f"{_ph['depleting']} a depletar 🟠")
            if _ph.get("recovering"):_parts.append(f"{_ph['recovering']} a recuperar 🔵")
            if _parts:
                phase_info = f"Patches ({', '.join(_parts)}). "
        sections.append({
            "icon": "🌿", "title": "Foraging Behaviour",
            "text": (
                f"{phase_info}"
                f"Average energy was <b>{s['avg_energy']:.2f}</b> (normalised 0–1). "
                f"Energy levels {energy_adj} from "
                f"<b>{s['energy_early']:.2f}</b> (early) to <b>{s['energy_late']:.2f}</b> (late) — "
                f"{energy_interp} "
                f"{cross_info}"
                "Organisms use an olfactory gradient (1/d² weighted sectors) to detect food at long range, "
                "then activate turn-toward-food reflexes learned via R-STDP spike timing."
            )
        })

        # --- Evolução ---
        gen_speed = (
            "extremely fast" if s["gen_rate"] > 1.0
            else ("fast" if s["gen_rate"] > 0.5 else ("moderate" if s["gen_rate"] > 0.2 else "slow"))
        )
        diversity_interp = (
            "High cv_fc1 (>0.3) means organisms carry very different weight distributions — "
            "multiple strategies coexist, which is healthy for long-term adaptability."
            if s["cv_final"] > 0.3 else
            ("Low cv_fc1 (<0.05) is a warning sign — the population is genetically homogeneous. "
             "A single environmental shock could wipe it out (inbreeding risk)."
             if s["cv_final"] < 0.05 else
             "Moderate diversity suggests the population is converging on a dominant strategy "
             "while retaining some variation.")
        )
        div_trend_str = {"rising": "↑ increasing", "falling": "↓ decreasing", "stable": "→ stable"}[s["diversity_trend"]]
        sections.append({
            "icon": "🧬", "title": "Evolutionary Dynamics",
            "text": (
                f"<b>{s['gen_max']}</b> generations were reached — "
                f"{gen_speed} evolution ({s['gen_rate']:.2f} gen/100s). "
                f"Genetic diversity (cv_fc1) reached a peak of <b>{s['cv_peak']:.3f}</b> "
                f"and ended at <b>{s['cv_final']:.3f}</b> ({div_trend_str}). "
                f"{diversity_interp}"
            )
        })

        # --- Render HTML ---
        parts = []
        for sec in sections:
            parts.append(
                f'<div class="nar-section">'
                f'<div class="nar-title">{sec["icon"]} {sec["title"]}</div>'
                f'<div class="nar-text">{sec["text"]}</div>'
                f'</div>'
            )
        return "\n".join(parts)

    # --- Modelo local HuggingFace ---
    if llm_model.strip():
        try:
            import torch
            tok, result = _load_llm(llm_model.strip())
            if tok is None:
                raise RuntimeError(result)  # result is the error string
            mdl = result
            s = stats
            prompt_text = (
                f"You analyse a multi-agent SNN simulation. "
                f"World: {s['world_size']}×{s['world_size']}, {s['n_patches']} food patches, "
                f"{s['total_time']:.0f}s duration. "
                f"Population: peak {s['peak_pop']}, final {s['final_pop']}. "
                f"Tribes: max {s['max_tribes']}, switch rate {100*s['avg_switch_rate']:.0f}%/frame, "
                f"multi-tribe in {s['pct_frames_multi_tribe']:.0f}% of frames. "
                f"Energy: avg {s['avg_energy']:.2f}, trend {s['energy_trend']} "
                f"(early {s['energy_early']:.2f} → late {s['energy_late']:.2f}). "
                f"Cross-territory births: {s['pct_cross']:.1f}%. "
                f"Evolution: gen {s['gen_max']} ({s['gen_rate']:.2f}/100s), "
                f"cv_fc1 peak {s['cv_peak']:.3f} final {s['cv_final']:.3f} ({s['diversity_trend']}).\n\n"
                f"Write exactly 4 paragraphs, each starting with one of these headings on its own line:\n"
                f"Overview\nWhy Tribes Form\nForaging Behaviour\nEvolutionary Dynamics\n"
                f"Each paragraph: 60-80 words, plain prose, cite the numbers. No markdown, no lists."
            )
            messages = [
                {"role": "system", "content": "You are a concise scientific narrator for multi-agent simulations."},
                {"role": "user", "content": prompt_text},
            ]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(tok, "apply_chat_template"):
                text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text_in = prompt_text
            inputs = tok(text_in, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl.generate(
                    **inputs, max_new_tokens=400,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tok.eos_token_id,
                )
            raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            # Parse into sections
            icons = {"Overview": "🌍", "Why Tribes Form": "🏘️",
                     "Foraging Behaviour": "🌿", "Evolutionary Dynamics": "🧬"}
            paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
            html_parts = []
            for para in paras[:4]:
                first_line, _, body = para.partition("\n")
                first_line = first_line.strip()
                icon = icons.get(first_line, "💡")
                html_parts.append(
                    f'<div class="nar-section">'
                    f'<div class="nar-title">{icon} {first_line}</div>'
                    f'<div class="nar-text">{body.strip() or first_line}</div>'
                    f'</div>'
                )
            if html_parts:
                return "\n".join(html_parts)
        except Exception:
            pass  # fall through to rule-based

    return _rule_based()


def _build_sim_json(
    df_pos: pd.DataFrame,
    df_log: pd.DataFrame,
    df_lt: pd.DataFrame,
    patches: list,
    world_size: int,
) -> str:
    """Serializa dados de uma simulação para JSON compacto (embedding no canvas HTML)."""
    import json
    frames = []
    for t_s_val, grp in df_pos.groupby("t_s"):
        orgs = [
            {"id": int(r.org_id), "x": int(r.x), "y": int(r.y),
             "tribo": str(r.tribo), "energy": float(r.energia_norm),
             "gen": int(r.geracao)}
            for r in grp.itertuples()
        ]
        frames.append({"t_s": float(t_s_val), "orgs": orgs})

    lifetimes = []
    if not df_lt.empty:
        for r in df_lt.itertuples():
            lifetimes.append({
                "x_nasc": int(r.x_nasc), "y_nasc": int(r.y_nasc),
                "is_ct": bool(r.is_cross_territory),
            })

    metrics_ts = []
    if not df_log.empty:
        for r in df_log.itertuples():
            metrics_ts.append({
                "t_s": float(r.t_s), "n_vivos": int(r.n_vivos),
                "gen_max": int(r.gen_max),
                "cv_fc1": float(r.cv_fc1) if r.cv_fc1 is not None else 0.0,
            })

    data = {
        "world_size": int(world_size),
        "patches": patches,
        "frames": frames,
        "lifetimes": lifetimes,
        "metrics_ts": metrics_ts,
    }
    return json.dumps(data, separators=(",", ":"))


def _build_canvas_html(sim_json: str, narrative_html: str = "", canvas_px: int = 560) -> str:
    """Constrói HTML completo: canvas 2D animado (esquerda) + painel narrativa (direita)."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;background:#111;color:#ddd;font-family:monospace;font-size:12px}}
#ctrl{{display:flex;gap:6px;align-items:center;padding:6px 8px;background:#1a1a1a;border-bottom:1px solid #333}}
button{{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:4px 8px;cursor:pointer;border-radius:3px;font-size:11px}}
button:hover{{background:#3a3a3a}}
button.active{{background:#4CAF50;color:#000;border-color:#4CAF50}}
#timeline{{flex:1;height:10px;background:#2a2a2a;border-radius:5px;cursor:pointer;position:relative;overflow:hidden}}
#tlprog{{height:100%;background:#4CAF50;border-radius:5px;width:0%;transition:width 0.05s linear}}
#tlabel{{min-width:52px;text-align:right;color:#aaa;font-size:10px}}
#legend{{display:flex;flex-wrap:wrap;gap:4px;padding:4px 8px;background:#151515;border-bottom:1px solid #222}}
.leg-item{{display:flex;align-items:center;gap:3px;font-size:10px}}
.leg-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
#body{{display:flex;height:{canvas_px}px}}
canvas{{display:block;background:#0d0d0d;flex-shrink:0}}
/* Narrative panel */
#panel{{
  flex:1;overflow-y:auto;background:#0f0f0f;border-left:1px solid #2a2a2a;
  display:flex;flex-direction:column;gap:0;
}}
#live-stats{{
  background:#141414;border-bottom:1px solid #222;padding:10px 12px;
  flex-shrink:0;
}}
#live-stats h3{{font-size:11px;color:#4CAF50;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px}}
.stat-row{{display:flex;justify-content:space-between;margin:3px 0;font-size:11px}}
.stat-label{{color:#888}}
.stat-value{{color:#eee;font-weight:bold}}
.stat-bar{{flex:1;margin:0 8px;height:6px;background:#222;border-radius:3px;align-self:center}}
.stat-bar-fill{{height:100%;border-radius:3px;background:#4CAF50;transition:width 0.15s}}
#mood-indicator{{
  margin-top:8px;padding:5px 8px;border-radius:4px;font-size:11px;
  background:#1a2a1a;border:1px solid #2a3a2a;color:#8BC34A;
  transition:all 0.3s;
}}
#narrative{{padding:12px;flex:1}}
.nar-section{{margin-bottom:16px}}
.nar-title{{font-size:12px;font-weight:bold;color:#80CBC4;margin-bottom:5px;padding-bottom:3px;border-bottom:1px solid #1a2a2a}}
.nar-text{{font-size:11px;color:#bbb;line-height:1.55;font-family:sans-serif}}
.nar-text b{{color:#ddd}}
</style>
</head>
<body>
<script type="application/json" id="sim-data">{sim_json}</script>

<div id="ctrl">
  <button id="btnReset" title="Recomeçar">⏮</button>
  <button id="btnPlay" class="active">⏸ Pause</button>
  <button class="spd" data-v="0.5">0.5×</button>
  <button class="spd active" data-v="1">1×</button>
  <button class="spd" data-v="2">2×</button>
  <button class="spd" data-v="5">5×</button>
  <button class="spd" data-v="20">20×</button>
  <div id="timeline"><div id="tlprog"></div></div>
  <span id="tlabel">0s / 0s</span>
</div>
<div id="legend"></div>

<div id="body">
  <canvas id="canvas" width="{canvas_px}" height="{canvas_px}"></canvas>

  <div id="panel">
    <div id="live-stats">
      <h3>⚡ Live State</h3>
      <div class="stat-row">
        <span class="stat-label">Time</span>
        <span class="stat-value"><span id="ls-time">0</span>s</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Organisms</span>
        <span class="stat-value" id="ls-orgs">–</span>
        <div class="stat-bar"><div class="stat-bar-fill" id="ls-orgs-bar" style="width:0%"></div></div>
      </div>
      <div class="stat-row">
        <span class="stat-label">Tribes</span>
        <span class="stat-value" id="ls-tribes">–</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Generation</span>
        <span class="stat-value" id="ls-gen">–</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Diversity cv_fc1</span>
        <span class="stat-value" id="ls-cv">–</span>
        <div class="stat-bar"><div class="stat-bar-fill" id="ls-cv-bar" style="width:0%;background:#9C27B0"></div></div>
      </div>
      <div class="stat-row">
        <span class="stat-label">Avg energy</span>
        <span class="stat-value" id="ls-energy">–</span>
        <div class="stat-bar"><div class="stat-bar-fill" id="ls-energy-bar" style="width:0%;background:#FF9800"></div></div>
      </div>
      <div id="mood-indicator">…</div>
    </div>
    <div id="narrative">{narrative_html}</div>
  </div>
</div>

<script>
const D = JSON.parse(document.getElementById('sim-data').textContent);
const PALETTE = {{
  "T0":"#4CAF50","T1":"#2196F3","T2":"#FF5722","T3":"#9C27B0",
  "T4":"#FF9800","T5":"#00BCD4","T6":"#E91E63","T7":"#CDDC39",
  "Isolado":"#444"
}};

const cv = document.getElementById('canvas');
const ctx = cv.getContext('2d');
const SZ = {canvas_px}, WS = D.world_size, SC = SZ / WS;

D.frames.sort((a,b) => a.t_s - b.t_s);
const t_max = D.frames.length > 0 ? D.frames[D.frames.length-1].t_s : 1;
const peak_pop = D.metrics_ts.length ? Math.max(...D.metrics_ts.map(m => m.n_vivos)) : 1;

let playing = true, speed = 1, t_sim = 0, t_wall_last = null;

// Build legend
const seen_tribos = new Set();
D.frames.forEach(f => f.orgs.forEach(o => seen_tribos.add(o.tribo)));
const leg = document.getElementById('legend');
seen_tribos.forEach(t => {{
  if (t === 'Isolado') return;
  const d = document.createElement('div');
  d.className = 'leg-item';
  d.innerHTML = `<div class="leg-dot" style="background:${{PALETTE[t]||'#888'}}"></div><span>${{t}}</span>`;
  leg.appendChild(d);
}});
if (seen_tribos.has('Isolado')) {{
  const d = document.createElement('div'); d.className = 'leg-item';
  d.innerHTML = `<div class="leg-dot" style="background:#444"></div><span>Isolado</span>`;
  leg.appendChild(d);
}}
if (D.patches.length > 0) {{
  const PHASE_LABELS = {{growing:'A crescer',peak:'Pico',depleting:'A depletar',recovering:'A recuperar'}};
  const PHASE_COLORS = {{growing:'#00c864',peak:'#ffd700',depleting:'#ff6820',recovering:'#40a0ff'}};
  Object.entries(PHASE_LABELS).forEach(([st, lbl]) => {{
    const d = document.createElement('div'); d.className = 'leg-item';
    d.innerHTML = `<div class="leg-dot" style="background:${{PHASE_COLORS[st]}};opacity:0.6"></div><span>Patch ${{lbl}}</span>`;
    leg.appendChild(d);
  }});
}}
if (D.lifetimes.some(l => l.is_ct)) {{
  const d = document.createElement('div'); d.className = 'leg-item';
  d.innerHTML = `<div class="leg-dot" style="background:gold;border-radius:0;transform:rotate(45deg)"></div><span>Nasc. cross-terr.</span>`;
  leg.appendChild(d);
}}

function lerp(a, b, t) {{ return a + (b - a) * t; }}

function getFrame(t) {{
  if (D.frames.length === 0) return [];
  let lo = 0, hi = D.frames.length - 1;
  for (let i = 0; i < D.frames.length - 1; i++) {{
    if (D.frames[i].t_s <= t) lo = i; else {{ hi = i; break; }}
  }}
  const flo = D.frames[lo], fhi = D.frames[hi];
  if (lo === hi || fhi.t_s === flo.t_s) return flo.orgs;
  const alpha = (t - flo.t_s) / (fhi.t_s - flo.t_s);
  const map_lo = {{}};
  flo.orgs.forEach(o => map_lo[o.id] = o);
  return fhi.orgs.map(o_hi => {{
    const o_lo = map_lo[o_hi.id];
    if (!o_lo) return o_hi;
    return {{...o_hi, x: lerp(o_lo.x, o_hi.x, alpha), y: lerp(o_lo.y, o_hi.y, alpha)}};
  }});
}}

function getMet(t) {{
  if (!D.metrics_ts.length) return {{n_vivos:0,gen_max:0,cv_fc1:0}};
  let m = D.metrics_ts[0];
  for (const r of D.metrics_ts) {{ if (r.t_s <= t) m = r; else break; }}
  return m;
}}

// Mood messages based on current state
function getMood(orgs, met, n_tribes) {{
  const avg_e = orgs.length ? orgs.reduce((s,o) => s+o.energy,0)/orgs.length : 0;
  if (orgs.length === 0) return {{text:"⚰️ Population extinct.", color:"#FF5722", bg:"#2a1010"}};
  if (avg_e > 0.7 && n_tribes > 1)
    return {{text:`✨ ${{n_tribes}} thriving tribes compete for patches — energy high.`, color:"#8BC34A", bg:"#1a2a1a"}};
  if (avg_e > 0.7)
    return {{text:"🌱 Abundant food, single dominant group foraging efficiently.", color:"#4CAF50", bg:"#1a2a1a"}};
  if (avg_e < 0.25 && orgs.length < peak_pop * 0.3)
    return {{text:"⚠️ Population crash — starvation risk high.", color:"#FF9800", bg:"#2a1a10"}};
  if (n_tribes > 2)
    return {{text:`🏘️ ${{n_tribes}} territorial groups — spatial partitioning in progress.`, color:"#00BCD4", bg:"#101a2a"}};
  if (met.gen_max > 5)
    return {{text:`🧬 Gen ${{met.gen_max}} reached — inherited SNN weights shaping behaviour.`, color:"#9C27B0", bg:"#1a102a"}};
  return {{text:"🔍 Organisms exploring — learning food gradient cues.", color:"#aaa", bg:"#1a1a1a"}};
}}

let last_mood_update = -999;

function draw(ts) {{
  if (playing) {{
    if (t_wall_last !== null) {{
      t_sim = Math.min(t_max, t_sim + (ts - t_wall_last) / 1000 * speed);
    }}
    t_wall_last = ts;
    if (t_sim >= t_max) {{
      playing = false;
      document.getElementById('btnPlay').textContent = '▶ Play';
      document.getElementById('btnPlay').classList.remove('active');
    }}
  }} else {{ t_wall_last = null; }}

  const orgs = getFrame(t_sim);
  const met = getMet(t_sim);
  const n_tribos = new Set(orgs.map(o => o.tribo).filter(t => t !== 'Isolado')).size;
  const avg_energy = orgs.length ? orgs.reduce((s,o) => s+o.energy,0)/orgs.length : 0;

  // ---- Canvas draw ----
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, SZ, SZ);

  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(0.75, 0.75, SZ - 1.5, SZ - 1.5);

  const PATCH_FILL   = {{growing:'rgba(0,200,100,0.10)',peak:'rgba(255,215,0,0.14)',depleting:'rgba(255,104,32,0.10)',recovering:'rgba(64,160,255,0.09)'}};
  const PATCH_STROKE = {{growing:'rgba(0,200,100,0.45)',peak:'rgba(255,200,0,0.60)',depleting:'rgba(255,104,32,0.50)',recovering:'rgba(64,160,255,0.40)'}};
  D.patches.forEach(p => {{
    const st = p.state || 'peak';
    ctx.beginPath();
    ctx.arc(p.cx * SC, p.cy * SC, p.radius * SC, 0, Math.PI * 2);
    ctx.fillStyle = PATCH_FILL[st] || PATCH_FILL.peak;
    ctx.fill();
    ctx.strokeStyle = PATCH_STROKE[st] || PATCH_STROKE.peak;
    ctx.lineWidth = 1;
    ctx.stroke();
  }});

  ctx.fillStyle = 'rgba(255,215,0,0.35)';
  D.lifetimes.forEach(lt => {{
    if (!lt.is_ct) return;
    const x = lt.x_nasc * SC, y = lt.y_nasc * SC, s = 4;
    ctx.beginPath();
    ctx.moveTo(x, y - s); ctx.lineTo(x + s, y);
    ctx.lineTo(x, y + s); ctx.lineTo(x - s, y);
    ctx.closePath(); ctx.fill();
  }});

  orgs.forEach(o => {{
    const x = o.x * SC, y = o.y * SC;
    const r = Math.max(2, 3 + o.energy * 4);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = PALETTE[o.tribo] || '#888';
    ctx.fill();
    if (r > 4) {{
      ctx.strokeStyle = 'rgba(0,0,0,0.5)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }}
  }});

  // Timeline
  const pct = t_max > 0 ? t_sim / t_max : 0;
  document.getElementById('tlprog').style.width = (pct * 100) + '%';
  document.getElementById('tlabel').textContent =
    Math.round(t_sim) + 's / ' + Math.round(t_max) + 's';

  // ---- Side panel live stats (update every ~200ms to avoid jank) ----
  if (ts - last_mood_update > 200) {{
    last_mood_update = ts;
    document.getElementById('ls-time').textContent = Math.round(t_sim);
    document.getElementById('ls-orgs').textContent = orgs.length;
    const orgPct = peak_pop > 0 ? Math.min(100, orgs.length / peak_pop * 100) : 0;
    document.getElementById('ls-orgs-bar').style.width = orgPct + '%';
    document.getElementById('ls-tribes').textContent = n_tribos;
    document.getElementById('ls-gen').textContent = met.gen_max || 0;
    const cv = met.cv_fc1 || 0;
    document.getElementById('ls-cv').textContent = cv.toFixed(3);
    document.getElementById('ls-cv-bar').style.width = Math.min(100, cv * 200) + '%';
    document.getElementById('ls-energy').textContent = avg_energy.toFixed(2);
    document.getElementById('ls-energy-bar').style.width = (avg_energy * 100) + '%';
    const mood = getMood(orgs, met, n_tribos);
    const moodEl = document.getElementById('mood-indicator');
    moodEl.textContent = mood.text;
    moodEl.style.color = mood.color;
    moodEl.style.background = mood.bg;
    moodEl.style.borderColor = mood.color + '55';
  }}

  requestAnimationFrame(draw);
}}

// Controls
document.getElementById('btnPlay').addEventListener('click', () => {{
  playing = !playing;
  const btn = document.getElementById('btnPlay');
  btn.textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) btn.classList.add('active'); else btn.classList.remove('active');
}});
document.getElementById('btnReset').addEventListener('click', () => {{
  t_sim = 0; playing = true;
  const btn = document.getElementById('btnPlay');
  btn.textContent = '⏸ Pause'; btn.classList.add('active');
}});
document.querySelectorAll('.spd').forEach(btn => {{
  btn.addEventListener('click', () => {{
    speed = parseFloat(btn.dataset.v);
    document.querySelectorAll('.spd').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }});
}});
document.getElementById('timeline').addEventListener('click', e => {{
  const rect = e.currentTarget.getBoundingClientRect();
  t_sim = Math.max(0, Math.min(t_max, ((e.clientX - rect.left) / rect.width) * t_max));
}});

requestAnimationFrame(draw);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🧬 Tribal Monitor")

    db_url = st.text_input(
        "DATABASE_URL",
        value=os.getenv("DATABASE_URL", "postgresql://snn_user:snn_secure_pw_2026@localhost:5432/snn_organismo"),
        type="password",
    )

    if not db_url:
        st.warning("Define DATABASE_URL para continuar.")
        st.stop()

    try:
        conn_test = _get_conn(db_url)
    except Exception as e:
        st.error(f"Conexão falhou: {e}")
        st.stop()

    treinos = _load_treinos(db_url)
    if not treinos:
        st.warning("Sem dados em episodios_populacao_log. Inicia uma simulação primeiro.")
        st.stop()

    treino_sel = st.selectbox("Treino", treinos)

    st.divider()

    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    dbscan_eps = st.slider("DBSCAN eps (raio tribal, células)", 30, 200, 80, step=10)
    color_by = st.radio("Colorir por", ["Tribo (DBSCAN)", "Geração"], horizontal=True)
    llm_model = st.text_input(
        "Modelo LLM (narrativa)",
        value="Qwen/Qwen2.5-0.5B-Instruct",
        help="ID HuggingFace carregado na GPU na primeira utilização e cacheado. "
             "Deixa vazio para usar narrativa baseada em regras.",
    )

    st.divider()
    st.caption("Dados actualizados de 15 em 15s com cache.")

# ---------------------------------------------------------------------------
# Header com resumo
# ---------------------------------------------------------------------------

df_met = _load_metricas(db_url, treino_sel)

if df_met.empty:
    st.info("A aguardar dados do treino...")
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    st.stop()

latest = df_met.iloc[-1]
n_eps = len(df_met)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Episódios", n_eps)
col2.metric("Fase actual", int(latest.get("fase", 1)))
col3.metric("Comida/ep (last)", f"{latest['comida_media']:.1f}")
col4.metric("Geração máx", int(latest.get("geracao_max") or 0))
col5.metric("cv_fc1 (diversidade)", f"{(latest.get('cv_fc1') or 0):.3f}")

st.markdown(f"**Treino**: `{treino_sel}`")
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

df_sims = _load_simulacoes(db_url)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Métricas de População", "🗺️ Mapa Tribal", "🧬 Evolução Genética", "🌍 Multi-Mundo"])

# ---- Tab 1: Métricas -------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(df_met, x="episodio", y=["comida_media", "comida_max"],
                      title="Comida por episódio",
                      labels={"value": "comida", "variable": ""},
                      color_discrete_map={"comida_media": "#4CAF50", "comida_max": "#8BC34A"})
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(df_met, x="episodio", y=["vida_media", "vida_max"],
                      title="Vida por episódio (steps)",
                      labels={"value": "steps", "variable": ""},
                      color_discrete_map={"vida_media": "#2196F3", "vida_max": "#03A9F4"})
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.line(df_met, x="episodio", y="geracao_max",
                      title="Geração máxima (velocidade evolutiva)",
                      labels={"geracao_max": "geração"},
                      color_discrete_sequence=["#FF9800"])
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        nasc = df_met.dropna(subset=["n_nascimentos"])
        if not nasc.empty:
            fig = px.bar(nasc, x="episodio", y="n_nascimentos",
                         title="Nascimentos por episódio",
                         color_discrete_sequence=["#9C27B0"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dados de nascimentos indisponíveis.")

    # Fase transitions
    fases = df_met.groupby("fase")["episodio"].agg(["min", "max"]).reset_index()
    if len(fases) > 1:
        st.markdown("**Transições de fase:**")
        cols = st.columns(len(fases))
        for i, row in fases.iterrows():
            cols[i].metric(f"Fase {int(row['fase'])}", f"ep {int(row['min'])}–{int(row['max'])}")

# ---- Tab 2: Mapa Tribal ----------------------------------------------------
with tab2:
    snap_eps = _load_snapshot_episodes(db_url, treino_sel)

    if not snap_eps:
        st.info("Sem snapshots de posições. Activa DATABASE_URL na simulação e aguarda o primeiro snapshot.")
    else:
        # Slider de episódio
        ep_idx = st.select_slider(
            "Episódio do snapshot",
            options=snap_eps,
            value=snap_eps[-1],
            format_func=lambda x: f"ep {x}",
        )

        df_snap = _load_snapshot(db_url, treino_sel, ep_idx)

        if df_snap.empty:
            st.warning("Snapshot vazio.")
        else:
            # Clustering tribal
            df_snap["tribe"] = _dbscan_tribes(df_snap, eps=float(dbscan_eps))
            df_snap["tribe_label"] = df_snap["tribe"].apply(
                lambda x: f"Tribo {x}" if x >= 0 else "Isolado"
            )

            n_tribes = df_snap[df_snap["tribe"] >= 0]["tribe"].nunique()
            n_isolados = (df_snap["tribe"] == -1).sum()

            c1, c2, c3 = st.columns(3)
            c1.metric("Organismos no snapshot", len(df_snap))
            c2.metric("Tribos detectadas", n_tribes)
            c3.metric("Organismos isolados", n_isolados)

            # Scatter tribal
            color_col = "tribe_label" if color_by == "Tribo (DBSCAN)" else "geracao"
            size_col = df_snap["comida_lifetime"].clip(lower=1)

            # Determinar bounds do mundo a partir da fase
            fase_ep = df_met[df_met["episodio"] <= ep_idx]["fase"].iloc[-1] if not df_met[df_met["episodio"] <= ep_idx].empty else 1
            world_size = {1: 200, 2: 500, 3: 1000}.get(int(fase_ep), 1000)

            fig = px.scatter(
                df_snap,
                x="x", y="y",
                color=color_col,
                size=size_col,
                size_max=18,
                hover_data={"geracao": True, "idade": True, "comida_lifetime": True,
                            "angulo": ":.0f", "vivo": True, "organismo_idx": True},
                title=f"Mapa Tribal — Episódio {ep_idx}  |  Mundo {world_size}×{world_size}",
                range_x=[0, world_size], range_y=[0, world_size],
                height=600,
            )
            fig.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(autorange="reversed"),   # y=0 topo (convenção grid)
                plot_bgcolor="#1a1a2e",
                paper_bgcolor="#16213e",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabela resumo por tribo
            if n_tribes > 0:
                tribo_stats = (
                    df_snap[df_snap["tribe"] >= 0]
                    .groupby("tribe_label")
                    .agg(
                        n_org=("organismo_idx", "count"),
                        geracao_media=("geracao", "mean"),
                        comida_media=("comida_lifetime", "mean"),
                        x_centro=("x", "mean"),
                        y_centro=("y", "mean"),
                    )
                    .round(1)
                    .reset_index()
                )
                st.markdown("**Resumo das tribos:**")
                st.dataframe(tribo_stats, use_container_width=True)

# ---- Tab 3: Evolução Genética -----------------------------------------------
with tab3:
    df_div = df_met.dropna(subset=["cv_fc1"])

    if df_div.empty:
        st.info("Dados de diversidade genética indisponíveis (activa DATABASE_URL).")
    else:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.line(df_div, x="episodio", y="cv_fc1",
                          title="Diversidade genética (cv_fc1) — divergência tribal",
                          labels={"cv_fc1": "coef. variação fc1"},
                          color_discrete_sequence=["#E91E63"])
            fig.add_hline(y=0.05, line_dash="dash", line_color="grey",
                          annotation_text="threshold diversidade mínima")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "geracao_media" in df_met.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_met["episodio"], y=df_met["geracao_max"],
                                         name="gen_max", line=dict(color="#FF5722")))
                if df_met["geracao_media"].notna().any():
                    fig.add_trace(go.Scatter(x=df_met["episodio"], y=df_met["geracao_media"],
                                             name="gen_media", line=dict(color="#FF9800", dash="dot")))
                fig.update_layout(title="Progressão geracional",
                                  xaxis_title="episódio", yaxis_title="geração",
                                  legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Como ler a diversidade tribal:**
    - **cv_fc1 crescente** → pesos SNN a divergir entre organismos → tribos especializam-se
    - **Pico seguido de queda** → selecção natural convergiu para estratégia dominante → colapso tribal
    - **cv_fc1 > 0.5** → diversidade alta, tribos com estratégias distintas
    - **cv_fc1 < 0.05** → população geneticamente homogénea (inbreeding ou gargalo evolutivo)
    """)

# ---- Tab 4: Multi-Mundo ----------------------------------------------------
with tab4:
    import json as _json

    if df_sims.empty:
        st.info("Sem simulações multi-mundo na BD. Corre `launch_multi_world.sh` primeiro.")
    else:
        col_sim, col_mundo = st.columns(2)
        with col_sim:
            sim_sel = st.selectbox("Simulação", options=df_sims["simulacao_id"].tolist(),
                                   format_func=lambda s: s)
        df_mcfg = _load_mundos_config(db_url, sim_sel)
        with col_mundo:
            mundo_opts = df_mcfg["mundo_id"].tolist()
            mundo_id = st.selectbox("Mundo",
                                    options=mundo_opts,
                                    format_func=lambda m: f"Mundo {m}")

        if mundo_id is None:
            st.info("Selecciona um mundo.")
        else:
            df_pos = _load_posicoes(db_url, sim_sel, mundo_id)
            df_log = _load_mundos_log(db_url, sim_sel, mundo_id)
            df_lt  = _load_lifetimes_multi(db_url, sim_sel, mundo_id)
            row_cfg = df_mcfg[df_mcfg.mundo_id == mundo_id].iloc[0]
            world_sz  = int(row_cfg["world_size"])
            n_patches = int(row_cfg["n_patches"])

            # Ler posições dos patches
            raw_pj = row_cfg["patches_json"]
            try:
                patches = _json.loads(raw_pj) if isinstance(raw_pj, str) else (raw_pj or [])
            except Exception:
                patches = []

            st.markdown(f"**Mundo {mundo_id}** — {n_patches} patches  |  {world_sz}×{world_sz}")

            if df_pos.empty:
                st.warning("Sem snapshots de posições para este mundo.")
            else:
                # DBSCAN tribal clustering por fotograma
                df_pos = _tribos_por_frame(df_pos, eps=float(dbscan_eps))

                # Métricas de topo
                m1, m2, m3, m4 = st.columns(4)
                last_log = df_log.iloc[-1] if not df_log.empty else {}
                m1.metric("Organismos (fim)", int(last_log.get("n_vivos", 0)))
                m2.metric("Geração máx", int(df_log["gen_max"].max()) if not df_log.empty else 0)
                n_cross = int(df_lt["is_cross_territory"].sum()) if not df_lt.empty else 0
                m3.metric("Cross-territory", f"{100*n_cross/max(1,len(df_lt)):.1f}%")
                last_frame = df_pos[df_pos.t_s == df_pos.t_s.max()]
                m4.metric("Tribos (último frame)", int(last_frame["n_tribos"].max()) if not last_frame.empty else 0)

                # Canvas 2D animado + painel narrativa
                nar_stats = _compute_narrative_stats(df_pos, df_log, df_lt, patches, world_sz)
                with st.spinner("A gerar narrativa (1.ª vez: descarrega modelo para GPU, pode demorar)…"):
                    narrative_html = _generate_narrative(nar_stats, llm_model=llm_model)
                sim_json = _build_sim_json(df_pos, df_log, df_lt, patches, world_sz)
                html_canvas = _build_canvas_html(sim_json, narrative_html, canvas_px=560)
                st.components.v1.html(html_canvas, height=640, scrolling=False)

                # Gráficos de série temporal
                if not df_log.empty:
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.plotly_chart(
                        px.line(df_log, x="t_s", y="n_vivos", title="Organismos vivos",
                                template="plotly_dark", labels={"t_s": "Tempo (s)"}),
                        use_container_width=True,
                    )
                    tc2.plotly_chart(
                        px.line(df_log, x="t_s", y="gen_max", title="Geração máx",
                                template="plotly_dark", labels={"t_s": "Tempo (s)"}),
                        use_container_width=True,
                    )
                    tc3.plotly_chart(
                        px.line(df_log, x="t_s", y="cv_fc1", title="Diversidade cv_fc1",
                                template="plotly_dark", labels={"t_s": "Tempo (s)"}),
                        use_container_width=True,
                    )

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    placeholder = st.empty()
    for remaining in range(30, 0, -1):
        placeholder.caption(f"⏱ Auto-refresh em {remaining}s…")
        time.sleep(1)
    placeholder.empty()
    st.cache_data.clear()
    st.rerun()
