# Histórico de Treinos — Milestones por Era

> Os parâmetros completos de cada treino estão na BD PostgreSQL:
> `SELECT id, nota, criado_em FROM treinos ORDER BY criado_em;`
> 
> Este documento regista as **descobertas chave** por era, não todos os treinos individuais.
> Total de treinos registados: 115 (v2 → v80p, Março 2026)

---

## Era 1 — Validação da Arquitectura (v2–v22, 12–16 Mar 2026)

**Objectivo**: confirmar que o pipeline SNN+R-STDP funciona de ponta a ponta.

| Marco | Versão | Descoberta |
|-------|--------|-----------|
| Neurónios mortos | v20 | `threshold_lif=0.15` matava 100% dos neurónios. Fix: `threshold_lif=0.01` — maior descoberta da era |
| RPE baseline | v5 | `rpe_baseline` (reward - rolling_mean) necessário para R-STDP aprender |
| LR warmup | v16 | warmup linear 0→1 em 100eps evita destruição dos pesos ANN pré-treino |
| Epsilon | v22 | decay=0.995, floor=0.15 (eps@ep200=0.15) — exploração sustentada |

**Estado no fim da era**: SNN aprende a associar visão→acção. Precisão ~0.35-0.40. Sem curriculum.

---

## Era 2 — Reward Shaping (v23–v58, 19–22 Mar 2026)

**Objectivo**: descobrir o sinal de reward correcto para o organismo aprender a navegar.

| Marco | Versão | Descoberta |
|-------|--------|-----------|
| WRONG-ADV essencial | v41 | penalty dinâmico quando `p_av_roll>0.35` → resolve avançar-em-loop |
| lambda_naca fixo | v42 | `lambda_naca=1e-6` sempre. Era 0.1 nos primeiros 100eps → destruía pesos ANN |
| AGMP desactivado | v43 | `N_OBSERVACAO=99999` → aprendizagem contínua sem recalibração constante |
| energia=200 | v51 | energia_inicial=100→200: starvation de 480→950 steps, mais tempo para aprender |
| Ceiling SNN | v57/v58 | Ceiling confirmado em prec≈0.474-0.478. Threshold_f1 baixado para 0.46 |
| biased_eps | v56 | biased_eps melhora precisão medida mas não a política real — não usar para avanço de fase |

**Aprendizagem chave**: reward shaping deve ser **condicional por fase** — o que ensina em fase 1 (virar) prejudica em fase 2+ (avançar).

**Estado**: precision 0.46+ atingida. Fase 1 → Fase 2 funcional. Fase 3: comida=0.

---

## Era 3 — Curriculum e Fase 3 (v59–v67, 22–24 Mar 2026)

**Objectivo**: atingir Fase 3 com comida >0.

| Marco | Versão | Descoberta |
|-------|--------|-----------|
| Shaping por fase | v63 | Fase1: turn_bonus+WRONG-ADV; Fase2+: food reward puro + starvation_bonus |
| Epsilon reset por fase | v62 | eps reinicia a 0.90 em cada nova fase (novelty nova → exploração nova) |
| Curriculum fix | v64/v65 | threshold_f2=vida_media≥1500 (remove precisao=0.80 impossível em fase2) |
| Fase 3 atingida | v65 | Primeira vez que o organismo chega a Fase 3. Comida=0 (olfato saturado) |

**Problema identificado em v65**: olfato uniformemente saturado em fases 1 e 2 → SNN aprende a ignorar olfato inteiramente → Fase 3 = random walk.

---

## Era 4 — Olfato: Experimentos Fracassados (v66–v79, 24–25 Mar 2026)

**Objectivo**: fazer o SNN usar o gradiente olfactivo em Fase 3.

| Marco | Versão | Ideia Tentada | Resultado |
|-------|--------|--------------|-----------|
| Olfato v3.0 sectorial | v66 | 3 inputs circular sectors R=100 | Saturado em fase1/2 → ignorado |
| Olfato shaping | v67/v68 | bonus quando acção → sector dominante | prec melhorou, mas SNN virava sem comer |
| Normalização relativa | v73/v74 | sector_dom/max → sempre 1.0 | fix de distribution shift, não de gradient |
| Pirouette biológico | v75/v76 | C. elegans pirouette modulation | complicou sem melhorar food/ep |
| advance_bonus fase3 | v70/v71 | +0.05 constante para bootstrap avançar | prec→0 (avançar sem visão=ruído) |
| olfato_bonus condicional | v77 | +0.05 só quando sector_dom>60% | boot lento, não escala |
| Sem p_av cap | v79 | fix bug #39 (p_av cap directo) | p_av não descontrolou, mas comida=0 |

**Root cause confirmada**: com raio olfactivo=100 e densidade=3.75% (fase 1), o organismo detecta ~47 food por sector → gradiente uniform (frente≈esq≈dir≈0.33) → SNN aprende a ignorar como ruído.

---

## Era 5 — Olfato 1/d² — Breakthrough (v80–v80p, 25–26 Mar 2026)

**Objectivo**: criar gradiente olfactivo real desde fase 1.

| Marco | Versão | Fix |
|-------|--------|-----|
| Arquitectura v4.0 | v80 | Olfato relativo-normalizado + raio por fase (30/60/100) + gate novelty |
| Ponderação 1/d² | v80p | `_w = 1/(dist² + 1)` → food próxima domina sinal. d=5→frente=0.83; d=10→frente=0.44 |
| Breakthrough | v80p | comida avg=20.7/ep em Fase 3 (0.15%, 1000×1000). **23× acima do baseline aleatório** |
| Threshold ajustado | v80m/n/o | threshold_prec=0.40, threshold_vida_media=800, min_INF_req=0 em fase2 |

**v80p = checkpoint base** (`snn_ep2000.pt`) para toda a simulação multi-organismo.

---

## Linha de Versões → Parâmetros Actuais

Os parâmetros consolidados do organismo validado (v53b + v64 + v80p):

```
threshold_lif   = 0.01          # v20
lambda_naca     = 1e-6          # v42 — sempre
N_OBSERVACAO    = 99999         # v43 — AGMP desactivado
energia_inicial = 200           # v51
eps_inicio      = 0.90          # v22
eps_decay       = 0.990/ep      # v22
eps_floor       = 0.15          # v22
turn_bonus      = +0.30 (f1)    # v55; 0 em f2+
WRONG-ADV       = 0.70 (f1)     # v41; 0 em f2+
threshold_f1    = 0.46          # v58
threshold_f2    = vida≥800      # v80o
olfato          = 1/d²          # v80p
raio_olfato     = 30/60/100     # v80 (por fase)
olfato_bonus    = 0.020×sector  # v80p (fase3 apenas)
```

---

## Como Consultar Qualquer Treino

```sql
-- Todos os treinos por ordem
SELECT id, nota, criado_em, estado FROM treinos ORDER BY criado_em;

-- Episódios de um treino específico
SELECT episodio, comida, tempo_vida, precisao_direcional, modo_plasticidade
FROM episodios_log WHERE treino_id = '<id>' ORDER BY episodio;

-- Sumário de um treino
SELECT MAX(episodio), AVG(comida), MAX(comida), AVG(precisao_direcional)
FROM episodios_log WHERE treino_id = '<id>';
```
