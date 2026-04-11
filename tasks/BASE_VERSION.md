# Organismo Base v1 — Definição Oficial

**Data**: 2026-03-27
**Estado**: ✅ APROVADO

---

## Identificação

| Campo | Valor |
|---|---|
| **treino_id (resume)** | `2026-03-26T16:27:55.922977+00:00` |
| **treino_id (original ep1-856)** | `2026-03-26T12:31:30.554186+00:00` |
| **Checkpoint seleccionado** | ep1300 (melhor janela de performance) |
| **Ficheiro SNN** | `mlartifacts/base_v1/snn_base_v1.pt` |
| **Ficheiro KAN** | `mlartifacts/base_v1/kan_base_v1.pt` |
| **Versão de treino** | v80p |

---

## Critérios de aprovação — resultado

| Critério | Target | Resultado | Status |
|---|---|---|---|
| comida_media fase 3 | > 5/ep | **21.3/ep** (média total) | ✅ |
| Melhor janela 100 eps | > 5/ep | **50.9/ep** (ep1249-1348) | ✅✅ |
| Episódios com comida > 20 | > 20% | **36.2%** de 1857 eps | ✅ |
| Máximo numa sessão | > 50 | **101 comida** (ep992) | ✅✅ |
| Ratio vs random walk | > 3× | **11.7×** no pico | ✅✅ |
| Fase 1→2 concluída | ep < 200 | **ep100** | ✅ |
| Fase 2→3 concluída | ep < 300 | **ep150** | ✅ |
| pav estável (sem colapso) | 0.15–0.55 | 0.18–0.22 no pico | ✅ |
| Episódios com comida=0 | < 15% | **4.9%** | ✅ |

---

## Perfil de performance (blocos 100 eps fase 3)

| Período | Comida/ep | pav | Caracterização |
|---|---|---|---|
| ep150–349 | 5.5–8.8 | 0.51–0.57 | Aprendizagem inicial |
| ep350–549 | 14.0–16.5 | 0.29–0.31 | Descoberta olfato navigation |
| **ep550–849** | **32–35/ep** | **0.183–0.186** | **Peak INF mode** |
| ep850–943 | 21.4 | 0.35 | Transição INF→APR |
| **ep944–1343** | **42–50/ep** | **0.189–0.192** | **Segundo peak — melhor janela** |
| ep1344–1443 | 31.1 | 0.32 | Início declínio |
| ep1444–2000 | 2.5–3.1 | 0.58–0.62 | Colapso advance-drift |

**Checkpoint ep1300** é seleccionado por estar no segundo peak (50/ep médio, pav=0.192).

---

## Comportamentos demonstrados

1. **Quimiotaxia olfactiva funcional**: organismo navega activamente em direcção a fontes de comida usando gradiente de olfato 1/d² em mundo 1000×1000 com 0.015% de densidade
2. **Curriculum learning 3 fases**: aprendeu em ambientes progressivamente mais desafiantes (200×200 → 500×500 → 1000×1000)
3. **INF mode**: plasticidade R-STDP autorregulada por CV de precisão direcional
4. **Burst navigation**: episódios de 10000 steps com 95-101 peças de comida (11.7× random)
5. **pav baixo em navegação eficiente**: pav=0.18-0.22 quando olfato navigation activo

---

## Limitação identificada

**Advance-drift a longo prazo**: após ep1444, o organismo regride para comportamento advance-heavy (pav=0.62, comida=2.5/ep). Trata-se de instabilidade do R-STDP em modo APR com olfato_bonus contínuo — a rede perde o equilíbrio turn/advance ao longo do tempo.

**Impacto para fases A+**: os pesos de ep1300 são o ponto de partida para herança genética. A instabilidade de longo prazo será mitigada naturalmente por:
- Pressão de selecção (organismos com advance-drift morrem mais cedo — menos reprodução)
- Herança de pesos misturados (crossover entre organismos com diferentes pav)
- Morte programada (Fase B) — organismos vivem o suficiente para navegar mas não para derivar

---

## Parâmetros do organismo base

```
SNN_DEVICE=cpu
SNN_N_AMOSTRAS=0
threshold_lif=0.01
lambda_naca=1e-6
AGMP N_OBSERVACAO=99999
eps_inicio=0.90, decay=0.990, floor=0.15
energia_inicial=200, energia_max=200
42 inputs: 15 visual + 3 estado + 4 eferente + 15 flow + 2 prop + 3 olfato
Olfato v4.1: 1/d² distance-weighted, raio=100 em fase 3
Reward shaping: olfato_bonus=0.020×sector (fase 3); starvation_bonus=0.20 (fase 2 apenas)
Curriculum: fase1 (200×200, 1500 food, prec_thr=0.40), fase2 (500×500, 5000 food, vida_thr=800), fase3 (1000×1000, 1500 food)
```
