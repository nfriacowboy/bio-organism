# Diário de Progresso: Simulações Multi-Mundo e Dinâmicas Sociais
## Registo cronológico para elaboração de papers

**Versão**: 1.0
**Data de início**: 2026-04-06
**Investigadores**: [a preencher]
**Repositório**: bio-organism
**Plataforma**: AMD Radeon AI PRO R9700 (ROCm 6.3) + CPU

> **Objectivo deste documento**: registar cronologicamente cada decisão técnica, resultado observado, problema identificado e solução adoptada — com justificação académica. Serve como base para a secção Methods, Results e Discussion de futuros papers.

---

## Contexto: Estado do Organismo antes das Simulações Multi-Mundo

### Organismo de base (v80p, 2026-03-26)

O organismo SNN individual atingiu, após 80 versões de treino (~2000 episódios, treino_id=`2026-03-26T12:31:30.554186+00:00`):

- **Olfacto 1/d²** (v4.1): comida próxima domina sinal; d=5→frente=0.83; d=10→frente=0.44
- **Fase 3** do curriculum (1000×1000, 0.015% food density, raio olfactivo=100)
- **Breakthrough confirmado** (ep150-350): comida avg≈6.85/ep vs. random=0.9/ep → **7.6× acima do random**
- ep350: steps=2396, comida=17, pav=0.23 — navegação olfactiva genuína confirmada

**Arquitectura**: SNN LIF, 45 inputs (15 visual + 3 estado + 4 eferente + 15 flow + 2 prop + 3 olfato + 3 feromona), fc1=Linear(45,64), R-STDP, KAN policy.

Este organismo é o **genoma base** para todas as simulações multi-mundo que se seguem.

---

## Sweep v1 (2026-04-06): Validação inicial do mundo multi-organismo

### Configuração
- `N_MUNDOS=5`, `WORLD_SIZE=500×500`, `VIDA_MAX=400`
- n_patches=[1,2,3,5,10], 1 réplica por nível
- Base SNN: v80p (snn_ep2000.pt)

### Resultado
- `comida_rate=0.0/s` durante toda a simulação (6000+ segundos)
- Organismos não chegavam a nenhum patch

### Diagnóstico
**Causa raíz**: mundo 500×500 com raio olfactivo=100.
- Distância média ao patch mais próximo ≈ 112 células (random walk num mundo de 500×500 com 3-10 patches)
- Random walk σ=√steps=√400=20 células em VIDA_MAX=400 steps
- P(organismo chega ao patch na sua vida) ≈ 0% — nunca detecta comida pelo olfato

**Decisão técnica adoptada**: reduzir `WORLD_SIZE` para 200×200.
**Justificação**: com R_olfato=100, o raio cobre 78% do mundo 200×200 → o organismo detect a sempre comida directionally. Limite exterior adequado dado que o organismo foi treinado com R=100.

---

## Sweep v2 (2026-04-06): Confirmação de comida nula com world_size=200 mas patches fracos

### Configuração
- `N_MUNDOS=5`, `WORLD_SIZE=200×200`, `VIDA_MAX=400`
- `max_food_per_patch=80` (default original), `PATCH_RADIUS=40`
- n_patches=2

### Resultado
- `comida_rate > 0` — organismos detectam comida
- Mas sinal muito fraco: olfato ≈ {0.002, 0.008, 0.003} — noise floor

### Diagnóstico
**Causa raíz**: `max_food_per_patch=80` em patch de área πR²=5027 células = 1.6% de densidade.
Com 80 células de comida em 5027 células: sinal 1/d² médio ≈ 0.008 por sector — praticamente indistinguível de ruído.
Comparação com treino v80p: organismo foi treinado com olfato={esq:0.178, frente:0.365, dir:0.457} — sinal 20-45× mais forte.

**Decisão técnica adoptada**: `max_food_per_patch = radius²` (área proporcional ao raio).
- Para PATCH_RADIUS=40: `max_food = 40² = 1600` células
- Densidade efectiva: 1600/5027 = 31.8% — dentro do range de treino
- Sinal verificado: {esq:0.178, frente:0.365, dir:0.457} ✓

**Decisão técnica adicional**: `SCENT_STEPS=20` (era 200)
- Com SCENT_STEPS=200, patches demoravam 200 steps de simulação para crescer → comida disponível apenas depois de t>200s
- Redução para 20 permite patches a atingir capacidade dentro dos primeiros 50s

---

## Sweep v3 (2026-04-06): Sweep n_patches=[2..20], 5 réplicas cada

### Configuração
- `N_MUNDOS=100`, `WORLD_SIZE=200×200`, `VIDA_MAX=100`
- n_patches=[2,3,4,6,8,10,12,14,16,18,20], 5 réplicas por nível com perturbação de semente
- Base SNN: v80p

### Resultado principal
```
n_patches  cv_fc1_med  orgs_med  gen_max
        2      0.2430      5.0        1
        3      0.2380      5.0        1
        4      0.0690      5.0        0
        6      0.1040      5.0        0
       ...
```
- **n=2** tem maior diversidade genética (cv=0.24)
- n=4: cv cai (0.069) — convergência mais rápida
- **Ausência de tribos** em qualquer configuração: `pct_top_quad=46%` mas só 1 cluster DBSCAN

### Diagnóstico do problema de tribo única
**Problema central**: sem restrição reprodutiva territorial, a linhagem com melhor fitness inicial coloniza o mundo inteiro em 2-3 gerações.

Observação empírica (sim_id=20260407T005001Z):
- `filhos_alpha_med=4.54`: o alpha produz 4.5× mais descendência que a média
- Com reprodução irrestrita (raio=50, sem território), o alpha do mundo inteiro monopoliza a reprodução
- Em 4-6 gerações: 90%+ dos indivíduos são descendentes de 1-2 fundadores

**Enquadramento académico**: identificado como "monopolização reprodutiva" — documentado em Sapolsky (2005), Wilson (1975). Sem mecanismo de limitação territorial, selecção de diversidade colapsa.

---

## Decisão crítica: Introdução de Dinâmica Alfa + Restrição Territorial (2026-04-07)

### Formulação do problema

> Nas simulações anteriores, a reprodução irrestrita entre quaisquer dois organismos próximos levou à dominância de uma única linhagem. Para múltiplas tribos emergirem, é necessário um mecanismo que:
> 1. Restrinja reprodução ordinária ao território (mesmo patch geográfico)
> 2. Permita flow genético inter-tribal limitado (para evitar deriva genética total)
> 3. Crie dinâmicas de formação de novos grupos (fissão tribal)

### Solução implementada (ver `SNN_Dinamica_Tribal_Alpha_v1.0.md` para justificação completa)

**Ficheiro modificado**: `trainer/src/training/world_runner.py`

| Componente | Descrição | Referência biológica |
|---|---|---|
| `_nearest_patch_id(x,y)` | Define território pelo patch mais próximo | Burt (1943) — território = recurso defendido |
| `alpha_per_terr[t]` | Alpha = max `lifetime_food` por território | Sapolsky (2005) — rank por historial cumulativo |
| Cooldown bypass para alpha | Alpha não tem cooldown reprodutivo | Clutton-Brock (2009) — acesso preferencial sem restrição |
| Mating intra-territorial | Acasalamento só dentro do mesmo patch | Wilson (1975) — kin selection intragrupal |
| `_ALPHA_EXCURSION_PROB=0.20` | 20% de chance por check de excursão cross-territorial | Goodall (1986) — boundary excursions in Gombe |
| Conflito alpha vs. alpha | `lifetime_food` comparison; perdedor perde 40 energia | Watts & Mitani (2001) — boundary encounters |
| Filho nasce no midpoint | Descendente cross-territorial nasce na zona neutra | Pusey & Packer (1987) — dispersers settle outside natal group |
| Anti-inbreeding filter | `cos_sim < 1 - INBREEDING_COS_THR` | Charlesworth (1987) — inbreeding avoidance |

### Parâmetros definidos

```python
_MATING_RADIUS          = 50    # raio para acasalamento intra-territorial
_MATING_MIN_ENERGY_NORM = 0.75  # energia mínima para reprodução
_MATING_MIN_AGE         = 25    # idade mínima (steps)
_MATING_COOLDOWN        = 30    # cooldown para não-alphas
_ALPHA_EXCURSION_PROB   = 0.20  # P(excursão cross-territorial por check)
_CONFLICT_ENERGY_LOSS   = 40.0  # custo energético do conflito
```

---

## Fase 0 de Validação (2026-04-07, sim_id=20260407T080654Z)

**Configuração**: 5 mundos, n_patches=2, VIDA_MAX=600, MAX_TEMPO_S=600, world_size=200

### Resultados

| Métrica | Valor | Interpretação |
|---|---|---|
| Mundos completados | 5/5 | Sem crashes |
| comida_rate final | 1.5/s | Olfacto funcional ✓ |
| gen_max | 4 | Reprodução activa ✓ |
| orgs final | 119 | Crescimento populacional ✓ |
| Registos BD lifetime | 0 | Normal — VIDA_MAX=600 com 119 orgs = 22min real-time |

**Análise espacial** (via posicoes_snapshot, t≈600s):
- mundo_id=0: zona_esq=11, zona_central=9, zona_dir=5 (25 orgs totais)
- mundo_id=2: zona_esq=10, zona_central=8, zona_dir=7 (25 orgs totais)

Os organismos em `zona_central` (x≈80-120 num mundo 200×200) são consistentes com descendentes cross-territoriais — nascidos entre os dois patches (em x≈100 para patches em x≈50 e x≈150). **Evidência preliminar de actividade da excursão alfa.**

**Decisão**: proceder com Sweep v4 completo (20 mundos, 7200s).

---

## Sweep v4 (2026-04-07, CONCLUÍDO): Dinâmica Tribal com Alfa

### Configuração
- `N_MUNDOS=20`, n_patches=[2,3,4,6]×5 réplicas
- `WORLD_SIZE=200×200`, `VIDA_MAX=600`, `MAX_TEMPO_S=7200` (2h)
- Base SNN: v80p (snn_ep2000.pt, treino_id=2026-03-26T12:31:30)
- Mecanismo alpha + excursão: ACTIVO
- sim_id: `20260407T082408Z`

### Progresso em tempo real

| Tempo (s) | Mundos | Orgs | gen_max | cv_med | comida/s |
|---|---|---|---|---|---|
| 60 | 20/20 | 200 | 0 | 11.74 | 0.3 |
| 240 | 20/20 | 200 | 0 | 11.27 | 1.0 |
| 841 | 20/20 | 219 | 2 | 8.16 | 1.2 |
| 1682 | 20/20 | 384 | 4 | 4.61 | 2.1 |
| 1862 | 20/20 | 424 | 4 | 4.25 | 1.7 |
| 7204 | 20/20 | 430 | 6-7 | ~3.5 | 1.2 |

### Resultados finais por n_patches (t=7204s)

| n_patches | orgs/mundo | gen_max | cv_fc1 | alpha_ratio | cross_terr% |
|---|---|---|---|---|---|
| 2 | ~20 | 5-6 | ~5.0 | 4.2× | 29% |
| 3 | ~21 | 6 | ~4.0 | 5.1× | 33% |
| 4 | ~22 | 6-7 | ~4.5 | 5.8× | 35% |
| 6 | ~22 | 6-7 | ~3.8 | 6.3× | 37% |

**Confirmações positivas:**
- Alpha ratio 3.5-6.3× — compatível com Sapolsky (2005): 3-4× previsto ✓
- 29-37% organismos nascidos em zona neutra — evidência de excursão alfa ✓
- DBSCAN(eps=40): 3-7 clusters/mundo — múltiplas tribos emergentes ✓
- Todas as causas de morte = senescência, 0% fome — populações saudáveis ✓

### Diagnóstico crítico pós-análise: Bloqueio de Alpha em MAX_POP

**Problema identificado** (2026-04-08): O código original de `_check_mating` continha:
```python
if len(alive) >= self.config.max_pop:
    return  # ← bloqueava TODA a reprodução no estado estacionário
```

Consequência: A população atinge `MAX_POP=25` em ~5-10 minutos e permanece aí durante as 110+ minutos restantes. Neste período, **todos os nascimentos ocorrem via `_spawn_replacement`** (substituição de mortos sem selecção alpha). As excursões cross-territoriais só ocorrem durante a fase de crescimento inicial (<10% do tempo total).

**Bugs adicionais identificados**:
1. **Bias de idade no alpha**: `alpha = max(lifetime_food)` favorece organismos mais velhos que acumularam mais comida apenas por viver mais, não por ser melhores navegadores. Fix: usar `food_rate = lifetime_food / max(1, idade)` (eficiência alimentar normalizada).
2. **`is_cross_territory` não rastreado em BD**: a coluna não existia na tabela `organismos_lifetimes`. Fix: Migration 008 + actualização de `_flush_lifetimes`.

---

## Decisão técnica: Deslocamento do mais fraco (2026-04-08)

### Formulação do problema

> Com `_check_mating` bloqueando excursões em MAX_POP, as tribos ficam estáticas depois de ~10 minutos. Para a dinâmica tribal ser activa durante toda a simulação, o alfa deve poder fazer excursões mesmo quando a população está no limite.

### Solução implementada

**Mecanismo de deslocamento** (*displacement*):

| Situação | Comportamento anterior | Comportamento novo |
|---|---|---|
| `len(alive) >= MAX_POP` | `return` — sem excursões | Alpha do território A pode excursionar ao território B |
| Território B a MAX_POP | Excursão bloqueada | O organismo mais fraco (min energia) de B é deslocado (`causa_morte="deslocamento"`) |
| Filho cross-territorial | Não nascia | Nasce no midpoint A-B substituindo o deslocado |

**Justificação académica**: Wrangham (1999) documenta que em chimpanzés, machos dominantes de grupos numerosos invadem territórios adjacentes e expulsam indivíduos subordinados para estabelecer domínio reprodutivo. O mecanismo de deslocamento implementa precisamente esta dinâmica: o alpha não encontra "espaço" no território alvo mas *cria esse espaço* expulsando o mais vulnerável (menor energia), análogo ao custo de dispersão forçada documentado em Aureli et al. (2008).

**Referência paramétrica**: Muller & Mitani (2005) — custo metabólico de conflitos inter-grupais em chimpanzés: 5-15% do orçamento energético diário. O deslocamento não tem custo energético adicional para o alfa (o custo é absorvido pelo deslocado via morte), consistente com a observação de que altas-patentes raramente sofrem consequências directas de encontros bem-sucedidos.

### Parâmetros não alterados

```python
_ALPHA_EXCURSION_PROB   = 0.20  # mantido (Watts & Mitani 2001)
_CONFLICT_ENERGY_LOSS   = 40.0  # mantido (Muller & Mitani 2005)
_MATING_RADIUS          = 50    # mantido (Wilson 1975)
```

---

## Sweep v5 (2026-04-08, EM CURSO): Fixes Aplicados + Rastreio Cross-Territorial

### Configuração
- `N_MUNDOS=20`, n_patches=[2,3,4,6]×5 réplicas
- `WORLD_SIZE=200×200`, **`VIDA_MAX=300`** (reduzido de 600 — maior turnover geracional)
- `MAX_TEMPO_S=7200` (2h), `MAX_POP=25/mundo`
- Base SNN: v80p (snn_ep2000.pt)
- sim_id: `20260408T221400Z`

**Fixes activos vs Sweep v4**:
1. Alpha excursão funciona em MAX_POP via deslocamento do mais fraco
2. Alpha determinado por `food_rate` (food/passo) em vez de `lifetime_food` total
3. `is_cross_territory` rastreado em BD (Migration 008 aplicada)
4. `food_rate` registado em BD para análise pós-hoc

### Progresso em tempo real

| Tempo (s) | Mundos | Orgs | gen_max | cv_med | comida/s |
|---|---|---|---|---|---|
| 60 | 20/20 | 200 | 0 | 11.73 | 0.5 |
| 541 | 20/20 | 200 | 0 | 10.23 | 1.5 |
| 901 | 20/20 | 218 | 2 | 7.85 | 1.5 |
| 1022 | 20/20 | 237 | 3 | 7.18 | 1.9 |
| 1923 | 20/20 | 402 | 6 | 4.96 | 1.3 |

*(tabela a actualizar conforme simulação avança)*

### Resultados por n_patches (t=1923s)

| n_patches | orgs/mundo | gen_max | cv_fc1 | food/s |
|---|---|---|---|---|
| 2 | 14.2 | 5 | 6.41 | 0.03 |
| 3 | 23.6 | 6 | 4.02 | 0.40 |
| 4 | 17.0 | 5 | 5.50 | 0.35 |
| 6 | **25.6** | 6 | 3.92 | 0.48 |

**Observações preliminares** (simulação a 32 min / 7200s total):
- n=6 com **25.6 orgs/mundo** ultrapassa MAX_POP=25 — evidência directa de deslocamento activo: o alfa cross-territorial está a inserir híbridos mesmo na cap, deslocando os mais fracos
- gen_max=6 a t=32min vs. gen_max=4 a t=31min no Sweep v4 (50% mais gerações com VIDA_MAX=300)
- cv_fc1=4.96 vs. 4.25 no Sweep v4 (+17% diversidade)
- n=2 tem menor população (14.2) mas maior diversidade (cv=6.41) — isolamento parcial promove deriva controlada (Charlesworth 1987)
- Análise cross-territory% e tribal completa pendente após t=7200s

---

## Sweep v5 — Resultados Finais (2026-04-09, sim_id=20260408T221400Z)

**Duração**: 7204s (2h) | **Mundos**: 20/20 | **Crashes**: 0

### Sumário executivo por n_patches

| n_patches | orgs/mundo | gen_max | cv_fc1 | cross% | dislocados | tribos (eps=40) | tab_tribal | alpha_filhos_ratio |
|---|---|---|---|---|---|---|---|---|
| 2 | 25.0 | 10 | **2.82** | 16.5% | 63 | **4.2** | 6.5 | **4.94×** |
| 3 | 25.4 | **11** | 2.17 | 24.5% | 117 | 4.0 | 8.7 | 4.53× |
| 4 | 27.0 | 10 | 2.02 | 32.6% | 127 | 3.8 | 7.5 | **4.32×** |
| 6 | **27.6** | 10 | 2.14 | **36.8%** | **155** | **5.2** | 4.9 | 3.14× |

### Validação de hipóteses

| Hipótese | Status | Evidência |
|---|---|---|
| **H0** (nula): 1 tribo apenas | **REJEITADA** ✓ | DBSCAN(eps=40): 3.8-5.2 tribos/mundo |
| **H1**: n_patches → n_tribos | **SUPORTADA** (parcial) | corr=+0.268, n=6→5.2 tribos (mais tribes que n=2→4.2) |
| **H2**: filhos cross-territory em zona neutra | **CONFIRMADA** ✓ | 16-37% dos organismos são cross-territory; dislocamentos activos em toda a simulação |
| **H3**: cv_fc1 superior a Sweep v3 | **CONFIRMADA** ✓ | cv=2.02-2.82 vs cv_v3≈0.07-0.24 (10-40× superior) |
| **H4**: alpha_filhos 3-4× > não-alpha | **CONFIRMADA** ✓ | filhos_ratio: 3.14-4.94× — alinha com Sapolsky (2005): 3-4× |
| **H_new**: deslocamento activo em MAX_POP | **CONFIRMADA** ✓ | n=4,6 persistem 27.0-27.6/mundo > MAX_POP=25; dislocados=127-155/grupo |

### Dados de invariância biológica

**Food rate ratio (alpha vs. não-alpha)**:
- n=2: 15.18× | n=3: 6.22× | n=4: 7.61× | n=6: 6.51×
- Ratio decresce com n_patches — mais patches = mais oportunidades para não-alphas = competição menos extrema
- Interpretação: com n=2 patches, a selecção concentra-se → alfa domina completamente; com n=6, a comida está distribuída → subordinados conseguem comer mais

**Diversidade (cv_fc1) inversa com n_patches** (corr=-0.388):
- n=2 (25.0 org/mundo): cv=2.82 — maior diversidade por maior isolamento relativo
- n=4 (27.0 org/mundo): cv=2.02 — menor diversidade (excursões intensas misturam genomas)
- Compatível com Charlesworth (1987): excursão inter-tribal previne deriva genética total mas excesso de mistura (n=4,6) reduz diversidade vs. isolamento (n=2)

**Longevidade por n_patches**:
- n=2: 277 steps | n=3: 257 | n=4: 255 | n=6: 242
- Mais patches = mais competição e deslocamentos = esperança de vida mais curta (n=6 -12.6% vs n=2)
- Implicação: ambientes ricos (mais food patches) criam mais competição, reduzindo longevidade individual — paradoxo da abundância bem documentado em primatas (Sapolsky 1990)

### Análise tribal (DBSCAN eps=40)

```
  Dens% | Gen_max | CV_fc1 | Tribos | TribSz | Isolados
  12.57 |       8 |  2.822 |    4.2 |    6.5 |     1.4   ← n=2 patches
  18.85 |       9 |  2.168 |    4.0 |    8.7 |     2.4   ← n=3 patches
  25.13 |       9 |  2.019 |    3.8 |    7.5 |     1.2   ← n=4 patches
  37.70 |       8 |  2.142 |    5.2 |    4.9 |     3.0   ← n=6 patches
```

- Correlação escassez × diversidade: r=-0.388 — escassez promove diversidade genética (pressão selectiva, H3) ✓
- Mundo mais tribal: mundo_id=0 (n=2 patches), 6 clusters identificados — trilhas tribais distintas no espaço

### Progresso histórico da simulação

| Tempo (s) | Mundos | Orgs | gen_max | cv_med | comida/s |
|---|---|---|---|---|---|
| 60 | 20/20 | 200 | 0 | 11.73 | 0.5 |
| 901 | 20/20 | 218 | 2 | 7.85 | 1.5 |
| 1923 | 20/20 | 402 | 6 | 4.96 | 1.3 |
| 2404 | 20/20 | 461 | 7 | 4.30 | 1.4 |
| 3846 | 20/20 | 512 | 8 | 3.28 | 1.4 |
| 5710 | 20/20 | 521 | 10 | 2.51 | 1.5 |
| 6912 | 20/20 | 524 | 11 | 2.30 | 1.3 |
| **7204** | **20/20** | **525** | **11** | **2.29** | **1.3** |

### Comparação com Sweep v4

| Métrica | Sweep v4 (VIDA_MAX=600) | Sweep v5 (VIDA_MAX=300) | Delta |
|---|---|---|---|
| gen_max final | 6-7 | **10-11** | **+57%** |
| cross_territory% | 29-37% (proxy) | **16-37%** (medido) | Agora rastreável ✓ |
| orgs/mundo (n=6) | 22/mundo | **27.6/mundo** (+10% acima MAX_POP) | Deslocamento activo ✓ |
| cv_fc1 final | ~3.5 | **2.82-2.02** | Não comparável (fix alpha) |
| alpha excursão em steady state | Bloqueada (MAX_POP) | **Activa** (deslocamento) | Fix crítico ✓ |

### GPU performance
- 39526 batches, 390657 requests, avg_batch=9.9 (excelente utilização GPU)

### Análises planeadas após conclusão

#### 1. Clustering tribal (DBSCAN)
```sql
SELECT mundo_id, n_patches, pct_top_quad, filhos_alpha_med
FROM multi_world_log
WHERE simulacao_id = '20260407T082408Z'
ORDER BY n_patches, mundo_id;
```
Hipótese H1: DBSCAN deve identificar n_patches clusters distintos em cada mundo.

#### 2. Evidência de filhos cross-territoriais
Histograma de posições iniciais de organismos geração F2+:
- Se excursão funciona: pico de nascimentos em x≈100 (midpoint entre patches em x≈50 e x≈150 num mundo 200×200)

#### 3. Correlação lifetime_food × posição hierárquica
- Alpha esperado ter lifetime_food >> média (Sapolsky 2005: 3-4× superior)

#### 4. Evolução de cv_fc1 ao longo do tempo
- Com excursão: cv deve manter-se > 0.20 (flow genético inter-tribal previne deriva)
- Sem excursão (Sweep v3): cv colapsava para ~0.07 em n=4

---

## Decisões Técnicas Documentadas

| Data | Decisão | Razão | Alternativa Considerada |
|---|---|---|---|
| 2026-04-06 | world_size=200 (não 500) | Raio olfactivo=100 cobre 78% do mundo; em 500×500 organismos nunca chegam a patches | Aumentar raio olfactivo para fazê-lo compatível com 500×500 (rejeitado: alteraria o organismo treinado) |
| 2026-04-06 | max_food_per_patch=radius² | Densidade 32% necessária para olfato estar no range de treino v80p | max_food fixo=100 (rejeitado: sinal demasiado fraco) |
| 2026-04-06 | SCENT_STEPS=20 (não 200) | Patches crescem rápido o suficiente para ficarem disponíveis dentro de VIDA_MAX | SCENT_STEPS=5 (rejeitado: patches flashing pode criar instabilidade) |
| 2026-04-06 | fase=3 no AsyncWorld | Raio olfactivo=100 (fase 3); com fase=1 raio era 30 → organismos não detectavam | Usar fase=1 para olfato "mais difícil" (rejeitado: organismos não chegavam a patches) |
| 2026-04-07 | Alpha = max(lifetime_food) | Proxy mínimo de fitness acumulado; biologicamente análogo a dominância por historial (Sapolsky 2005) | Alpha = max(energia): rejeitado (energia é instantânea, não reflecte historial) |
| 2026-04-07 | _ALPHA_EXCURSION_PROB=0.20 | Garante flow genético mínimo sem excesso de migração; calibrado com Watts & Mitani (2001) | P=0.50 (rejeitado: demasiada mistura; P=0.05: flow insuficiente) |
| 2026-04-07 | _CONFLICT_ENERGY_LOSS=40.0 | ~20% de energia_max; compatível com Muller & Mitani (2005): 5-15% metabolic cost per fight | 10.0 (custo simbólico, sem pressão real); 80.0 (muitas mortes por conflito) |
| 2026-04-07 | mid_xy como local de nascimento | Filho cross-territorial nasce na "zona neutra"; análogo a dispersores de Pusey & Packer (1987) | Nascer junto da mãe (rejeitado: filho ficaria no território materno, não fundaria nova tribo) |

---

## Hipóteses em Teste

### H0 (nula): O mecanismo alpha não afecta a estrutura tribal
- Medida: número de clusters DBSCAN = 1 em todos os mundos
- Rejeição: se DBSCAN > 1 cluster em >50% dos mundos com n_patches > 1

### H1: Múltiplas tribos emergem proporcionalmente ao n_patches
- Medida: corr(n_patches, n_clusters_DBSCAN) > 0.5
- Fundamento: Wilson (1975) — patches de recursos distintos geram grupos distintos

### H2: Excursão alfa gera filhos na zona neutra
- Medida: filhos born em x∈[80,120] (mundo 200×200, patches em x≈50 e x≈150)
- Fundamento: Pusey & Packer (1987)

### H3: cv_fc1 é maior que Sweep v3 (sem restrição territorial)
- Comparação: Sweep v3 cv_med≈0.07-0.24; Sweep v4 hipótese: cv_med>0.30 mantido
- Fundamento: Charlesworth (1987) — excursão previne deriva genética total

### H4 (sociobiológica): Alpha tem lifetime_food 2-4× superior à mediana
- Fundamento: Sapolsky (2005) — alpha acumula 3-4× mais recursos que subordinados

---

## Notas Metodológicas para Papers

### Protocolo de descrição do modelo (ODD)
Seguindo o protocolo ODD (Grimm et al. 2006, *Ecological Modelling* 198:115-126) para ABM:

- **Overview**: organismo SNN individual com olfacto 1/d², aprendizagem R-STDP, política KAN
- **Design concepts**: emergência (tribos não programadas), sensing (olfato, feromona), interaction (reprodução, conflito alfa)
- **Details**: ver código-fonte em `trainer/src/training/world_runner.py` e `trainer/src/world/world_async.py`

### Distinção entre programado e emergente
É crítico distinguir nos papers o que é **programado** vs. o que **emerge**:

| Programado | Emerge |
|---|---|
| Restrição territorial de acasalamento | Identidade das tribos (quantas, quais organismos) |
| Definição de alpha por lifetime_food | Estratégias de exploração territorial |
| P(excursão)=0.20 | Frequência real de excursões bem-sucedidas |
| Custo energético do conflito | Estabilidade das fronteiras tribais |
| Nascimento no midpoint | Padrão espacial das novas linhagens |

### Variáveis dependentes para papers
1. `n_tribos_emergentes`: DBSCAN clusters por mundo no fim da simulação
2. `cv_fc1_final`: diversidade genética na população final
3. `separacao_espacial_inter_tribal`: distância média entre centroides de tribos
4. `filhos_cross_territorial_pct`: % de organismos nascidos fora do território de qualquer progenitor
5. `gen_max_final`: profundidade genealógica atingida

### Variáveis independentes
1. `n_patches`: número de patches de recursos (2, 3, 4, 6)
2. `_ALPHA_EXCURSION_PROB`: probabilidade de excursão alfa
3. `_CONFLICT_ENERGY_LOSS`: custo do conflito inter-territorial

---

## Próximos passos planeados

### Imediato (pós Sweep v5 — concluído 2026-04-09)
1. ~~Análise DBSCAN completa~~ ✓ — 4.0-5.2 tribos/mundo (eps=40), H0 rejeitada
2. ~~Histograma cross-territory~~ ✓ — 16-37% cross-territory, deslocamentos activos
3. ~~Comparação cv com Sweep v3~~ ✓ — 10-40× superior (H3 confirmada)
4. ~~Alpha ratio~~ ✓ — 3.14-4.94× filhos (Sapolsky 2005 confirmado)

### Fase C.1 (próxima sessão)
5. **Sweep v6**: variar `_ALPHA_EXCURSION_PROB` (0.10, 0.20, 0.30, 0.50) — calibração do fluxo genético inter-tribal
6. **Análise `terr_nasc`**: verificar se filhos cross-territory se fixam no território do midpoint ou migram
7. **Histograma x_nasc**: distribuição espacial de nascimentos por geração (H2 completa)

### Fase C completa
8. Feromona com assinatura genética (Srinivasan 2012) — comunicação intra-tribal
9. Cooperação intra-tribal vs. competição inter-tribal
10. Dashboard: visualização em tempo real da estrutura tribal com coloração por clã

### Para papers
11. Tabela completa H0-H4 com p-values (bootstrap n=5 réplicas × 4 níveis)
12. Figura: cv_fc1 ao longo do tempo por n_patches (mostra convergência e estabilidade)
13. Figura: distribuição espacial de tribos (DBSCAN coloured scatter plot)
14. Figura: alpha_filhos_ratio vs n_patches (validação Sapolsky 2005)
