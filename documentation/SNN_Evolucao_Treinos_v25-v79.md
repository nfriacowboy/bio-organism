# Evolução do bio-organism: v25 → v79

Data: 2026-03-25

---

## Fase 1: Caos motivacional (v25–v43)

**Problema central**: o organismo colapsa sempre para uma única acção dominante.

### v25–v31 — Descoberta dos bugs de distribuição

- v25/v26: Δdist × 0.05/0.10 + turn bonus → avançar collapse (Δdist domina num mundo denso)
- v27: Anti-avançar guard activa a cada 10 ep → ineficaz
- **v28–v30**: p_av=70% desde ep1 — meses a investigar. Bug raiz: `_vetor_vazio()` usava eferente fixo `[0,0,0,1]` (parar) → pré-treino ANN com distribution mismatch → SNN sai a escolher avançar ~89%
- **v31** (bug #20 corrigido): eferente aleatorizado no pré-treino → **p_av=8-10% ep1**, prec=0.38-0.49 acima de random desde ep1

### v32–v43 — Acertos cirúrgicos na arquitectura de aprendizagem

- v38: tau_trace=2 (curto) → food credita avançar directamente → collapse. Revert para tau=20
- **v42** (bug #21): `lambda_naca=1e-6 SEMPRE` (era 0.1 nos primeiros 100 eps → destruía pesos ANN em cada episódio)
- **v43** (bug #22): `AGMP N_OBSERVACAO=99999` → modo inferência nunca activa durante treino
- v43: pela primeira vez prec=0.46-0.49 sustentado ep190-200 — SNN aprende acima do random

---

## Fase 2: Estabilização e tecto de precisão (v44–v57)

**Problema**: prec oscila, TURN-OFF/RECOVERY ciclam, nunca cruza threshold.

### v44–v49 — Encontrar o regime estável

- v44–v45: Descoberta do equilibrio: turn_bonus=0.30 + WRONG-ADV(thr=0.35,pen=0.70) sem TURN-OFF e sem ANTI-AVANÇAR guard
- v45: **prec_roll30=0.478** — máximo histórico até aqui. Mas ANTI-AVANÇAR guard ainda presente → crash ep309
- **v49**: params v45 sem guard → **prec_roll30=0.476 sustentado**, tecto confirmado ~0.47-0.48

### v50–v57 — Confirmação do tecto e melhoria de energia

- v51: energia 100→200 → vida média 650→1198 steps, prec=0.478 (mesmo tecto)
- v52/v53: qualquer `wrong_turn_penalty` destrói viragens (NACA assimetria)
- v53b/v56: biased_eps ganha prec *medida* (0.62-0.72) mas SNN real fica em 0.46-0.50 → fase 2 falha
- **v57** (1500 eps com threshold=0.48): max=0.4746 — **tecto SNN confirmado definitivamente em ~0.47-0.48**

---

## Fase 3: Curriculum multi-fase funcional (v58–v65)

**Problema**: threshold 0.48 inatingível — é preciso chegar à fase 2.

### v58 — Breakthrough: threshold_f1 baixado para 0.46

- Avançou fase 2 ep205
- Fase 2: 800 eps anti-avançar → ep1100 colapso avançar → ep1100-1500 **thriving** (comida=25-34/ep)
- Identificado: shaping (WRONG-ADV+turn_bonus) sabota fase 2

### v59–v64 — Refinamento do curriculum

- **v63**: shaping fase-condicional (só fase 1) → anti-avançar em fase 2 durou apenas ~85 eps (vs 800 em v58!)
- **v64**: threshold_f2 muda para `vida_media≥1500` (prec impossível em advance-dominant). Bug: RECOVERY dispara em fase 2 (p_av=0.82 é correcto!) → crashes. Max roll30=1364
- **v65** (bug #33 corrigido: RECOVERY só fase 1): **FASE 3 ATINGIDA ep221!** Vida 1200+ confirmada em fase 2. Mas fase 3: steps=625, comida=0 (0.015% demasiado esparso para visão linear)

---

## Fase 4: O problema do olfato em fase 3 (v66–v79)

**Problema raiz**: 0.015% densidade → P(encontrar comida com só visão) ≈ 0. R-STDP não bootstrapa sem reward.

### Arquitectura olfato (v66+): 42 inputs (era 39)

- 3 novos inputs: sectores circulares raio=100, normalização por 5.0
- Fase 3: P(olfato detec. comida) = 99% com R=100 (vs 1.5% visual)
- Fase 1: olfato saturado (=1.0 sempre)

### v66–v68 — Bug de bootstrap olfato

- v66: Fase 3 ep301 mas comida=0 — ciclo: sem reward→R-STDP não aprende olfato→sem navegação→sem comida
- v67/v68: olfato shaping (+0.10 sector dominante) → falhou. v67: sector dominante ≈ igual (Poisson(1.18)) → reforço advance → p_av=0.85+. v68: fix dom>60% → ainda steps=430, comida=0

### v69–v72 — Exploração de advance_bonus

- v69/v70: num_comida 150→1500, advance_bonus=+0.03 → insuficiente para sair do pirouette mode
- **v71**: advance_bonus=+0.10 → breakthrough pirouette→advance ep540, avg_comida=1.14. MAS plateau/declínio → straight-line charging (p_av→0.90)
- **v72**: advance_bonus=+0.05 + olfato-biased eps → melhorou (60% ep≥1 vs 52%), mas R-STDP ignora olfato aprendido como =1.0 em fase 1/2

### v73–v74 — Normalização relativa olfato

- v73 (bug #35): normalização relativa em TODAS as fases → em fase 2 com depleção activa turn bias → preso 243+ eps em fase 2
- v74: normalização apenas fase≥3. Mesmo declining pattern (avg 1.19→0.56): **bug #36** — ratio advance_bonus 16× food reward

### v75–v77 — Tentativas pirouette (sem advance_bonus)

- v75: pirouette INDUCTION (advance→turn quando lateral>frontal). Deadlock: p_av cai para 20% → zero R-STDP signal → permanente
- v76: + pirouette SUPPRESSION (turn→advance quando frontal>50%) + advance_bonus=+0.02. Ainda deadlock: induction cancela advance_bonus
- v77: advance_bonus CONDICIONAL (só quando olfato_frente>0.40). Deadlock: olfato uniforme (cada sector ~33%) → threshold raramente cruzado

### v78–v79 — Cap directo de p_av

- v78 (bug #39): cap via `wrong_advance_penalty` → **nunca disparava** em fase 3 (requer comida visível, que é <5% dos steps a 0.015%)
- **v79** (fix bug #39): avancar_bonus dinâmico no outer loop (`-0.05` se p_av>0.65, `+0.05` se ≤0.65) → cap PARCIALMENTE funcionou (steps 500-700 vs 484-526), bursts ocasionais (ep1450=2, ep1650=2, ep1980=4). Comida média ~0.3/ep

---

## Marcos e estado actual

| Versão | Marco |
|--------|-------|
| v20 | Primeiro SNN acima de random (threshold_lif fix) |
| v31 | Bootstrap fiável (eferente fix) |
| v43 | AGMP+NACA estáveis |
| v49 | Regime ótimo fase 1 confirmado |
| v57 | Tecto SNN=0.47-0.48 confirmado |
| v63 | Shaping fase-condicional — chave para fase 2 |
| v65 | **Fase 3 atingida pela primeira vez** |
| v66 | Olfato adicionado (42 inputs) |
| v79 | Cap p_av funcional, mas bootstrap R-STDP olfato ainda falhado |

---

## Próximo passo: v80

**Problema raiz**: o salto de densidade 2% → 0.015% é demasiado abrupto. R-STDP recebe ~0.3 food/ep — insuficiente para aprender associações olfato→acção fiáveis.

**v80 (proposta principal)**: densidade gradual em fase 3 — iniciar com 5000 comida (0.05%) e reduzir 10%/100 eps até 1500 (0.015%). O organismo aprende a navegar por olfato com sinal rico, depois generaliza progressivamente para ambiente esparso.

**Alternativas**:
- **v80b**: batch R-STDP por episódio — acumular traces durante ep, aplicar Δw proporcional a comida_total_ep no fim (sinal 50× mais forte por evento alimentar)
- **v80c**: RECOVERY olfato — quando fase 3 e comida=0 por 20+ eps consecutivos → reset p_av_cap para 0.55 temporariamente
