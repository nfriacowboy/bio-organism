# Filosofia de Aprendizagem: Plasticidade Dependente de Novidade

Data: 2026-03-25

---

## Princípio central

O organismo alterna dinamicamente entre dois modos:

- **Modo aprendizagem**: R-STDP activo, epsilon alto, pesos actualizados
- **Modo inferência**: R-STDP congelado, epsilon baixo, exploração do que foi aprendido

A transição entre modos é ditada por **novidade** (surpresa perante o ambiente), não por um schedule fixo de épocas.

---

## Estrutura de fases como curriculum de novidade

### Fase 1 — Aprender a comer e mover-se
- Ambiente: denso (5% comida), mundo pequeno (200×200)
- Novidade inicial: ALTA — organismo não sabe nada
- Modo de entrada: aprendizagem
- O que deve aprender: associação visão→acção, uso rudimentar do olfato
- Transição para inferência: quando RPE (reward prediction error) estabiliza em janela deslizante
- Transição para fase 2: quando janela de aprendizagem se esgota — não há mais novidade neste ambiente

### Fase 2 — Optimizar a busca em escassez
- Ambiente: esparso (2% comida), mundo maior (500×500)
- Novidade: ALTA no início — ambiente mudou, comida é mais rara, olfato tem gradiente real
- Reabre janela de plasticidade automaticamente (RPE sobe com a escassez)
- O que deve aprender: usar olfato para orientação direccional, optimizar trajectórias
- Transição para inferência/fase 3: quando performance estabiliza

### Fase 3 — Validação
- Ambiente: extremamente esparso (0.015% comida), mundo máximo (1000×1000)
- Não é para aprender — é para **verificar** se as fases anteriores produziram um organismo capaz
- Se falha aqui, o problema está nas fases anteriores, não na fase 3

---

## Problema identificado na implementação actual (v25–v79)

Em fase 1/2, o olfato está sempre saturado: todos os sectores = 1.0 (há comida em todo o lado).
O R-STDP vê `[1.0, 1.0, 1.0]` constantemente → **aprende a ignorar o olfato completamente**.

Quando em fase 3 o olfato passa a ter gradiente real (ex: [0.1, 0.8, 0.3]), o SNN trata esses
valores como sinal fraco e não actua sobre eles. O organismo chega à fase 3 sem ter aprendido
a ferramenta que precisava.

### Consequência
O ciclo de bootstrap é irreversível: sem avançar → sem comida → sem reward → sem aprendizagem olfato → sem avançar.

---

## Arquitectura proposta para v80+

### 1. Olfato com gradiente real desde fase 1

Em vez de raio=100 (satura tudo em ambiente denso), usar:
- Fase 1: raio=20–30 (só detecta comida muito próxima → gradiente direcional real mesmo a 5% densidade)
- Fase 2: raio=50–70
- Fase 3: raio=100

Ou alternativamente: **normalização relativa desde fase 1** — o sector dominante sempre = 1.0,
os outros proporcionais. Assim o SNN aprende a interpretar o gradiente mesmo com saturação.

### 2. Gate de plasticidade baseado em novidade

```
novidade = |reward_actual - reward_previsto_rolling|

se novidade > threshold_novidade:
    modo = APRENDIZAGEM  # R-STDP activo, epsilon alto
else:
    modo = INFERÊNCIA    # R-STDP congelado, epsilon=floor
```

Implementação simples: RPE como proxy de novidade.
`reward_previsto = media_movel(reward, janela=20)`
`novidade = abs(reward_actual - reward_previsto)`

### 3. Checkpoint de consolidação

Ao transitar do modo aprendizagem para inferência:
- Guardar snapshot dos pesos SNN
- Se performance em inferência declining por N episódios → reabrir plasticidade

### 4. Critério de avanço de fase

Substituir thresholds fixos de `precisao_direcional` ou `vida_media` por:

> "A taxa de novidade colapsou para próximo de zero AND performance mínima atingida"

Ou seja: o organismo está em inferência estável → não vai aprender mais aqui → avançar fase.

---

## Analogia biológica (C. elegans)

Este modelo espelha o que se sabe do C. elegans real:

| Biológico | Implementação |
|-----------|---------------|
| Plasticidade sináptica dependente de actividade | R-STDP activo só em modo aprendizagem |
| Quimiotaxia AWA/AWC modulada por experiência | Olfato com gradiente real desde fase 1 |
| Pirouette rate aumenta em gradiente descendente | Exploração aumenta quando novidade alta |
| Pirouette suprimido em gradiente ascendente | Inferência quando gradiente estável e positivo |
| Memória de jejum modula sensibilidade ao olfato | Starvation bonus já implementado (fase 2+) |

---

## Resumo executivo

| Problema actual | Solução proposta |
|-----------------|------------------|
| Olfato sempre 1.0 em fase 1 → SNN ignora | Raio menor ou normalização relativa desde fase 1 |
| R-STDP sempre activo → sem consolidação | Gate de plasticidade (novidade > threshold) |
| Thresholds fixos de avanço de fase | Avanço baseado em colapso da taxa de novidade |
| Fase 3 como target de aprendizagem | Fase 3 como validação — aprendizagem deve estar concluída na fase 2 |

---

## Ficheiros relevantes

- `trainer/src/snn/learning.py` — R-STDP, RPE, gate de plasticidade a implementar
- `trainer/src/world/curriculum.py` — lógica de fases e thresholds de avanço
- `trainer/src/world/olfato.py` — `calcular_olfato(grid, x, y, angulo, R)` — raio configurável por fase
- `trainer/src/training/trainer.py` — loop principal, onde gate de plasticidade deve ser aplicado
