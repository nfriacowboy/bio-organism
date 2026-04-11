# Tarefa 4 — Fase D: Ética Emergente — Design Experimental

**Estado**: ⏳ INVESTIGAÇÃO EM CURSO (doc base criado; iniciar implementação após Fase C.2)
**Documento científico base**: `documentation/SNN_Etica_Emergente_Base_Cientifica.md`

---

## Reframing fundamental

> A Fase D NÃO consiste em programar ética.
> Consiste em **replicar condições ambientais e sociais conhecidas por produzir comportamento cooperativo/ético em organismos reais**, e observar se as mesmas condições produzem comportamentos análogos no organismo digital.

A ética não será inserida — será **observada ou não observada** como resultado de condições replicadas da biologia real.

---

## A Hipótese Central (reformulada)

> **Se** replicarmos as condições ambientais e sociais que — em organismos biológicos reais — produzem cooperação, partilha, altruísmo e normas de grupo, **então** o organismo digital deverá desenvolver comportamentos funcionalmente equivalentes como estratégias evolutivamente estáveis.

Esta hipótese tem duas consequências científicas:
1. Se **confirmada**: valida o organismo digital como modelo de dinâmicas sociais reais
2. Se **refutada**: identifica o que falta no modelo para capturar a complexidade social real

Ambos os resultados são publicáveis.

---

## Fundamento empírico — o que observamos em organismos reais

Ver documento completo: `documentation/SNN_Etica_Emergente_Base_Cientifica.md`

**Resumo das condições que produzem cooperação em biologia:**

| Condição | Organismo | Comportamento emergente | Referência |
|---|---|---|---|
| Alta densidade + escassez de comida | *C. elegans* | Feromona de agregação, dauer colectivo | Srinivasan et al. (2012) |
| Presença de predador | Peixes, aves, mamíferos | Comportamento de manada, alarmes | Hamilton (1971) |
| Parentesco genético elevado | Abelhas, formigas, esquilos | Altruísmo sacrificial, kin selection | Hamilton (1964) |
| Interacções repetidas | Primatas, humanos | Reciprocidade, tit-for-tat | Axelrod & Hamilton (1981) |
| Recurso partilhável não-rival | Micro-organismos, abelhas | Produção de bens públicos | West et al. (2007) |
| Observação social (outros vêem) | Primatas, humanos | Reputação, punição altruísta | Fehr & Gächter (2002) |
| Epigenética: fome parental | *C. elegans*, ratos | Offspring com comportamento de escassez | Rechavi et al. (2014) |

---

## Design Experimental Fase D

### Grupos experimentais

```
Mundo partilhado (2000×2000, recursos dinâmicos)
              │
    ┌─────────┴─────────┐
    │                   │
 TRIBO A             TRIBO B
(condições neutras)  (condições que produzem ética em biologia real)
    │                   │
    │  competem/interagem│
    └─────────┬─────────┘
              │
         Métricas ao longo de N gerações
```

### Condições de TRIBO A (controlo)
- Recursos abundantes e uniformes
- Sem pressão de predador
- Reprodução livre sem custo
- Sem comunicação inter-organismo

### Condições de TRIBO B (experimental — replicando condições biológicas)
Aplicadas progressivamente, cada uma derivada de evidência empírica:

**B.1 — Escassez cíclica de recursos** (inspirado em dauer *C. elegans*)
- Patches de comida com ciclos de depleção-regeneração
- Força dispersão e potencialmente cooperação para localizar novos patches
- Operacionalização: `patch_lifetime = 500 steps`, `regeneration_rate = 0.1`

**B.2 — Pressão de predador** (Hamilton 1971 — Selfish Herd)
- 1 predador por 15 organismos
- Cria pressão directa para agrupamento e sistemas de alarme
- Operacionalização: já planeado em Fase C.2

**B.3 — Parentesco e sinalização genética** (Hamilton 1964 — kin selection)
- Feromona com "assinatura genética" (distingue parentes de não-parentes)
- Organismos que ajudam parentes têm fitness colectivo superior
- Operacionalização: feromona tribal com distância genética como parâmetro

**B.4 — Interacções repetidas e memória social** (Axelrod 1981)
- Organismos reconhecem indivíduos que já encontraram (hash do genoma como ID)
- Histórico de interacções positivas → threshold de partilha menor
- Operacionalização: `social_memory[organismo_id] = história_de_interacções`

**B.5 — Epigenética social** (Rechavi 2014; Dias 2014)
- Organismos que experienciaram escassez transmitem essa "memória" nos pesos herdados
- Scaling factor nos pesos de forragem: multiplicado por `0.5 + 0.5 × energia_media_parental`
- Operacionalização: herança de R-STDP rolling mean com peso em função do histórico parental

---

## Métricas de observação

### Métricas de comportamento ético emergente

| Métrica | Definição | Análogo biológico |
|---|---|---|
| `gini_energia` | Desigualdade de energia na tribo (0=igual, 1=máximo desigual) | Distribuição de recursos |
| `taxa_partilha` | % steps com transferência de energia entre organismos | Altruísmo directo |
| `taxa_alarme_social` | Frequência de emissão de feromona de alarme quando predador detectado | Altruísmo de informação |
| `sacrificio_fitness` | Delta fitness individual em actos altruístas (custo médio por acto) | Custo biológico da cooperação |
| `ganho_colectivo` | Delta fitness médio da tribo por acto altruísta | Benefício colectivo |
| `r_Hamilton` | `sacrificio_fitness / ganho_colectivo` — deve ser < coeficiente de parentesco | Verificação kin selection |
| `coeficiente_cooperacao` | Correlação entre parentesco genético e taxa de ajuda | c/r ratio |

### Comparação Tribo A vs. Tribo B

| Critério | Medido em | Hipótese |
|---|---|---|
| Sobrevivência média | Steps/geração | B > A em condições de escassez |
| Diversidade genética | CV pesos por geração | B estabiliza; A diverge |
| Resiliência a perturbações | Recuperação após crise de recursos | B > A |
| Velocidade de evolução | Gerações para chegar a fitness>X | Depende da condição |
| `gini_energia` steady-state | Após 50 gerações | B < A (mais igualitário) |

---

## O papel da KAN como instrumento de leitura

A KAN NÃO implementa a ética — **lê** e **torna visíveis** os padrões comportamentais que emergiram.

Após N gerações, a análise das funções KAN aprendidas em cada tribo permite responder:
- Que inputs sociais (parentesco, alarme, energia relativa) influenciam as decisões?
- Com que funções? (linear, threshold, saturação?)
- As funções são diferentes entre Tribo A e Tribo B?
- Que "regras" emergiram — e são análogas às observadas em organismos reais?

**Exemplo de pergunta científica respondível**: "A Tribo B desenvolveu uma função `energia_vizinho → P(partilhar)` com threshold em 0.20 (análoga ao threshold de dauer em *C. elegans* de ~40% redução de food). A Tribo A não desenvolveu esta função. O que isso implica para a origem evolutiva do altruísmo?"

---

## Acceptance Criteria Fase D

- [ ] Tribo B (com condições biológicas replicadas) desenvolve comportamento cooperativo mensurável (taxa_partilha > 0.05/step)
- [ ] Tribo A (controlo neutro) mantém comportamento individualista (taxa_partilha < 0.02/step)
- [ ] `gini_energia` da Tribo B < `gini_energia` Tribo A após 20+ gerações
- [ ] Tribo B sobrevive mais que Tribo A em condição de escassez severa (p < 0.05)
- [ ] Funções KAN aprendidas por Tribo B mostram dependência de contexto social (inputs parentesco/alarme têm peso > random)
- [ ] Funções KAN são legíveis e interpretáveis (não apenas weights — curvas B-spline analisáveis)
- [ ] Pelo menos 1 comportamento análogo ao observado em biologia (listado na tabela acima) emerge sem ser programado

---

## Questões de investigação abertas

1. **Threshold de parentesco**: qual o mínimo de parentesco genético necessário para cooperação emergir? (biological analog: Hamilton's rule r > c/b)
2. **Punição altruísta**: organismos punem não-cooperadores espontaneamente? (Fehr & Gächter 2002)
3. **Estabilidade temporal**: o comportamento cooperativo persiste ou é eroded por "cheaters"?
4. **Selecção multi-nível**: a selecção opera no nível do indivíduo ou da tribo? (Wilson & Wilson 2007)
5. **Normas culturais**: comportamentos cooperativos são transmitidos via herança de pesos ou via aprendizagem? (distinguível porque podemos isolar cada mecanismo)
