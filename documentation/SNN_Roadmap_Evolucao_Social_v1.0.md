# Roadmap: Evolução Social do Organismo SNN
**Versão**: 1.0
**Data**: 2026-03-26
**Organismo de referência biológica**: *Caenorhabditis elegans* (base actual), *Drosophila melanogaster*, colónias de *Apis mellifera* (fases C+)

---

## Premissa Científica

O organismo SNN actual é baseado em *C. elegans*: 302 neurónios, quimiotaxia, aprendizagem hebbiana por recompensa. O projecto propõe escalar este organismo individual para populações com vida, morte, reprodução, comunicação, tribos e ética — seguindo a escada evolutiva real: de organismo unicelular/simples → colónia → sociedade → cultura.

> *"Nothing in biology makes sense except in the light of evolution."*
> — Theodosius Dobzhansky (1973)

---

## Pré-condição: Novo Mundo

Antes de qualquer fase, o mundo actual (grid 2D estático com comida permanente) precisa de evoluir.

### Mundo v2.0 — Requisitos

| Característica | Actual | Novo |
|---|---|---|
| Dimensão | 1000×1000 (fase 3) | Configurável, ex: 2000×2000 |
| Organismos | 1 | N (10–100) |
| Comida | Estática (reaparece em bloco) | **Dinâmica** — patches que crescem e desaparecem |
| Predadores | Nenhum | Introduzidos na Fase C.2 |
| Comunicação | Nenhuma | Feromona / sinal social |
| Tempo | Steps discretos | Steps com "idade biológica" |

### Modelo de Comida Dinâmica

Inspirado em patches de recursos naturais (*resource patch dynamics* — MacArthur & Wilson, 1967):
- Comida aparece em **clusters** (distribuição espacial agregada, não uniforme)
- Cada patch tem um **ciclo**: crescimento → pico → depleção → recuperação
- Taxa de regeneração dependente de parâmetros do ambiente
- Isto cria gradientes espaciais reais e pressão para mobilidade

**Referência académica**: Brown et al. (1999) — *Foraging theory, marginal value theorem* (MVT): organismos devem abandonar um patch quando a taxa de colheita cai abaixo da média do ambiente. Prevê exactamente quando o organismo deve mover-se — critério de validação observacional.

---

## Fase A — Reprodução Sexual e Herança de Conhecimento

### Motivação Biológica

*C. elegans* tem hermafroditas (auto-reprodução) e machos (reprodução cruzada). A reprodução cruzada gera diversidade genética. Em organismos mais complexos, a reprodução sexual é o mecanismo primário de recombinação de informação adaptativa.

A herança de **pesos SNN** como "genoma neuronal" tem base em:
- **Neuroevolution (NEAT)**: Stanley & Miikkulainen (2002) — *Evolving Neural Networks through Augmenting Topologies*, Evolutionary Computation 10(2)
- **Epigenetic inheritance**: Jablonka & Lamb (2005) — *Evolution in Four Dimensions* — transmissão de padrões de activação neural entre gerações
- **Baldwin Effect**: Baldwin (1896) — aprendizagem individual pode acelerar evolução genética; aprendizagem durante a vida guia selecção

### Implementação

**Genoma neuronal**: os pesos SNN (fc1, fc2, fc3, KAN) de cada organismo constituem o seu genoma.

**Reprodução**:
```
W_filho = crossover(W_pai_A, W_pai_B) + mutação(σ)

Crossover por bloco: cada layer herdada de um progenitor com P=0.5
Mutação: W += N(0, σ) com σ=0.01 (exploração genética)
```

**Condições de reprodução**:
- Ambos os progenitores com energia > limiar_reprodução (ex: 60%)
- Proximidade espacial (raio < 20 células)
- Energia consumida no acto (custo biológico real)

**Herança parcial de aprendizagem (epigenética)**:
- Rolling mean do R-STDP herdado com decay (filho começa "pré-tendenciado")
- Não herda epistemologia completa — só tendências (evitar Lamarckismo puro)

### Acceptance Criteria

- [ ] Pelo menos 2 organismos simulados no mesmo mundo sem crash
- [ ] Reprodução ocorre automaticamente quando condições satisfeitas
- [ ] Filho gerado com pesos misturados de ambos os pais (verificável por checksum)
- [ ] Filho aprende mais depressa que organismo com pesos aleatórios (baseline): prec@ep50 > prec_random@ep50
- [ ] Diversidade genética mantida ao longo de 10+ gerações (coeficiente de variação dos pesos > 0.05)
- [ ] Sem "genetic drift" total (não convergir todos para pesos idênticos)

---

## Fase B — Envelhecimento, Senescência e Morte

### Motivação Biológica

*C. elegans* tem duração de vida de ~21 dias (25°C). O envelhecimento é mediado pelas vias **daf-2/daf-16** (insulina/IGF-1). Organismos mais bem alimentados vivem mais (pleiotropia da restrição calórica).

**Senescência funcional** — degradação progressiva com a idade:
- Kenyon et al. (1993) — *A C. elegans mutant that lives twice as long as wild type*, Nature 366 (manipulação genética da longevidade)
- Partridge & Barton (1993) — *Evolution of Aging*: envelhecimento é seleccionado quando beneficia organismos jovens em detrimento dos velhos

**Pressão evolutiva criada**: organismos que acumulam mais recursos antes da senescência reproduzem mais → selecção para eficiência de forragem.

### Implementação

| Parâmetro | Valor sugerido |
|---|---|
| `vida_max` | 5000 steps (configurável) |
| `fertilidade_max_idade` | 70% de `vida_max` |
| `degradacao_velocidade` | Linear: v = v0 × (1 - 0.3 × idade/vida_max) |
| `degradacao_visao` | Ray precision reduz com idade |
| `morte` | Garantida em `vida_max` independente de energia |

### Acceptance Criteria

- [ ] Organismo morre ao atingir `vida_max` mesmo com energia > 0
- [ ] Velocidade de movimento observavelmente menor nos últimos 20% de vida
- [ ] Fertilidade cessa após 70% de `vida_max`
- [ ] Organismos fortemente alimentados produzem mais descendência (correlação energia_lifetime ~ n_filhos > 0.3)
- [ ] Distribuição etária da população é estável (pirâmide etária equilibrada após 20+ gerações)

---

## Fase C — Comunicação Inter-Organismo e Tribos

### Motivação Biológica

**Comunicação**: *C. elegans* usa **ascarosides** (feromónios) para comunicação química inter-individual — encontro de parceiros, aviso de stress, formação de dauer (estado dormente colectivo). Referência: Srinivasan et al. (2012) — *Identification of a neuroendocrine signaling pathway regulating C. elegans population density*, Nature Chemical Biology.

**Comportamento social em nematódeos**: de Bono & Bargmann (1998) — *Natural variation in a neuropeptide Y receptor homolog modifies social behavior and food response in C. elegans*, Cell. Variantes do gene *npr-1* determinam se organismos se alimentam solitariamente ou em grupo.

**Formação de grupos / tribos**:
Referências trans-espécie:
- Wilson (1971) — *The Insect Societies*: emergência de comportamento colectivo em insectos sociais
- Hamilton (1964) — *The Genetical Evolution of Social Behaviour*: **kin selection** — organismos cooperam mais com parentes próximos (r = coeficiente de parentesco)
- Axelrod & Hamilton (1981) — *The Evolution of Cooperation*, Science: cooperação emerge em jogos iterados sem necessidade de altruísmo puro
- **Stigmergy** (Grassé, 1959): comunicação indirecta via modificação do ambiente (ex: formigas deixam feromónia no caminho → outros seguem → reforço do caminho)

### Implementação

**Canal de feromona (stigmergy)**:
```
Quando organismo come → deposita sinal no grid (intensidade proporcional à quantidade)
Sinal difunde-se radialmente + decai com tempo (half-life configurável)
Outros organismos detectam via olfato (novo tipo de sinal além da comida)
Organismos com pesos SNN que respondem ao sinal = vantagem → seleccionado
```

**Formação de tribo**:
- Distância genética (cos-similarity de pesos) como proxy de parentesco
- Organismos "apareados" behavioralmente (seguem feromona uns dos outros) formam grupos de facto
- Tribo = cluster de organismos que interagem positivamente (partilha energia, feromona comum)

**Partilha de comida**:
- Quando dois organismos com diferença de energia > limiar estão próximos → transferência parcial
- R-STDP reforça partilha se fitness colectivo aumenta

### Acceptance Criteria

- [ ] Feromona visível no grid (debug visual) — deposição, difusão, decay
- [ ] Organismos seguem feromona de outros (taxa de convergência para fonte > random walk)
- [ ] Grupos estáveis emergem sem serem programados explicitamente (clustering mede-se por índice de Gini espacial)
- [ ] Organismos em grupo têm `vida_media` > organismos solitários (benefício mensurável)
- [ ] Partilha de comida ocorre espontaneamente entre organismos próximos geneticamente
- [ ] Tribo identificada como cluster: ≥3 organismos com distância genética < θ e distância espacial < R

---

## Fase C.2 — Predador

### Motivação Biológica

Em *C. elegans* o predador natural é *Pristionchus pacificus* (outro nematódeo carnívoro). Comportamento de fuga em *C. elegans* está bem documentado: mecanoreceptores (células ASH) detectam contacto → reversal behavior → dispersão.

A introdução de predadores tem efeitos profundos na evolução social:
- **Selecção de grupo**: Hamilton (1971) — *Geometry for the Selfish Herd*: prova matemática de que agrupar-se reduz risco individual de predação
- **Vigilância colectiva**: Lima & Dill (1990) — *Behavioral decisions made under the risk of predation*: grupos detectam predadores mais cedo
- **Alarm signaling**: Sherman (1977) — ground squirrels emitem alarmes que alertam parentes — custo individual, benefício colectivo

### Implementação

**Predador**:
- Agente simples (não-SNN): persegue o organismo mais próximo
- Velocidade: 1.5× a velocidade do organismo jovem, 0.8× do organismo sénior
- Kill: contacto directo → morte imediata do organismo
- Spawn: 1 predador por cada N organismos (densidade controlada)

**Resposta esperada (emergente)**:
- Organismos em grupo detectam predador via sinal social (feromona de alarme)
- Dispersão coordenada vs. solitários (comportamento de manada)
- Selecção para velocidade + vigilância

### Acceptance Criteria

- [ ] Predador implementado e funcional (mata organismos por contacto)
- [ ] Taxa de sobrevivência de organismos em tribo > organismos solitários (em condições idênticas)
- [ ] Sinal de alarme emerge espontaneamente (organismo detecta predador → emite feromona → outros afastam-se)
- [ ] Comportamento de manada (Selfish Herd geometry) observável: organismos tendem a posicionar-se no centro do grupo
- [ ] Velocidade média da população aumenta ao longo de gerações com predador vs. sem

---

## Fase D — Ética Emergente: Design Experimental Controlado

### Princípio fundamental

> A Fase D NÃO consiste em programar ética.
> Consiste em **replicar condições ambientais e sociais empiricamente documentadas** por produzirem comportamento cooperativo em organismos reais — e observar se emergem comportamentos análogos no organismo digital.

> *"Morality is a natural phenomenon."*
> — Frans de Waal (2006), *Primates and Philosophers*

Ver documento científico completo: `SNN_Etica_Emergente_Base_Cientifica.md`

### Hipótese experimental

**Se** replicarmos as condições que — em biologia — produzem cooperação e normas de grupo, **então** o organismo digital deverá desenvolver comportamentos funcionalmente equivalentes por selecção evolutiva.

| | Tribo A (controlo neutro) | Tribo B (condições biológicas replicadas) |
|---|---|---|
| **Recursos** | Abundantes e uniformes | Patches cíclicos (escassez real) |
| **Predação** | Ausente | Presente (Fase C.2 activa) |
| **Parentesco** | Não sinalizado | Feromona com assinatura genética |
| **Epigenética** | Herança simples de pesos | Herança modulated por histórico parental |
| **Hipótese short-term** | Vantagem individual | Custo individual |
| **Hipótese long-term** | Colapso por esgotamento / predação | Resiliência por cooperação |

**Ambos os resultados são publicáveis.** O valor científico está na medição rigorosa, não na confirmação da hipótese.

### Condições replicadas de biologia real

1. **Escassez cíclica** — *C. elegans* activa dauer colectivo (Srinivasan et al. 2012)
2. **Pressão de predador** — Selfish Herd geometry (Hamilton 1971)
3. **Sinalização de parentesco** — Kin selection (Hamilton 1964)
4. **Memória de interacções repetidas** — Reciprocidade (Axelrod & Hamilton 1981)
5. **Epigenética social** — Herança de experiência parental (Rechavi et al. 2014)

### KAN como instrumento de leitura (não de programação)

A KAN torna visíveis os padrões comportamentais emergentes:
- Que inputs sociais influenciam decisões? (parentesco, alarme, energia relativa do vizinho)
- Com que funções? (threshold, linear, saturação?)
- As funções são análogas às observadas em organismos reais?

Exemplo: se Tribo B desenvolve função `energia_vizinho → P(partilhar)` com threshold em ~0.20, isso é análogo ao threshold de starvation dispersal em *C. elegans* real — e é científicamente verificável.

---

## Questão: Usar WorldCraft/Minecraft como mundo?

### O que estás a perguntar

Provavelmente: usar um motor de mundo tipo Minecraft (voxel 3D, procedural, físicas) como substrato para o organismo.

### Análise

**A favor:**
- Mundos procedurais ricos — terreno variado, obstáculos, recursos naturais realistas
- Já existe: **MineRL**, **malmö** (Microsoft), **craftassist** (Facebook) são ambientes RL baseados em Minecraft
- Física mais realista — elevação, água, obstáculos criam pressões evolutivas novas

**Contra (para o nosso caso):**
- A complexidade visual de Minecraft não acrescenta nada ao SNN actual — o organismo "vê" em raios lineares 2D, não em 3D
- Overhead computacional enorme para simular 100 organismos simultaneamente
- Debugging extremamente difícil — não consegues visualizar feromona, campos SNN, etc.
- A ciência que queremos fazer é na dinâmica social, não na física do mundo

### Recomendação

**Não usar Minecraft/WorldCraft — expandir o mundo grid 2D actual com:**

| Feature | Implementação |
|---|---|
| Topografia | Zonas de velocidade diferente (pântano=lento, savana=rápido) |
| Obstáculos | Paredes/rochas no grid |
| Patches dinâmicos | Clusters de comida com ciclos de vida |
| Múltiplos biomas | Zonas com densidades diferentes |
| Visualização | **Mesa** (framework ABM Python) para render real-time |

**Alternativa sólida — Mesa (Python ABM):**
```
pip install mesa
```
Mesa é o standard académico para agent-based modelling em Python. Tem visualização web built-in, suporta grids, é compatível com o código actual, e é usado em investigação publicada. Referência: Masad & Kazil (2015) — *Mesa: An Agent-Based Modeling Framework*, SciPy Conference.

---

## Ordem de Implementação Sugerida

```
[AGORA]  Verificar se v80p é base suficiente → análise export 20260326T123130UTC

[Mundo v2.0]  Grid expandido + comida dinâmica (patches) + suporte N organismos
              + integração Mesa para visualização

[Fase A]  Reprodução + herança de pesos (2 organismos)
          → Testar: filho aprende mais depressa que random?

[Fase B]  Envelhecimento + morte programada
          → Testar: pressão evolutiva emerge?

[Fase C]  Feromona + partilha + tribos emergentes
          → Testar: fitness colectivo > individual?

[Fase C.2]  Predador
            → Testar: grupos sobrevivem mais?

[Investigação]  Como implementar ética num SNN? (paper review + design)

[Fase D]  Módulo ético + comparação tribal
```

---

## Sugestão adicional: Fase B.2 — Diversidade Genética

Entre a Fase B e C, considerar um mecanismo anti-inbreeding:
- Organismos com distância genética muito baixa (cos_sim > 0.98) têm fertilidade reduzida
- Garante diversidade → população mais resiliente a perturbações
- Referência: Charlesworth & Charlesworth (1987) — *Inbreeding depression and its evolutionary consequences*

---

## Resumo das Acceptance Criteria por Fase

| Fase | Critério-chave de sucesso |
|---|---|
| Mundo v2.0 | N organismos simultâneos sem crash; patches dinâmicos verificáveis |
| A | Filho aprende +20% mais depressa que pesos aleatórios |
| B | Correlação energia_lifetime ~ n_filhos > 0.3 |
| C | vida_media em grupo > solitário (significância estatística) |
| C.2 | Tribo sobrevive mais com predador do que organismos solitários |
| D | Tribo B desenvolve cooperação mensurável (taxa_partilha > 0.05); gini_energia B < A; Tribo B sobrevive mais em escassez severa |
