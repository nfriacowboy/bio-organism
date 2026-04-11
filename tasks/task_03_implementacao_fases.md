# Tarefa 3 — Implementação das Fases A → C.2

**Estado**: ⏳ AGUARDA (#1 concluído)
**Roadmap detalhado**: `documentation/SNN_Roadmap_Evolucao_Social_v1.0.md`
**Fase D**: aguarda investigação — ver `task_04_fase_d_investigacao.md`

---

## Pré-condição: Mundo v2.0

Antes de qualquer fase implementar o novo mundo.

### Mundo v2.0 — subtarefas

- [ ] **Grid expansível**: N organismos simultâneos sem crash (`world/mundo.py` refactor)
- [ ] **Comida dinâmica por patches**:
  - Clusters espaciais (distribuição agregada — não uniforme)
  - Cada patch: ciclo crescimento → pico → depleção → recuperação (`patch_lifetime` configurável)
  - Referência: Marginal Value Theorem (Charnov 1976) — critério de quando abandonar patch
- [ ] **Mesa integration**: visualização web real-time dos N organismos + patches + feromona
  - `pip install mesa` — já compatível com Python/numpy
  - Grid agent-based: cada organismo é um `Agent`, mundo é um `Model`
- [ ] **Testes baseline**: 1 organismo no Mundo v2.0 tem comportamento equivalente ao v80p
  - acceptance: comida_media ≥ 80% do v80p nos primeiros 100 eps fase 3

---

## Fase A — Reprodução e Herança de Conhecimento

**Referências**: Stanley & Miikkulainen (2002) NEAT; Jablonka & Lamb (2005); Baldwin (1896)

### Subtarefas

- [ ] **Genoma neuronal**: serialização dos pesos SNN+KAN como "DNA"
  - `organism.get_genome()` → dict de tensors
  - `organism.set_genome(genome)` → carrega pesos
- [ ] **Crossover por layer**:
  ```python
  for layer in ['fc1', 'fc2', 'fc3', 'kan']:
      child[layer] = parent_A[layer] if random() > 0.5 else parent_B[layer]
  ```
- [ ] **Mutação gaussiana**: `W += N(0, σ=0.01)` após crossover
- [ ] **Condições de reprodução**:
  - Energia ambos > 60%
  - Distância espacial < raio_reprodução=20
  - Idade ambos < fertilidade_max=0.70×vida_max
- [ ] **Energia consumida**: -30% de energia para cada progenitor no acto
- [ ] **Herança epigenética parcial**: rolling mean R-STDP herdado com decay=0.3
  - filho começa "pré-tendenciado" mas não com memória completa

### Acceptance Criteria Fase A

- [ ] Filho gerado com pesos que são combinação verificável de ambos os pais
- [ ] Filho atinge `prec > 0.25` em 20 eps (vs. pesos aleatórios: ~50 eps)
- [ ] Diversidade genética mantida após 10 gerações (CV pesos > 0.05)
- [ ] Reprodução não causa crash quando múltiplos organismos no mesmo mundo
- [ ] Sem convergência prematura (todos organismos com pesos idênticos)

---

## Fase B — Envelhecimento, Senescência e Morte

**Referências**: Kenyon et al. (1993) Nature; Partridge & Barton (1993); Hamilton (1966) — *Moulding of Senescence*

### Subtarefas

- [ ] **Contador de idade** por organismo (em steps totais vividos)
- [ ] **Degradação de velocidade**: `v = v0 × (1 - 0.3 × idade/vida_max)`
- [ ] **Degradação de visão**: ray precision reduz linearmente nos últimos 20% de vida
- [ ] **Fertilidade**: cessa quando `idade > 0.70 × vida_max`
- [ ] **Morte programada**: ao atingir `vida_max=5000` steps independente de energia
- [ ] **Pressão evolutiva**: organismos com maior `comida_lifetime` têm mais descendência
  - Fitness tracking por organismo: `n_filhos`, `comida_lifetime`, `geracao`

### Acceptance Criteria Fase B

- [ ] Organismo morre ao atingir `vida_max` com energia > 0 (morte programada confirmada)
- [ ] Velocidade observavelmente menor nos últimos 20% de vida (mensurável)
- [ ] Correlação `comida_lifetime ~ n_filhos` > 0.30 (pressão evolutiva activa)
- [ ] Distribuição etária estável após 20+ gerações (pirâmide equilibrada)
- [ ] Fertilidade cessa após 70% de `vida_max` (verificável por log)

### Fase B.2 — Anti-inbreeding (sugestão)

- [ ] Coeficiente de similaridade genética entre parceiros (cosine_sim de pesos)
- [ ] Fertilidade reduzida quando `cos_sim > 0.95` (inbreeding detection)
- [ ] Previne genetic drift em populações pequenas

---

## Fase C — Comunicação e Tribos Emergentes

**Referências**: de Bono & Bargmann (1998) Cell; Grassé (1959) stigmergy; Axelrod & Hamilton (1981) Science; Hamilton (1964) kin selection; Wilson (1971)

### Subtarefas

**C.1 — Canal de feromona (stigmergy)**:
- [ ] Grid de feromona sobreposto ao grid de comida (array separado)
- [ ] Deposição: quando organismo come → `feromona[x,y] += intensidade`
- [ ] Difusão: gaussian blur com `σ=3` por step
- [ ] Decay: `feromona *= 0.98` por step (half-life ~35 steps)
- [ ] Detecção: novos inputs SNN — `{olf_social_esq, olf_social_frente, olf_social_dir}`
  - Mesmo mecanismo do olfato de comida mas grid diferente
  - Total inputs: 42 → **45** (adicionar 3 inputs sociais)

**C.2 — Feromona de alarme**:
- [ ] Quando organismo detecta predador (ou morre) → `alarme[x,y] += intensidade_alta`
- [ ] Outros organismos detectam alarme como input separado
- [ ] R-STDP associa alarme → fuga → sobrevivência

**C.3 — Partilha de comida**:
- [ ] Quando dois organismos com `ΔEnergia > 40%` estão a distância < 5 células
- [ ] Transferência automática de 20% da diferença por step
- [ ] R-STDP reforça proximidade a organismos com mais energia (reciprocidade)

**C.4 — Tribos emergentes**:
- [ ] Distância genética (1 - cosine_sim) como proxy de parentesco
- [ ] `tribo_id` emerge de clustering online (DBSCAN por distância genética + proximidade espacial)
- [ ] Tribo = grupo de ≥3 organismos com `distância_genética < 0.15` E `distância_espacial < 100`
- [ ] Feromona de tribo: cada organismo deposita feromona com "assinatura genética"
  - Outros membros da tribo respondem à assinatura; não-membros ignoram

### Acceptance Criteria Fase C

- [ ] Feromona visível no debug render (deposição, difusão, decay correctos)
- [ ] Organismos convergem para fontes de feromona mais depressa que random walk (p < 0.05)
- [ ] Grupos estáveis emergem espontaneamente (índice Gini espacial > 0.4 após 50 gerações)
- [ ] `vida_media` em grupo > `vida_media` solitário (estatisticamente significativo)
- [ ] Partilha de comida ocorre entre organismos próximos geneticamente
- [ ] Tribos identificáveis: ≥3 organismos com distância genética < 0.15 e espacial < 100

---

## Fase C.2 — Predador

**Referências**: Hamilton (1971) — *Geometry for the Selfish Herd*; Lima & Dill (1990); Sherman (1977) — alarm calls

### Subtarefas

- [ ] **Agente predador** (não-SNN):
  - Persegue organismo mais próximo
  - Velocidade: `1.5 × velocidade_organismo_jovem`
  - Kill: contacto directo → morte imediata
  - Spawn: 1 predador por cada 20 organismos
- [ ] **Feromona de alarme**: organismo que detecta predador emite alarme (ver C.2 acima)
- [ ] **Resposta de fuga**: input alarme → R-STDP aprende fuga
- [ ] **Selfish Herd geometry**: organismos tendem a posicionar-se no centro do grupo
  - Medir: distribuição de posições relativas ao centroide do grupo

### Acceptance Criteria Fase C.2

- [ ] Predador implementado e mata organismos por contacto
- [ ] Taxa sobrevivência em grupo > solitário (mesmas condições, p < 0.05)
- [ ] Sinal de alarme emerge espontaneamente (sem ser programado explicitamente)
- [ ] Velocidade média da população aumenta ao longo de gerações com predador
- [ ] Comportamento Selfish Herd observável (organismos evitam periferia do grupo)

---

## Fase D — Módulo Ético via KAN

**Estado**: ⏳ AGUARDA investigação (ver `task_04_fase_d_investigacao.md`)

**Hipótese arquitectural**: KAN como camada de controlo executivo ético
- SNN = sistema límbico (instinto, sobrevivência)
- KAN actual = integração sensório-motora
- **KAN ético** = córtex pré-frontal (controlo executivo, regras sociais)

**Conexão biológica**:
- PFC em mamíferos inibe respostas límbicas e aplica normas sociais
- KAN aprende funções interpretáveis → regras éticas legíveis por humanos
- Diferente de MLP: podemos VER o que o organismo "aprendeu" como ética

Ver `task_04_fase_d_investigacao.md` para detalhes da investigação necessária.
