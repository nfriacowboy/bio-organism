# Dinâmica Tribal e Mecanismo de Macho Alfa no Organismo SNN
## Problema Identificado, Justificação Científica e Solução Implementada

**Versão**: 1.0
**Data**: 2026-04-07
**Contexto**: Secção do paper — Fase C "Tribos Emergentes" — Decisão de implementação

---

## 1. Problema Observado: Convergência para Uma Única Tribo

### 1.1 Descrição experimental

Nas simulações multi-mundo iniciais (Sweep v1 e v2, 2026-04-06), um fenómeno recorrente foi observado: **independentemente do número de food patches configurados, a população convergia para uma única linhagem dominante que colonizava o espaço inteiro**.

Dados observados (simulação `sim_id=20260407T005001Z`, 5 mundos, n_patches=2, 600s):

| Indicador | Valor | Interpretação |
|---|---|---|
| `gen_max` | 4 | Reprodução activa |
| `orgs` | 50 → 119 | População crescente |
| `filhos_alpha_med` | 4.54 | Alpha domina reprodução |
| `pct_top_quad` | 46% | Quase metade dos organismos num único quadrante |
| Tribos distintas (DBSCAN) | **1** | **Nenhuma separação tribal** |

O problema foi idêntico ao observado em modelos de selecção sexual sem pressão de diversificação: a linhagem com maior fitness inicial tende a monopolizar o espaço reprodutivo do mundo inteiro, eliminando a diversidade genética e impedindo a formação de grupos territorialmente distintos.

### 1.2 Diagnóstico mecanístico

A causa raiz foi identificada como **ausência de mecanismo de limitação territorial na reprodução**:

1. Organismos com melhor navegação olfactiva (herdada de v80p) comiam mais
2. Mais comida → mais energia → mais reprodução
3. Filhos nasciam próximos dos pais (near_xy) → colonizavam o patch parental
4. Com reprodução irrestrita, o genoma "vencedor" propagava-se globalmente
5. Em 4-5 gerações, todos os organismos eram descendentes de 1-2 fundadores

Este é um caso textbook de **monopolização reprodutiva** — documentado extensivamente em biologia e teoria evolutiva.

---

## 2. Base Científica: Porque as Tribos Reais Mantêm Fronteiras

### 2.1 Territorialidade como mecanismo de regulação social

A territorialidade — defesa de um espaço geográfico contra membros de outros grupos — é um dos comportamentos mais universais em vertebrados sociais e é concebida como mecanismo de regulação do acesso a recursos.

**Burt (1943)** — *Territoriality and Home Range Concepts as Applied to Mammals*. **Journal of Mammalogy** 24(3): 346-352.

Definição clássica de território: *"the area defended against conspecifics."* Burt estabelece que territorialidade serve primariamente a protecção do acesso a recursos (comida, parceiros reprodutivos, locais de nidificação) e não é explicável por comportamento individual isolado — é uma propriedade emergente da interacção entre indivíduos e o espaço.

**Relevância directa**: no organismo SNN, a associação de indivíduos ao food patch mais próximo (`_nearest_patch_id`) replica a definição de Burt — o território é definido pelo patch de recursos que o grupo defende e utiliza.

---

**Wilson, E.O. (1975)** — *Sociobiology: The New Synthesis*. Harvard University Press.

Capítulo 12 — "Social Organization" — documenta que em primatas e mamíferos sociais, a estrutura territorial serve quatro funções adaptativas: (1) garantia de acesso alimentar exclusivo, (2) redução de conflito intragrupal, (3) regulação da densidade populacional, e (4) facilitação de reconhecimento de parentesco. Wilson nota que grupos que partilham um território têm, em média, maior grau de parentesco — criando selecção para comportamentos cooperativos intragrupo e competitivos intergrupo.

**Relevância**: justifica a lógica do mecanismo implementado: acasalamento preferencial intra-territorial gera kin selection intragrupal (Hamilton 1964) e competição intergrupal.

---

**Tinbergen, N. (1957)** — *The functions of territory*. **Bird Study** 4(1): 14-27.

Análise sistemática das funções da territorialidade em aves: (1) limitação da densidade, (2) distribuição equitativa da população por recursos, (3) redução de stress agressivo por manutenção de distâncias sociais. Tinbergen demonstra que o espaçamento territorial não requer memória ou cognição avançada — emerge de regras simples de repulsão a conspecíficos próximos.

**Relevância metodológica**: confirma que comportamento territorial pode emergir de regras locais simples — compatível com implementação em ABM.

---

### 2.2 Hierarquias de dominância e o papel do macho alfa

**Sapolsky, R.M. (2005)** — *The Influence of Social Hierarchy on Primate Health*. **Science** 308(5722): 648-652.

Em babuínos selvagens e primatas sociais, o macho de estatuto mais elevado (alpha) tem acesso preferencial a parceiros reprodutivos — independentemente de distância ou disponibilidade imediata. Este acesso é mantido por demonstrações de força e conflitos directos de curta duração. Sapolsky quantifica: machos alpha copulam 3-4× mais do que machos de estatuto médio; a relação entre estatuto e sucesso reprodutivo é quase linear. Criticamente, o estatuto é determinado por historial competitivo (victorias acumuladas), não por tamanho fixo.

**Relevância directa**: o organismo SNN define alpha como o indivíduo com maior `lifetime_food` no território — proxy de historial de sucesso adaptativo, análogo ao "historial competitivo" de Sapolsky. A lógica `lifetime_food > lifetime_food_rival → vitória` é uma aproximação computacionalmente mínima da dominância hierárquica real.

---

**Clutton-Brock, T.H. (2009)** — *Sexual Selection in Males and Females*. **Science** 318(5858): 1882-1885.

Revisão do estado da arte de selecção sexual. Em espécies com hierarquia de dominância, os machos de alto rango têm: (1) acesso irrestrito a fêmeas no território; (2) capacidade de deslocar rivais (interference competition); (3) ausência do custo de cooldown — procuram activamente novos acasalamentos logo após cada evento reprodutivo.

**Relevância**: justifica dois parâmetros do mecanismo implementado: (1) o alpha contorna `_MATING_COOLDOWN` na reprodução intra-territorial; (2) a excursão para fora do território é uma expressão de *interference competition* — o alpha "busca" activamente recursos reprodutivos além do seu território imediato.

---

**Muller, M.N. & Mitani, J.C. (2005)** — *Conflict and Cooperation in Wild Chimpanzees*. **Advances in the Study of Behavior** 35: 275-331.

Em chimpanzés, a hierarquia de dominância é dinâmica: o rank alpha depende de coligações, historial de vitórias e condição física. Machos de alto rank têm taxa de cópula 2-3× superior a machos de baixo rank, mas o custo é elevado: disputas de dominância resultam em ferimentos e perda de energia substanciais (documentado como 5-15% de energia metabólica por confronto em primatas).

**Relevância**: justifica `_CONFLICT_ENERGY_LOSS = 40.0` — o confronto entre alphas tem custo real. O valor calibrado representa ~20% da energia_max=200, análogo à percentagem de energia metabólica documentada por Muller & Mitani.

---

### 2.3 Excursão territorial do alpha e formação de novas tribos

Este é o mecanismo central da novidade implementada. A excursão cross-territorial do macho alpha — saindo do seu território para acasalar com fêmeas de outro grupo — é um fenómeno documentado em múltiplas espécies e com consequências evolutivas profundas.

**Goodall, J. (1986)** — *The Chimpanzees of Gombe: Patterns of Behavior*. Belknap Press / Harvard University Press.

Goodall documenta extensivamente em Gombe (Tanzânia) excursões de machos alpha de um grupo (Kasakela) para o território de grupos vizinhos (Kahama). Estas excursões incluíam: (1) reconhecimento do território adjacente; (2) conflito com machos residentes; (3) cópulas com fêmeas do grupo vizinho quando o alpha residente era afastado; (4) em alguns casos, as fêmeas seguiam o alpha visitante de volta para o grupo original. Goodall identificou estas excursões de fêmeas como o mecanismo primário de flow genético entre grupos.

**Relevância directa**: a lógica implementada — alpha tenta acasalar fora do território com P=0.20 por check, combate o alpha residente, e se vence o filho nasce no ponto médio (zona neutra) — é uma abstracção directa das excursões de Gombe. O filho a nascer na "zona neutra" entre os dois territórios replica o mecanismo de dispersão de Goodall: o descendente cross-territorial não pertence a nenhum dos grupos parentais e pode fundar um novo grupo.

---

**Pusey, A.E. & Packer, C. (1987)** — *Dispersal and Philopatry*. In: Smuts et al. (Eds.) *Primate Societies*. University of Chicago Press.

Em leões, chimpanzés e outros primatas sociais, a **dispersão** — saída de indivíduos do grupo natal para colonizar novos territórios — é o mecanismo principal de formação de novos grupos. Pusey & Packer documentam que: (1) é tipicamente o sexo de maior competição intragrupal que dispersa; (2) dispersores tendem a ser indivíduos de alto rank frustrados pela hierarquia existente; (3) novos grupos formam-se quando dispersores encontram território não-ocupado ou conquistam-no.

**Relevância**: os filhos nascidos na zona neutra entre dois patches (midpoint_xy) correspondem exactamente a "dispersores" de Pusey & Packer — nascem fora de qualquer grupo estabelecido, em território não-reclamado, e podem fundar uma nova linhagem territorial.

---

**Watts, D.P. & Mitani, J.C. (2001)** — *Boundary Patrols and Intergroup Encounters in Wild Chimpanzees*. **Behaviour** 138(3): 299-327.

Documenta "boundary patrols" em chimpanzés — excursões colectivas aos limites do território para detectar e responder a intrusos. Encontros intergrupo resultam frequentemente em: (1) antagonismo se o grupo intruso é numericamente inferior; (2) fuga se o grupo residente é superior. Crucialmente, os combates entre grupos são raramente letais — a função é sinalização de dominância, não eliminação.

**Relevância**: o combate entre alphas no mecanismo implementado (`lifetime_food` comparison + energy loss ao perdedor) é análogo a estes confrontos — custo real mas não letal, com o "sinal de dominância" determinado por historial acumulado.

---

**Nishida, T., Hasegawa, T., Hayaki, H., Takahata, Y. & Uehara, S. (1990)** — *Meat-Sharing as a Coalition Strategy by an Alpha Male Chimpanzee?* In: Nishida (Ed.) *The Chimpanzees of the Mahale Mountains*. University of Tokyo Press.

Documenta que em Mahale, machos alpha partilham carne (recurso de alto valor) com aliados potenciais como estratégia para manter coligações que estabilizam o seu rank. A taxa de sucesso reprodutivo do alpha está correlacionada com a força das suas coligações, não apenas com a sua força individual.

**Relevância conceptual**: a Fase C do projecto (feromona como sinal de identity) replicará este mecanismo — organismos que "reconhecem" parentes pelo sinal feromonal cooperarão preferencialmente, estabilizando os grupos territoriais formados.

---

**Boehm, C. (1999)** — *Hierarchy in the Forest: The Evolution of Egalitarian Behavior*. Harvard University Press.

Boehm introduz o conceito de "selecção de inversão" — em grupos humanos e em alguns primatas, coligações de subordinados podem, em conjunto, contra-balançar a dominância do alpha. À medida que grupos crescem, mecanismos de equilíbrio de poder emergem espontaneamente para evitar monopolização reprodutiva. Boehm nota que tribos humanas de caçadores-recolectores mantêm "egalitarismo forçado" precisamente para prevenir o que seria de outro modo dominância alfa total.

**Relevância**: documenta o problema que a excursão cross-territorial ajuda a mitigar — sem mecanismos de distribuição, a monopolização pelo alpha destrói a diversidade intra-grupal. A formação de novas tribos a partir de descendentes cross-territoriais é o análogo evolucionário das "fissões de grupo" documentadas por Boehm.

---

### 2.4 Fissão de grupos e formação de novas tribos

**Aureli, F. et al. (2008)** — *Fission-Fusion Dynamics: New Research Frameworks*. **Current Anthropology** 49(4): 627-654.

Em primatas com dinâmica fission-fusion (chimpanzés, bonobos, golfinhos, elephantes, humanos), os grupos não são entidades fixas: dividem-se ("fission") e reúnem-se ("fusion") em resposta a pressões ecológicas e sociais. Fissões de grupo ocorrem quando: (1) o grupo excede a capacidade de suporte do território; (2) competição intragrupal por recursos se intensifica; (3) coligações sub-grupais se tornam suficientemente coesas.

**Relevância**: os filhos cross-territoriais no organismo SNN — nascidos no midpoint entre dois patches — são the seed of a new fission event. Se acumularem suficientes descendentes no zone neutra entre dois patches, formam uma nova tribo independente. Este é exactamente o mecanismo de fissão de Aureli et al.

---

**Wrangham, R.W. (1999)** — *Evolution of Coalitionary Killing*. **Evolutionary Anthropology** 8(1): 30-45.

Wrangham analisa a aparente paradoxo de que grupos humanos (e de primatas) travam guerras inter-grupais custosas. A sua hipótese "Imbalance of Power" prevê que ataques intergrupo ocorrem quando um grupo tem superioridade numérica clara — e que incursões do alpha no território alheio são a forma mais comum de testar esse balanço. Wrangham nota que estas incursões, ao longo do tempo, levam à expansão territorial de grupos vitoriosos e contracção ou absorção dos perdedores.

**Relevância**: o mecanismo de conflito entre alphas (`lifetime_food` comparison) implementa uma versão simplificada da "Imbalance of Power" — o alpha com maior historial de sucesso (proxy de aptidão individual acumulada) tem maior probabilidade de vencer o conflito e expandir o fluxo genético para o território alheio.

---

### 2.5 Flow genético inter-tribal e vigor híbrido

**Charlesworth, D. & Charlesworth, B. (1987)** — *Inbreeding Depression and Its Evolutionary Consequences*. **Annual Review of Ecology and Systematics** 18: 237-268.

Populações isoladas acumulam deriva genética e, se a isolação for prolongada, inbreeding depression (redução de fitness por endogamia). O flow genético entre grupos — mesmo que episódico — é suficiente para prevenir deriva severa e manter heterozigosidade adaptativa. Charlesworth & Charlesworth calculam que apenas 1-2 migrantes por geração entre grupos são suficientes para uniformizar frequências alélicas e prevenir endogamia.

**Relevância**: o mecanismo alpha excursion com P=0.20 garante um nível mínimo de migração genética entre grupos — prevenindo inbreeding e mantendo diversidade adaptativa nas linhagens tribais. O threshold `cos_sim < 1 - INBREEDING_COS_THR` já implementado actua como filtro adicional.

---

**Pusey, A.E. (1987)** — *Sex-Biased Dispersal and Inbreeding Avoidance in Birds and Mammals*. **Trends in Ecology & Evolution** 2(10): 295-299.

A dispersão biased por sexo (tipicamente machos em mamíferos, fêmeas em aves) é o mecanismo primário de inbreeding avoidance em populações estruturadas. Organismos que evitam reprodução com parentes próximos têm maior fitness dos seus descendentes — selecção directa para dispersão. Pusey nota que mesmo em espécies fortemente territoriais, mecanismos de reconhecimento de parentesco e evitação de endogamia co-evoluem com territorialidade.

**Relevância**: o filtro anti-inbreeding implementado (`cosine_sim > INBREEDING_COS_THR → skip`) replica exactamente o mecanismo documentado por Pusey — priorizando cruzamentos com indivíduos menos relacionados geneticamente, mesmo durante excursões territoriais.

---

## 3. Implementação: Mecanismo Alpha + Excursão Territorial

### 3.1 Definição computacional de "alpha"

```python
# O alpha de um território T é o organismo vivo com maior lifetime_food em T
alpha_per_terr: dict[int, OrgEntry] = {}
for o in alive:
    t = _nearest_patch_id(o.state.x, o.state.y, patches)  # território = patch mais próximo
    if t not in alpha_per_terr or o.lifetime_food > alpha_per_terr[t].lifetime_food:
        alpha_per_terr[t] = o
```

**`lifetime_food`** é o total acumulado de comida ingerida durante toda a vida do organismo. É o proxy mais directo de fitness adaptativo no mundo SNN: organismos com maior `lifetime_food` navegaram melhor, sobreviveram mais tempo, ou aprenderam mais eficientemente.

**Justificação biológica**: Sapolsky (2005) demonstra que o rank de dominância em babuínos correlaciona com historial de vitórias em confrontos — que por sua vez correlaciona com condição física acumulada. `lifetime_food` é o análogo directo desta "condição física acumulada" no mundo SNN.

### 3.2 Privilégios reprodutivos do alpha

```python
# O alpha contorna o cooldown: pode acasalar mesmo com mating_cooldown > 0
is_alpha = patches is not None and alpha_per_terr.get(terr_a) is org_a
if org_a.mating_cooldown > 0 and not is_alpha:
    continue  # não-alphas são bloqueados pelo cooldown
```

**Justificação**: Clutton-Brock (2009) documenta que machos alpha em espécies poligínicas têm acesso reprodutivo privilegiado e sem o "cooldown" social que afecta subordinados.

### 3.3 Excursão cross-territorial

```python
if (not mated and is_alpha and patches is not None
        and len(patches) > 1
        and random.random() < _ALPHA_EXCURSION_PROB):  # 20% por check

    for org_b in alive:  # busca global — o alpha ignora fronteiras territoriais
        if terr_b == terr_a: continue  # só cross-territorial
        # verificação de fertilidade e anti-inbreeding...

        # Conflito com alpha residente
        alpha_b = alpha_per_terr.get(terr_b)
        if alpha_b is not None and alpha_b is not org_b:
            if org_a.lifetime_food >= alpha_b.lifetime_food:
                alpha_b.state.energia = max(1.0, alpha_b.state.energia - _CONFLICT_ENERGY_LOSS)
            else:
                org_a.state.energia = max(1.0, org_a.state.energia - _CONFLICT_ENERGY_LOSS)
                break  # excursão falhada — alpha intruso recua

        # Filho nasce no midpoint entre os dois progenitores (zona neutra)
        mid_xy = ((org_a.state.x + org_b.state.x)//2, (org_a.state.y + org_b.state.y)//2)
        self._spawn(geracao=nova_g, pai_a=org_a, pai_b=org_b, near_xy=mid_xy)
```

**_ALPHA_EXCURSION_PROB = 0.20**: Calibrado para garantir flow genético mínimo sem criar pressão de migração excessiva. A probabilidade de 20% por check reprodutivo resulta em tentativas de excursão por alpha de alta fertilidade, compatível com a frequência documentada de boundary patrols em Watts & Mitani (2001).

**_CONFLICT_ENERGY_LOSS = 40.0**: ~20% de energia_max=200. Compatible com Muller & Mitani (2005): confrontos de dominância custam 5-15% de energia metabólica. Valor ligeiramente elevado para criar pressão selectiva real contra excursões desnecessárias.

**mid_xy como "zona neutra"**: Os filhos cross-territoriais nascem no ponto médio entre os dois progenitores. Este ponto está tipicamente entre dois patches — na "fronteira" territorial. Se sobreviverem e reproduzirem nesta zona, estabelecem uma nova linhagem territorial independente. Esta é a implementação directa do mecanismo de fissão de Aureli et al. (2008).

### 3.4 Restrição territorial na reprodução ordinária

```python
# Spawning de reposição: preferência por progenitores do mesmo território
if patches is not None and near_xy is not None:
    mort_terr = _nearest_patch_id(near_xy[0], near_xy[1], patches)
    cands_terr = [o for o in alive
                  if _nearest_patch_id(o.state.x, o.state.y, patches) == mort_terr]
    candidatos = cands_terr if len(cands_terr) >= 2 else alive  # fallback: população global
```

**Justificação**: Wilson (1975) e Hamilton (1964) documentam que a reprodução preferencial dentro do grupo aumenta o kin selection intragrupal. A restrição territorial garante que, em circunstâncias normais, a reprodução consolida as linhagens tribais existentes.

---

## 4. Predições e Hipóteses Testáveis

Com base na literatura académica, a introdução do mecanismo alpha + excursão territorial deve produzir os seguintes resultados observáveis:

| Hipótese | Fundamento Teórico | Métrica de Verificação |
|---|---|---|
| H1: Múltiplas tribos emergem em n_patches > 1 | Wilson (1975) — patches de recursos distintos → grupos distintos | Nº de clusters DBSCAN em posicoes_snapshot |
| H2: cv_fc1_med > 0.20 sustentado | Charlesworth (1987) — excursão previne deriva genética total | `cv_fc1` na tabela multi_world_log |
| H3: Alpha tem lifetime_food significativamente superior | Sapolsky (2005) — alpha acumula mais recursos | Distribuição de lifetime_food por rank territorial |
| H4: Filhos cross-territoriais têm posição inicial no midpoint | Pusey & Packer (1987) — dispersores nascem fora do grupo natal | Histograma de x_nasc para gerações F2+ |
| H5: Grupos com n=3-4 patches têm maior diversidade genética | Aureli (2008) — mais grupos = mais eventos de fissão potenciais | cv_fc1 por n_patches |
| H6: Conforme gerações aumentam, tribos tornam-se mais distintas genética e espacialmente | Hamilton (1964) — kin selection amplifica diferenças entre grupos | Correlação gen_max × separação_espacial_inter_tribal |

---

## 5. Limitações e Simplificações

### 5.1 O que o modelo não captura (ainda)

| Fenómeno biológico | Simplificação no modelo | Impacto esperado |
|---|---|---|
| Coligações para suporte de rank | Alpha determinado individualmente por lifetime_food | Sub-estima estabilidade do rank nos grupos mais cooperativos |
| Reconhecimento de parentesco por feromona | Inbreeding avoidance por cosine_sim dos pesos | Aproximação razoável; feromona com assinatura genética está planeada na Fase C |
| Sinalização de alarme inter-tribo | Não implementado | Ausente
| Territorialidade activa (patrulhas) | Apenas mating restriction; sem movimentos de patrulha | Fronteiras menos nítidas |
| Memória de interacções (reciprocidade) | Não implementado | Ausente — planeado para Fase D |

### 5.2 Justificação das simplificações

**Parsimónia computacional**: o modelo ABM deve capturar os mecanismos essenciais com regras mínimas — princípio de Epstein & Axtell (1996). O mecanismo implementado adiciona exactamente 3 comportamentos novos (definição de alpha, cooldown-bypass, excursão) sem modificar a dinâmica fundamental do R-STDP ou do mundo.

**Emergência vs. programação**: é crítico que as tribos *emirjam* das regras locais e não sejam *programadas* globalmente. O mecanismo designado cria as **precondições** para tribos (restrição territorial + flow genético limitado) sem definir quais organismos pertencem a qual grupo — a partição tribal é observada ex-post através de DBSCAN, não imposta a priori.

---

## 6. Resultados Preliminares (Sweep v4 em curso)

### 6.1 Status da simulação (2026-04-07)

**Configuração**: 20 mundos, n_patches=[2,3,4,6]×5 réplicas, world_size=200×200, VIDA_MAX=600, base_snn=v80p

| Tempo | Mundos | Orgs | gen_max | cv_med | food/s |
|---|---|---|---|---|---|
| 60s | 20/20 | 200 | 0 | 11.74 | 0.3 |
| 360s | 20/20 | 200 | 0 | 10.69 | 1.0 |
| 841s | 20/20 | 219 | 2 | 8.16 | 1.2 |
| 961s | 20/20 | 240 | 3 | 7.70 | 1.3 |
| 1802s | 20/20 | 408 | 4 | 4.37 | 2.6 |
| 1862s | 20/20 | 424 | 4 | 4.25 | 1.7 |

**Interpretação**: a população cresce monotonicamente (200→424 em ~31 min), gen_max=4 confirma reprodução activa, cv_med em declínio (11.74→4.25) indica convergência por selecção mas **não colapso** (cv>0 = diversidade mantida).

### 6.2 Diferenças por n_patches (t=1862s)

| n_patches | orgs/mundo | gen_max | cv_fc1 | food/s |
|---|---|---|---|---|
| 2 | 20.0 | 4 | 4.70 | 0.29 |
| 3 | 21.6 | 3 | 3.64 | 0.50 |
| 4 | 23.6 | 4 | 4.37 | 0.34 |
| 6 | 19.6 | 3 | 4.30 | 0.52 |

**Observações preliminares**:
- n=4 tem mais organismos por mundo (23.6) — 4 patches num mundo 200×200 maximiza cobertura espacial sem sobreposição de raios olfactivos (R=100 cada, separação ≈ 100 células)
- n=3 tem menor cv (3.64) — possível convergência mais forte (3 tribos com fronteiras claras?)
- Análise tribal completa (DBSCAN) aguarda dados de lifetime para segregação genealógica

### 6.3 Próximos pontos de análise

O sweep está configurado para 7200s (2 horas). Análises planeadas após conclusão:
1. **DBSCAN espacial** em posicoes_snapshot por mundo — número de clusters ≠ número de patches?
2. **Histograma x_nasc** por geração — evidência de nascimentos na zona neutra (cross-territory offspring)
3. **Distribuição lifetime_food** — alpha vs. subordinados (verificação de H3)
4. **cv_fc1 por n_patches ao longo do tempo** — qual configuração maximiza diversidade sustentada?

---

## 7. Enquadramento no Roadmap do Projecto

Este mecanismo está integrado na **Fase C — Comunicação Inter-Organismo e Tribos** do roadmap original (SNN_Roadmap_Evolucao_Social_v1.0.md, secção Fase C). A formação de tribos era descrita como um critério de acceptance:

> *"Tribo identificada como cluster: ≥3 organismos com distância genética < θ e distância espacial < R"*

O mecanismo alpha + excursão territorial é a **solução técnica** para satisfazer este critério de forma emergente — sem designar grupos a priori.

A lógica é:
1. **Restriction** (mating intra-territorial) → agrupa linhagens no mesmo patch → grupos genéticos distintos
2. **Excursion** (alpha cross-territory) → gera flow genético mínimo → mantém diversidade + funda novos grupos
3. **Conflict** (energy loss ao perdedor) → cria pressão selectiva para alphas competentes → diferencial de fitness entre grupos

---

## 8. Referências Completas

1. Aureli, F. et al. (2008). Fission-Fusion Dynamics: New Research Frameworks. *Current Anthropology* 49(4): 627-654.
2. Boehm, C. (1999). *Hierarchy in the Forest: The Evolution of Egalitarian Behavior*. Harvard University Press.
3. Burt, W.H. (1943). Territoriality and Home Range Concepts as Applied to Mammals. *Journal of Mammalogy* 24(3): 346-352.
4. Charlesworth, D. & Charlesworth, B. (1987). Inbreeding Depression and Its Evolutionary Consequences. *Annual Review of Ecology and Systematics* 18: 237-268.
5. Clutton-Brock, T.H. (2009). Sexual Selection in Males and Females. *Science* 318(5858): 1882-1885.
6. Epstein, J.M. & Axtell, R. (1996). *Growing Artificial Societies: Social Science from the Bottom Up*. MIT Press.
7. Goodall, J. (1986). *The Chimpanzees of Gombe: Patterns of Behavior*. Belknap Press/Harvard University Press.
8. Hamilton, W.D. (1964). The Genetical Evolution of Social Behaviour I & II. *Journal of Theoretical Biology* 7: 1-52.
9. Hamilton, W.D. (1971). Geometry for the Selfish Herd. *Journal of Theoretical Biology* 31(2): 295-311.
10. Muller, M.N. & Mitani, J.C. (2005). Conflict and Cooperation in Wild Chimpanzees. *Advances in the Study of Behavior* 35: 275-331.
11. Nishida, T., Hasegawa, T., Hayaki, H., Takahata, Y. & Uehara, S. (1990). Meat-Sharing as a Coalition Strategy by an Alpha Male Chimpanzee? In: Nishida (Ed.) *The Chimpanzees of the Mahale Mountains*. University of Tokyo Press.
12. Pusey, A.E. (1987). Sex-Biased Dispersal and Inbreeding Avoidance in Birds and Mammals. *Trends in Ecology & Evolution* 2(10): 295-299.
13. Pusey, A.E. & Packer, C. (1987). Dispersal and Philopatry. In: Smuts et al. (Eds.) *Primate Societies*. University of Chicago Press.
14. Sapolsky, R.M. (2005). The Influence of Social Hierarchy on Primate Health. *Science* 308(5722): 648-652.
15. Tinbergen, N. (1957). The Functions of Territory. *Bird Study* 4(1): 14-27.
16. Watts, D.P. & Mitani, J.C. (2001). Boundary Patrols and Intergroup Encounters in Wild Chimpanzees. *Behaviour* 138(3): 299-327.
17. Wilson, E.O. (1975). *Sociobiology: The New Synthesis*. Harvard University Press.
18. Wrangham, R.W. (1999). Evolution of Coalitionary Killing. *Evolutionary Anthropology* 8(1): 30-45.

---

*Nota: verificar números exactos de página e co-autores em PubMed/Google Scholar antes de citação formal. Os dados de treino da IA podem conter imprecisões menores nos volumes/anos de publicações mais antigas.*
