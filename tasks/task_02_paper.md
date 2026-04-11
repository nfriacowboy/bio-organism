# Tarefa 2 — Paper Científico

**Estado**: ⏳ AGUARDA (#1 concluído)
**Output**: `documentation/SNN_Paper_Draft_v1.0.md`
**Audiência alvo**: conferências ALife, GECCO, ICLR (workshop Neuro-AI), ou Nature Machine Intelligence

---

## Título provisório

*"A Biologically-Grounded Spiking Neural Network Organism as Foundation for Computational Social Evolution"*

---

## Estrutura do paper

### Secção 1 — Introduction
- [ ] Gap identificado: falta de organismos digitais biologicamente fundamentados para estudar evolução social
- [ ] Proposta: SNN baseado em *C. elegans* como plataforma mínima e validada
- [ ] Preview dos resultados: 3-fase curriculum, olfato navigation emergente, base para fases sociais

### Secção 2 — Related Work
- [ ] Digital twins de *C. elegans* (OpenWorm project)
- [ ] Neuroevolution (NEAT — Stanley & Miikkulainen 2002)
- [ ] ALife organisms (Tierra, Avida, lenia)
- [ ] Multi-agent RL social (MADDPG, hide-and-seek OpenAI)
- [ ] R-STDP em RL (Frémaux & Gerstner 2016)

### Secção 3 — Arquitectura SNN (já documentado)
- [ ] Incorporar `SNN_Justificacao_Arquitectura_v1.0.md`
- [ ] Adicionar subsecção: justificação KAN vs. MLP para política
  - KAN: funções aprendidas (não apenas pesos) → interpretabilidade
  - KAN: composição de funções univariadas → mais fácil de analisar quais inputs importam
  - Referência: Liu et al. (2024) — *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756
- [ ] Adicionar diagrama da arquitectura (42 inputs → SNN → KAN → 3 acções)

### Secção 4 — Correspondência com *C. elegans* biológico

Esta secção é o coração do argumento de validação:

- [ ] **Conectoma**: comparar topologia SNN com conectoma real de *C. elegans*
  - *C. elegans*: 302 neurónios, ~7000 sinapses
  - SNN: 64 neurónios internos, ~2700 pesos activos
  - Referência: White et al. (1986); Cook et al. (2019) — updated connectome
- [ ] **Quimiotaxia**: comportamento de navegação olfactiva real vs. simulado
  - *C. elegans* real: neurónios AWA/AWC detectam diacetil/benzaldeído → viram em gradiente
  - SNN: olfato 1/d² cria gradiente → R-STDP aprende virar em direcção dominante
  - Referência: Bargmann et al. (1993); Pierce-Shimomura et al. (1999)
- [ ] **STDP e dopamina**: aprendizagem por recompensa em nematódeos
  - *C. elegans*: dopamina (neurónios CEP/ADE/PDE) modula aprendizagem associativa
  - SNN: R-STDP usa sinal de recompensa como análogo dopaminérgico
  - Referência: Sanyal et al. (2004) — dopamine modulates C. elegans behavior
- [ ] **Energia e forragem**: modelo de energia vs. comportamento real
  - *C. elegans* real: dispersal quando comida escassa (daf-7/TGF-β pathway)
  - SNN: starvation dispersal em fase 2 (energia_norm < 0.3 → avançar)
  - Referência: Shtonda & Avery (2006) — dietary choice in C. elegans
- [ ] **Curriculum ontogénico**: 3 fases como L1→L2→L3 larval stages
  - *C. elegans* larva L1: ambiente local, recursos abundantes
  - *C. elegans* larva L3+: mundo maior, recursos mais esparsos
  - Paralelismo com fases 1→2→3 do curriculum

### Secção 5 — Resultados

- [ ] **Fase 1**: curva de aprendizagem prec, transição para INF mode
  - Gráfico: prec vs. episódio; cv_c vs. episódio
  - Tabela: eps, pav, prec@ep50/100/150
- [ ] **Fase 2**: survival na density reduzida
  - Gráfico: steps vs. episódio; comida vs. episódio
  - Demonstrar que `threshold_vida_media=800` é atingível
- [ ] **Fase 3**: olfato navigation emergente
  - Gráfico: comida/ep vs. episódio (mostrar crescimento ep150→ep900+)
  - Comparação com random walk baseline (comida=0.9/ep vs. ~15–25/ep)
  - Análise KAN: inputs de olfato têm sensibilidade elevada
  - Burst episodes (eps com comida > 30)
- [ ] **Comparação with baseline**: organismo sem olfato vs. com olfato (1/d²)
  - v80o (count uniform) vs. v80p (1/d²) — demonstrar que gradiente direccional é necessário

### Secção 6 — Limitações

- [ ] SNN inferior a deep RL em benchmarks standard — e porquê isto é esperado/aceitável
- [ ] 80+ versões de treino necessárias — sensibilidade a hiperparâmetros
- [ ] Correspondência com *C. elegans* é qualitativa, não quantitativa
- [ ] Mundo 2D simplificado vs. ambiente 3D real de nematódeo

### Secção 7 — Future Work (Roadmap social)

- [ ] Incorporar sumário de `SNN_Roadmap_Evolucao_Social_v1.0.md`
- [ ] Fases A→D — descrição breve de cada
- [ ] Hipótese experimental da Fase D (ética): Tribo A vs. Tribo B

---

## Fontes adicionais para recolher

- [ ] Cook et al. (2019) — *Whole-animal connectomes of both C. elegans sexes*, Nature (updated connectome)
- [ ] Bargmann et al. (1993) — *Odorant-selective genes and neurons mediate olfaction in C. elegans*, Cell
- [ ] Pierce-Shimomura et al. (1999) — *The fundamental role of pirouettes in C. elegans chemotaxis*, J. Neuroscience (pirouette behavior!)
- [ ] Liu et al. (2024) — *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756
- [ ] OpenWorm Project — openWorm.org (digital C. elegans precedente)

---

## Dados necessários do export v80p

- [ ] Curva comida/ep para todas as fases (gráfico)
- [ ] Tabela de transições de fase (ep, prec, vida_media, modo)
- [ ] KAN sensitivity por input (secção `snn` do export)
- [ ] Distribuição de steps por episódio em fase 3
- [ ] Comparação v80o vs. v80p (dados de ambos os exports)
