# TASKS — bio-organism roadmap
**Última actualização**: 2026-03-27 (Fase C.2 concluída)

---

## Estado geral

| # | Tarefa | Estado | Detalhe |
|---|---|---|---|
| 1 | Validação do organismo base | ✅ CONCLUÍDO | ep1300, 50.9/ep, 11.7× random — ver BASE_VERSION.md |
| 2 | Paper científico | ⏳ AGUARDA | depende de conclusão Fase C |
| 3a | Mundo v2.0 (multi-organismo + patches) | ✅ CONCLUÍDO | N organismos, patches dinâmicos, trainer_populacao.py |
| 3b | Fase A — Reprodução + herança de pesos | ✅ CONCLUÍDO | crossover SNN/KAN + mutação + seleção natural + _renovacao_geracional |
| 3c | Fase B — Envelhecimento + pressão evolutiva | ✅ CONCLUÍDO | vida_max=5000, vel_factor linear, morte garantida, fertilidade cessa 70% |
| 3d | Fase B.2 — Anti-endogamia | ✅ CONCLUÍDO | cosine_sim_snn, INBREEDING_COS_THR=0.95, guarda em _spawn_filho + _renovacao_geracional |
| 3e | Fase C — Feromónios + tribos emergentes | ✅ CONCLUÍDO | feromona_grid, deposit/decay/diffusion, SNN 42→45 inputs, stigmergy |
| 3f | Fase C.2 — Predador | ✅ CONCLUÍDO | Predador persegue org mais próximo, mata por contacto, spawn distante; SNN_USAR_PREDADOR |
| 4 | Fase D — investigação ética emergente | ⏳ AGUARDA | iniciar após Fase C.2 |

---

## Tarefa 1 — Validação do organismo base

Ver: `task_01_validacao_base.md`

**Critério de entrada**: treino v80p completar 2000 episódios
**Critério de saída**: organismo aprovado ou evoluído até versão aceitável

---

## Tarefa 2 — Paper científico

Ver: `task_02_paper.md`

**Critério de entrada**: organismo base validado (#1 concluído)
**Critério de saída**: draft completo do paper

---

## Tarefa 3 — Implementação das fases

Ver: `task_03_implementacao_fases.md`

**Critério de entrada**: organismo base validado (#1 concluído)
**Critério de saída**: Fases A + B + C + C.2 implementadas e com acceptance criteria satisfeitos
**Nota**: Fase D aguarda investigação adicional sobre ética em SNN

---

## Documentação de referência

| Documento | Localização |
|---|---|
| Roadmap social (fases A→D) | `documentation/SNN_Roadmap_Evolucao_Social_v1.0.md` |
| Justificação arquitectura SNN | `documentation/SNN_Justificacao_Arquitectura_v1.0.md` |
| Especificação técnica do organismo | `documentation/SNN_Organismo_Especificacao_Tecnica_v1.0.docx` |
| Evolução dos treinos v25–v79 | `documentation/SNN_Evolucao_Treinos_v25-v79.md` |
