# Tarefa 1 — Validação do Organismo Base

**Estado**: 🔄 EM CURSO
**Depende de**: treino v80p (em curso, ep ~970/2000)
**Treino activo**: `treino_id = 2026-03-26T12:31:30.554186+00:00`
**Log**: `logs/treino_v80p.log`

---

## 1.1 — Aguardar conclusão do treino v80p

- [ ] treino v80p atingir ep 2000
- [ ] Verificar que processo terminou normalmente (sem crash)
- [ ] Confirmar que export automático correu (`exports/20260326T123130UTC_export.json`)

---

## 1.2 — Análise do export

Após conclusão, analisar o ficheiro JSON de export para validar:

**Performance de navegação (fase 3):**
- [ ] `comida_media` (últimos 100 eps) > 5/ep — demonstra olfato navigation estável
- [ ] `pav_medio` (últimos 100 eps) entre 0.15 e 0.50 — sem colapso advance-all nem pirouette
- [ ] `steps_medio` (últimos 100 eps) > 1000 — organismo sobrevive no mundo esparso

**Aprendizagem (fases 1 e 2):**
- [ ] Transição fase 1→2 ocorreu (prec atingiu threshold=0.40)
- [ ] Transição fase 2→3 ocorreu (vida_media atingiu threshold=800)
- [ ] INF mode activou pelo menos 1× em fase 3 (cv_c atingiu 15 consecutivos)

**Comportamento olfato:**
- [ ] Correlação olfato_frente vs. acção_avancar > 0.2 (verificável nos steps_log)
- [ ] Episódios com comida > 20 ocorreram (bursts de olfato navigation confirmados)
- [ ] Ausência de straight-line charging (pav nunca > 0.75 por >50 eps consecutivos)

**Critério KAN:**
- [ ] KAN sensibilidade — inputs de olfato têm peso relevante (análise export secção `snn`)

---

## 1.3 — Decisão: aprovado ou evoluir?

### Condições de aprovação (PASS)
Organismo serve de base se satisfizer **todos**:
1. comida_media (últ. 100 eps) > 5/ep
2. 3 fases completadas com sucesso
3. Olfato navigation demonstrável (burst eps com comida > 15)
4. pav estável (não colapsa para 0 nem para 1)

### Se falhar — evolução necessária (FAIL)
Identificar qual critério falhou e lançar versão correctiva:

| Critério falhado | Acção |
|---|---|
| comida_media < 5 | v80q — aumentar olfato_bonus ou ajustar raio |
| pav colapsa > 0.75 | v80q — reintroduzir cap dinâmico |
| pav colapsa < 0.10 | v80q — aumentar advance_bonus |
| Fase 3 não atingida | Rever threshold_vida_media fase 2 |

Iterar até PASS.

---

## 1.4 — Documentar versão base

Quando PASS:
- [ ] Guardar checkpoint final como `base_v1.pt` (SNN) e `base_v1_kan.pt` (KAN)
- [ ] Registar `treino_id` da versão base em `tasks/BASE_VERSION.md`
- [ ] Anotar parâmetros definitivos da versão base
- [ ] Exportar relatório final (se não foi exportado automaticamente)
