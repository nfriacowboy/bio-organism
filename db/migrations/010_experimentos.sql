-- Migration 010: Sistema de Baseline e Experimentos
-- Rastreabilidade científica: toda a simulação referencia uma baseline
-- ou é marcada como 'undocumented_run' via trigger automático.

-- ============================================================
-- TABELA: baselines
-- Referência nomeada derivada de uma simulação concreta.
-- Serve como ponto de comparação para variações futuras.
-- ============================================================
CREATE TABLE IF NOT EXISTS baselines (
    baseline_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nome             VARCHAR(100) NOT NULL UNIQUE,
    descricao        TEXT,
    tipo             VARCHAR(30) NOT NULL CHECK (tipo IN (
                         'random_walk',        -- floor: agentes sem SNN treinado
                         'fixed_snn',          -- contribuição da selecção: SNN fixo, sem evolução
                         'no_alpha',           -- controlo: sem dinâmica alpha
                         'experiment_control', -- referência para um sweep específico
                         'production'          -- melhor configuração actual
                     )),
    simulacao_id     TEXT REFERENCES simulacoes(simulacao_id) ON DELETE SET NULL,
    config_snapshot  JSONB NOT NULL,   -- snapshot exacto dos parâmetros do mundo
    metricas_referencia JSONB,         -- métricas-chave da simulação baseline
    criado_em        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- TABELA: experimentos
-- Variação controlada de uma baseline, com hipótese e critérios.
-- ============================================================
CREATE TABLE IF NOT EXISTS experimentos (
    experimento_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id          UUID NOT NULL REFERENCES baselines(baseline_id),
    simulacao_id         TEXT REFERENCES simulacoes(simulacao_id) ON DELETE SET NULL,
    nome                 VARCHAR(100),
    -- sweep: se a simulação varia um parâmetro entre mundos
    sweep_variable       VARCHAR(50),   -- ex: 'n_patches'
    sweep_values         JSONB,         -- ex: [2, 3, 4, 6]
    n_replicas_per_value INTEGER,       -- mundos por grupo de sweep
    -- hipótese científica
    parametros_alterados JSONB NOT NULL, -- {"SNN_WORLD_SIZE": {"de": 200, "para": 500}, ...}
    justificacao         TEXT NOT NULL,
    resultado_esperado   TEXT NOT NULL,
    criterios_validacao  JSONB,
    -- ex: [{"metrica": "cv_fc1_media", "op": ">", "valor": 1.0, "janela_final_s": 300}]
    -- resultado
    criterios_atingidos  BOOLEAN,       -- preenchido automaticamente pós-run
    resultado_real       JSONB,         -- métricas reais no fim da simulação
    analise_conclusao    TEXT,          -- análise humana pós-run (opcional)
    -- ciclo de vida
    estado               VARCHAR(20) NOT NULL DEFAULT 'pendente' CHECK (estado IN (
                             'pendente',         -- criado, ainda não iniciado
                             'a_correr',         -- simulação em curso
                             'concluido',        -- simulação terminada, critérios verificados
                             'cancelado',        -- interrompido manualmente
                             'undocumented_run'  -- run sem experimento declarado (via trigger)
                         )),
    criado_em            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    iniciado_em          TIMESTAMPTZ,
    concluido_em         TIMESTAMPTZ
);

-- ============================================================
-- COLUNAS: simulacoes → ligar à camada de rastreabilidade
-- ============================================================
ALTER TABLE simulacoes
    ADD COLUMN IF NOT EXISTS experimento_id UUID REFERENCES experimentos(experimento_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS baseline_id    UUID REFERENCES baselines(baseline_id) ON DELETE SET NULL;

-- ============================================================
-- ÍNDICES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_exp_baseline   ON experimentos(baseline_id);
CREATE INDEX IF NOT EXISTS idx_exp_estado     ON experimentos(estado);
CREATE INDEX IF NOT EXISTS idx_sim_exp        ON simulacoes(experimento_id) WHERE experimento_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sim_baseline   ON simulacoes(baseline_id)    WHERE baseline_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sim_undoc      ON simulacoes(simulacao_id)
    WHERE experimento_id IS NULL AND baseline_id IS NULL;

-- ============================================================
-- TRIGGER: enforcement de disciplina científica
--
-- Se uma simulação não referencia nem baseline nem experimento,
-- o trigger prefija a nota com 'UNDOCUMENTED_' e emite WARNING.
-- Nunca bloqueia (RAISE WARNING, não EXCEPTION).
-- Pode ser promovido a bloqueio mudando para RAISE EXCEPTION.
-- ============================================================
CREATE OR REPLACE FUNCTION enforce_experiment_discipline()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.experimento_id IS NULL AND NEW.baseline_id IS NULL THEN
        IF NEW.nota IS NULL OR NEW.nota NOT LIKE 'UNDOCUMENTED%' THEN
            NEW.nota := 'UNDOCUMENTED_' || COALESCE(NEW.nota, '');
            RAISE WARNING
                'bio-organism: simulação % iniciada sem baseline ou experimento. '
                'Registada como UNDOCUMENTED. '
                'Usar SNN_IS_BASELINE=1 ou SNN_EXPERIMENT_FILE para disciplina científica.',
                NEW.simulacao_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS simulacoes_discipline_check ON simulacoes;
CREATE TRIGGER simulacoes_discipline_check
    BEFORE INSERT ON simulacoes
    FOR EACH ROW EXECUTE FUNCTION enforce_experiment_discipline();
