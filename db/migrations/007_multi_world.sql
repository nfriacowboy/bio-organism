-- Migração 007: tabelas para simulação multi-mundo
-- Permite correr N mundos em paralelo com parâmetros variados (ex: density sweep)
-- e recolher dados ricos por organismo para análise tribal posterior.

-- Metadata de cada run de simulação multi-mundo
CREATE TABLE IF NOT EXISTS simulacoes (
    simulacao_id    TEXT        PRIMARY KEY,
    nota            TEXT,
    n_mundos        INTEGER,
    config_json     JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Configuração por mundo dentro de uma simulação
CREATE TABLE IF NOT EXISTS mundos_config (
    id              SERIAL      PRIMARY KEY,
    simulacao_id    TEXT        NOT NULL,
    mundo_id        INTEGER     NOT NULL,
    world_size      INTEGER,
    n_food          INTEGER,
    food_pct        REAL,       -- n_food / (world_size^2)
    n_org_inicial   INTEGER,
    vida_max        INTEGER,
    max_pop         INTEGER,
    seed            INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (simulacao_id, mundo_id)
);

CREATE INDEX IF NOT EXISTS idx_mc_sim ON mundos_config (simulacao_id);

-- Lifetime completo por organismo (nasce e morre)
CREATE TABLE IF NOT EXISTS organismos_lifetimes (
    id              SERIAL      PRIMARY KEY,
    simulacao_id    TEXT        NOT NULL,
    mundo_id        INTEGER     NOT NULL,
    org_id          INTEGER     NOT NULL,
    geracao         INTEGER,
    x_nasc          INTEGER,
    y_nasc          INTEGER,
    x_morte         INTEGER,
    y_morte         INTEGER,
    idade           INTEGER,
    comida_total    INTEGER,
    nasc_em_s       REAL,       -- segundos desde início da simulação
    mort_em_s       REAL,
    causa_morte     TEXT,       -- 'fome' | 'senescencia'
    n_filhos        INTEGER     DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ol_sim_mundo   ON organismos_lifetimes (simulacao_id, mundo_id);
CREATE INDEX IF NOT EXISTS idx_ol_geracao     ON organismos_lifetimes (simulacao_id, geracao);

-- Métricas periódicas por mundo (time-series)
CREATE TABLE IF NOT EXISTS mundos_log (
    id              SERIAL      PRIMARY KEY,
    simulacao_id    TEXT        NOT NULL,
    mundo_id        INTEGER     NOT NULL,
    t_s             REAL,       -- segundos desde início da simulação
    step            INTEGER,
    n_vivos         INTEGER,
    comida_rate_s   REAL,       -- comidas/segundo no último intervalo
    gen_max         INTEGER,
    gen_media       REAL,
    cv_fc1          REAL,       -- diversidade genética
    n_nascimentos   INTEGER,    -- nascimentos desde último log
    n_mortes        INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_sim_mundo ON mundos_log (simulacao_id, mundo_id);
CREATE INDEX IF NOT EXISTS idx_ml_t         ON mundos_log (simulacao_id, t_s);

-- Snapshots posicionais para mapa tribal (DBSCAN offline)
CREATE TABLE IF NOT EXISTS posicoes_snapshot (
    id              SERIAL      PRIMARY KEY,
    simulacao_id    TEXT        NOT NULL,
    mundo_id        INTEGER     NOT NULL,
    t_s             REAL,
    step            INTEGER,
    org_id          INTEGER,
    x               INTEGER,
    y               INTEGER,
    geracao         INTEGER,
    energia_norm    REAL,
    comida_acc      INTEGER,    -- comida acumulada neste lifetime
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ps_sim_mundo_t ON posicoes_snapshot (simulacao_id, mundo_id, t_s);
