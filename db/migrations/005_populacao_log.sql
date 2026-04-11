-- Migração 005: tabela de métricas de população (Mundo v2.0)
-- Regista métricas agregadas por episódio para treinos multi-organismo.

CREATE TABLE IF NOT EXISTS episodios_populacao_log (
    id               SERIAL PRIMARY KEY,
    treino_id        TEXT        NOT NULL,
    episodio         INTEGER     NOT NULL,
    fase             INTEGER     NOT NULL,
    n_organismos     INTEGER     NOT NULL,
    comida_media     REAL,
    comida_total     INTEGER,
    comida_max       INTEGER,
    vida_media       REAL,
    vida_max         INTEGER,
    eps_valor        REAL,
    elapsed_s        REAL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_epl_treino_ep
    ON episodios_populacao_log (treino_id, episodio);
