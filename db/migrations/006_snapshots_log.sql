-- Migração 006: snapshots de posições dos organismos para visualização tribal
-- Adiciona colunas extra a episodios_populacao_log e cria snapshots_log.

-- Alargar episodios_populacao_log com métricas evolutivas
ALTER TABLE episodios_populacao_log
    ADD COLUMN IF NOT EXISTS geracao_max    INTEGER,
    ADD COLUMN IF NOT EXISTS n_nascimentos  INTEGER,
    ADD COLUMN IF NOT EXISTS cv_fc1         REAL,
    ADD COLUMN IF NOT EXISTS geracao_media  REAL;

-- Tabela de posições periódicas (a cada N episódios) para visualização tribal
CREATE TABLE IF NOT EXISTS snapshots_log (
    id              SERIAL      PRIMARY KEY,
    treino_id       TEXT        NOT NULL,
    episodio        INTEGER     NOT NULL,
    organismo_idx   INTEGER     NOT NULL,
    x               INTEGER,
    y               INTEGER,
    angulo          REAL,
    geracao         INTEGER,
    idade           INTEGER,
    comida_lifetime INTEGER,
    vivo            BOOLEAN     DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_treino_ep
    ON snapshots_log (treino_id, episodio);
