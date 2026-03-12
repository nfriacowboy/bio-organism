-- Tabela de sessões de treino
CREATE TABLE IF NOT EXISTS treinos (
    id              TEXT PRIMARY KEY,               -- timestamp ISO8601
    data_human      TEXT NOT NULL,
    nota            TEXT DEFAULT '',
    conclusao       TEXT DEFAULT '',
    mlflow_run_id   TEXT,
    estado          TEXT NOT NULL DEFAULT 'pendente', -- pendente | a_correr | concluido | erro
    parametros      JSONB NOT NULL DEFAULT '{}',
    metricas_finais JSONB NOT NULL DEFAULT '{}',
    criado_em       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabela de inferências
CREATE TABLE IF NOT EXISTS inferencias (
    inf_id          TEXT PRIMARY KEY,
    treino_id       TEXT NOT NULL REFERENCES treinos(id),
    nota            TEXT DEFAULT '',
    seed_mundo      INTEGER NOT NULL,
    fase_mundo      SMALLINT NOT NULL,
    duracao_steps   INTEGER,
    metricas        JSONB NOT NULL DEFAULT '{}',
    criado_em       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabela de log por step (append-only — sem UPDATE nem DELETE)
CREATE TABLE IF NOT EXISTS steps_log (
    id              BIGSERIAL PRIMARY KEY,
    treino_id       TEXT NOT NULL,
    episodio        INTEGER NOT NULL,
    step            INTEGER NOT NULL,
    timestamp_ms    BIGINT NOT NULL,
    pos_x           SMALLINT NOT NULL,
    pos_y           SMALLINT NOT NULL,
    angulo          REAL NOT NULL,
    energia_actual  REAL NOT NULL,
    energia_max     REAL NOT NULL,
    visao           JSONB NOT NULL,
    accao           TEXT NOT NULL,
    modo            TEXT NOT NULL,              -- aprendizagem | inferencia
    novidade        REAL,
    recompensa      REAL NOT NULL,
    bias_explorar   REAL,
    bias_exploitar  REAL,
    comeu           BOOLEAN NOT NULL DEFAULT FALSE,
    fase_mundo      SMALLINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_treino_episodio ON steps_log(treino_id, episodio);

-- Tabela de modelos guardados
CREATE TABLE IF NOT EXISTS modelos (
    id              SERIAL PRIMARY KEY,
    treino_id       TEXT NOT NULL REFERENCES treinos(id),
    episodio        INTEGER NOT NULL,
    path_snn        TEXT,
    checksum_snn    TEXT,
    path_kan        TEXT,
    checksum_kan    TEXT,
    guardado_em     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
