-- Tabela de métricas por episódio (single source of truth)
--
-- Agrega as métricas calculadas pelo trainer a cada episódio.
-- Substitui a dependência do MLflow para análise de aprendizagem.
-- A ligação à BD é feita via treino_id (FK para treinos.id).
CREATE TABLE IF NOT EXISTS episodios_log (
    id                   SERIAL PRIMARY KEY,
    treino_id            TEXT NOT NULL REFERENCES treinos(id),
    episodio             INTEGER NOT NULL,

    -- Desempenho no episódio
    tempo_vida           INTEGER NOT NULL,
    comida               INTEGER NOT NULL DEFAULT 0,
    eficiencia           REAL NOT NULL DEFAULT 0.0,
    recompensa_total     REAL NOT NULL DEFAULT 0.0,

    -- Métricas SNN/AGMP
    taxa_inferencia      REAL NOT NULL DEFAULT 0.0,
    novidade_media       REAL,
    bias_explorar_medio  REAL,
    bias_exploitar_medio REAL,

    -- Curriculum
    fase_curriculum      SMALLINT NOT NULL DEFAULT 1,
    morreu               BOOLEAN NOT NULL DEFAULT FALSE,

    -- Métricas de aprendizagem direcional (v2.0)
    precisao_direcional  REAL,        -- P(acção correcta | comida visível)
    steps_com_visao      INTEGER NOT NULL DEFAULT 0,
    taxa_perseguicao     REAL,        -- P(come | comida curta detectada, janela=20)

    -- Metadados
    seed                 INTEGER,     -- seed do mundo neste episódio (NULL = aleatório)

    UNIQUE (treino_id, episodio)
);

CREATE INDEX IF NOT EXISTS idx_episodios_treino ON episodios_log(treino_id);
