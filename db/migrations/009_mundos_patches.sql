-- Migration 009: adicionar coluna patches_json a mundos_config
-- Permite gravar as posições dos patches (cx, cy, radius) para visualização no dashboard.
ALTER TABLE mundos_config
    ADD COLUMN IF NOT EXISTS patches_json JSONB DEFAULT '[]'::jsonb;
