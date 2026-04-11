-- Migration 008: rastreio de nascimentos cross-territoriais e causa detalhada
-- Aplicar: docker cp db/migrations/008_cross_territory.sql snn-db:/tmp/ && docker exec snn-db psql -U snn_user -d snn_organismo -f /tmp/008_cross_territory.sql

-- Adicionar coluna is_cross_territory à tabela organismos_lifetimes
ALTER TABLE organismos_lifetimes
    ADD COLUMN IF NOT EXISTS is_cross_territory BOOLEAN DEFAULT FALSE;

-- Adicionar coluna food_rate (comida_total / idade) — calculada no INSERT para evitar divisão por zero
ALTER TABLE organismos_lifetimes
    ADD COLUMN IF NOT EXISTS food_rate REAL DEFAULT 0.0;

-- Índice para queries cruzadas
CREATE INDEX IF NOT EXISTS idx_ol_cross ON organismos_lifetimes (simulacao_id, is_cross_territory);
CREATE INDEX IF NOT EXISTS idx_ol_food_rate ON organismos_lifetimes (simulacao_id, food_rate DESC);
