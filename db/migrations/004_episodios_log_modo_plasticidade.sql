-- Migration 004: adicionar coluna modo_plasticidade a episodios_log
-- v4.0 arquitectura: rastrear se cada episódio correu em modo aprendizagem ou inferência.

ALTER TABLE episodios_log
    ADD COLUMN IF NOT EXISTS modo_plasticidade VARCHAR(20) DEFAULT 'aprendizagem';
