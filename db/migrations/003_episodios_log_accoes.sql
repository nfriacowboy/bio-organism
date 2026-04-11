-- Adiciona coluna de distribuição de acções à tabela episodios_log.
--
-- p_avancar: fracção de steps onde a acção foi "avancar".
-- Valores > 0.70 indicam colapso de distribuição (organismo sempre em frente).
ALTER TABLE episodios_log
    ADD COLUMN IF NOT EXISTS p_avancar REAL;
