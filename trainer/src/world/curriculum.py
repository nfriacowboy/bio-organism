"""
Configuração das 3 fases do Curriculum Learning.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumConfig:
    fase: int
    largura: int
    altura: int
    num_comida: int
    num_obstaculos: int
    reaparecimento_comida: int | None  # cada N steps; None=nunca
    # Thresholds para avançar de fase
    threshold_vida_media: float | None    # Fase 1: tempo de vida médio
    threshold_eficiencia: float | None    # Fase 2: eficiência


FASES: dict[int, CurriculumConfig] = {
    1: CurriculumConfig(
        fase=1,
        largura=200,
        altura=200,
        num_comida=50,
        num_obstaculos=0,
        reaparecimento_comida=None,
        threshold_vida_media=500.0,
        threshold_eficiencia=None,
    ),
    2: CurriculumConfig(
        fase=2,
        largura=500,
        altura=500,
        num_comida=80,
        num_obstaculos=0,
        reaparecimento_comida=200,
        threshold_vida_media=None,
        threshold_eficiencia=0.3,
    ),
    3: CurriculumConfig(
        fase=3,
        largura=1000,
        altura=1000,
        num_comida=150,
        num_obstaculos=20,
        reaparecimento_comida=200,
        threshold_vida_media=None,
        threshold_eficiencia=None,   # definido pelo utilizador
    ),
}


class CurriculumManager:
    """
    Monitoriza o desempenho do organismo e decide quando avançar de fase.
    A transição é validada com base nos últimos 10 episódios.
    """

    JANELA = 10   # episódios consecutivos para validar threshold

    def __init__(self, fase_inicial: int = 1):
        self.fase_actual = fase_inicial
        self._historico_vida: list[float] = []
        self._historico_eficiencia: list[float] = []

    def registar_episodio(self, tempo_vida: int, eficiencia: float) -> bool:
        """
        Regista métricas de um episódio.
        Devolve True se a fase avançou.
        """
        self._historico_vida.append(float(tempo_vida))
        self._historico_eficiencia.append(eficiencia)

        # Manter só os últimos JANELA episódios
        self._historico_vida = self._historico_vida[-self.JANELA:]
        self._historico_eficiencia = self._historico_eficiencia[-self.JANELA:]

        if len(self._historico_vida) < self.JANELA:
            return False

        cfg = FASES[self.fase_actual]
        proxima_fase = self.fase_actual + 1
        if proxima_fase not in FASES:
            return False   # já na fase máxima

        avancou = False
        if cfg.threshold_vida_media is not None:
            media_vida = sum(self._historico_vida) / len(self._historico_vida)
            if media_vida >= cfg.threshold_vida_media:
                avancou = True

        if cfg.threshold_eficiencia is not None:
            media_efic = sum(self._historico_eficiencia) / len(self._historico_eficiencia)
            if media_efic >= cfg.threshold_eficiencia:
                avancou = True

        if avancou:
            self.fase_actual = proxima_fase
            self._historico_vida.clear()
            self._historico_eficiencia.clear()

        return avancou

    @property
    def config(self) -> CurriculumConfig:
        return FASES[self.fase_actual]
