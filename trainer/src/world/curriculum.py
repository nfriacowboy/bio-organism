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
    threshold_vida_media: float | None    # tempo de vida médio mínimo
    threshold_eficiencia: float | None    # fracção de comida consumida (legado)
    threshold_precisao: float | None      # precisão direcional: P(acção correcta | comida visível)
    min_episodios: int = 0                # mínimo de episódios na fase antes de poder avançar
    min_episodios_inferencia: int = 0
    # v4.0: mínimo de eps CONSECUTIVOS em modo inferência antes de avançar fase.
    # 0 = não requer inferência (compatível com v3 se necessário).


FASES: dict[int, CurriculumConfig] = {
    1: CurriculumConfig(
        fase=1,
        largura=200,
        altura=200,
        num_comida=1500,        # v4.0: 3.75% — era 2000 (5%); reduzido para gradiente olfato mais útil
        num_obstaculos=0,
        reaparecimento_comida=None,
        threshold_vida_media=None,
        threshold_eficiencia=None,
        threshold_precisao=0.40,  # v80m: recalibrado — baseline random+eps≈0.36; 0.40 = genuíno acima random em qualquer regime pav
        min_episodios=100,        # pelo menos 100 eps em fase 1 antes de avançar
        min_episodios_inferencia=5,   # v80k: reduzido de 10 — INF estável 200+ eps, bottleneck é prec não INF count
    ),
    2: CurriculumConfig(
        fase=2,
        largura=500,
        altura=500,
        num_comida=5000,        # v80n: restaurado para 5000 (2% = v65 proven); 3000 (1.2%) insuficiente — precisa 33+ food/ep p/ vida≥1200
        num_obstaculos=0,
        reaparecimento_comida=200,
        threshold_vida_media=800.0,   # v80o: v80n avg fase2 ~860 steps (max 1379); 1200 inatingível sem RECOVERY em fase2 (RECOVERY só fase1). 800 = safely below observed avg.
        threshold_eficiencia=None,
        threshold_precisao=None,      # v64: removido — advance-dominant tem prec=0.10-0.28, 0.80 impossível
        min_episodios=50,             # pelo menos 50 eps em fase 2 antes de avançar
        min_episodios_inferencia=0,   # v80n: 0 — prec=0.05-0.26 em fase2 → CV>0.20 sempre → INF impossível; survival test só
    ),
    3: CurriculumConfig(
        fase=3,
        largura=1000,
        altura=1000,
        num_comida=1500,          # 0.15% — fase de validação; olfato R=100 é ferramenta primária
        num_obstaculos=20,
        reaparecimento_comida=200,
        threshold_vida_media=None,
        threshold_eficiencia=None,
        threshold_precisao=None,
        min_episodios=0,
        min_episodios_inferencia=0,   # fase terminal — sem requisito de inferência
    ),
}


class CurriculumManager:
    """
    Monitoriza o desempenho do organismo e decide quando avançar de fase.
    A transição é validada com base nos últimos JANELA episódios E requer
    um mínimo de episódios na fase actual (min_episodios).

    Critérios de transição (OR entre os thresholds activos):
      threshold_vida_media  — tempo de vida médio ≥ valor
      threshold_eficiencia  — fracção de comida consumida ≥ valor  (legado)
      threshold_precisao    — P(acção correcta | comida visível) ≥ valor
    """

    JANELA = 30   # episódios consecutivos para validar threshold (era 10)

    def __init__(self, fase_inicial: int = 1):
        self.fase_actual = fase_inicial
        self._episodios_na_fase: int = 0
        self._historico_vida: list[float] = []
        self._historico_eficiencia: list[float] = []
        self._historico_precisao: list[float] = []
        self._eps_inferencia_consec: int = 0   # v4.0: eps consecutivos em modo inferência

    def registar_episodio(
        self,
        tempo_vida: int,
        eficiencia: float,
        precisao: float = 0.0,
        em_inferencia: bool = False,   # v4.0: modo inferência neste episódio
    ) -> bool:
        """
        Regista métricas de um episódio.
        Devolve True se a fase avançou.

        Parâmetros
        ----------
        tempo_vida     : número de steps do episódio
        eficiencia     : fracção de comida disponível consumida (0–1)
        precisao       : P(acção correcta | comida visível) no episódio (0–1)
        em_inferencia  : True se este episódio correu em modo inferência (v4.0)
        """
        self._episodios_na_fase += 1
        # v4.0: rastrear episódios consecutivos em modo inferência
        if em_inferencia:
            self._eps_inferencia_consec += 1
        else:
            self._eps_inferencia_consec = 0
        self._historico_vida.append(float(tempo_vida))
        self._historico_eficiencia.append(float(eficiencia))
        self._historico_precisao.append(float(precisao))

        # Manter só os últimos JANELA episódios
        self._historico_vida       = self._historico_vida[-self.JANELA:]
        self._historico_eficiencia = self._historico_eficiencia[-self.JANELA:]
        self._historico_precisao   = self._historico_precisao[-self.JANELA:]

        if len(self._historico_vida) < self.JANELA:
            return False

        cfg = FASES[self.fase_actual]
        proxima_fase = self.fase_actual + 1
        if proxima_fase not in FASES:
            return False   # já na fase máxima

        # Verificar mínimo de episódios na fase actual
        if self._episodios_na_fase < cfg.min_episodios:
            return False

        avancou = False

        if cfg.threshold_vida_media is not None:
            media_vida = sum(self._historico_vida) / len(self._historico_vida)
            if media_vida >= cfg.threshold_vida_media:
                avancou = True

        if cfg.threshold_eficiencia is not None:
            media_efic = sum(self._historico_eficiencia) / len(self._historico_eficiencia)
            if media_efic >= cfg.threshold_eficiencia:
                avancou = True

        if cfg.threshold_precisao is not None:
            media_prec = sum(self._historico_precisao) / len(self._historico_precisao)
            if media_prec >= cfg.threshold_precisao:
                avancou = True

        # v4.0: requisito de inferência — performance atingida mas novidade não colapsou ainda
        if avancou and cfg.min_episodios_inferencia > 0:
            if self._eps_inferencia_consec < cfg.min_episodios_inferencia:
                avancou = False

        if avancou:
            self.fase_actual = proxima_fase
            self._episodios_na_fase = 0
            self._eps_inferencia_consec = 0   # v4.0: reset ao avançar
            self._historico_vida.clear()
            self._historico_eficiencia.clear()
            self._historico_precisao.clear()

        return avancou

    @property
    def config(self) -> CurriculumConfig:
        return FASES[self.fase_actual]
