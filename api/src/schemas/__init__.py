from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class TreinoCreate(BaseModel):
    nota: Optional[str] = ""
    parametros: dict[str, Any] = {}


class TreinoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    data_human: str
    nota: Optional[str]
    conclusao: Optional[str]
    mlflow_run_id: Optional[str]
    estado: str
    parametros: dict[str, Any]
    metricas_finais: dict[str, Any]
    criado_em: datetime


class TreinoUpdate(BaseModel):
    nota: Optional[str] = None
    conclusao: Optional[str] = None
    estado: Optional[str] = None
    metricas_finais: Optional[dict[str, Any]] = None


class InferenciaCreate(BaseModel):
    treino_id: str
    seed_mundo: int
    fase_mundo: int = 1
    nota: Optional[str] = ""


class InferenciaRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    inf_id: str
    treino_id: str
    nota: Optional[str]
    seed_mundo: int
    fase_mundo: int
    duracao_steps: Optional[int]
    metricas: dict[str, Any]
    criado_em: datetime


class InferenciaUpdate(BaseModel):
    duracao_steps: Optional[int] = None
    metricas: Optional[dict[str, Any]] = None


class StepLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    treino_id: str
    episodio: int
    step: int
    timestamp_ms: int
    pos_x: int
    pos_y: int
    angulo: float
    energia_actual: float
    energia_max: float
    visao: Any
    accao: str
    modo: str
    novidade: Optional[float]
    recompensa: float
    bias_explorar: Optional[float]
    bias_exploitar: Optional[float]
    comeu: bool
    fase_mundo: int


class ModeloRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    treino_id: str
    episodio: int
    path_snn: Optional[str]
    checksum_snn: Optional[str]
    path_kan: Optional[str]
    checksum_kan: Optional[str]
    guardado_em: datetime
