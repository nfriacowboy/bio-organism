from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, Real, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql.sqltypes import TIMESTAMP

from src.database import Base


class Treino(Base):
    __tablename__ = "treinos"

    id = Column(Text, primary_key=True)
    data_human = Column(Text, nullable=False)
    nota = Column(Text, default="")
    conclusao = Column(Text, default="")
    mlflow_run_id = Column(Text, nullable=True)
    estado = Column(Text, nullable=False, default="pendente")
    parametros = Column(JSONB, nullable=False, default=dict)
    metricas_finais = Column(JSONB, nullable=False, default=dict)
    criado_em = Column(TIMESTAMP(timezone=True), nullable=False, server_default="NOW()")

    inferencias = relationship("Inferencia", back_populates="treino")
    modelos = relationship("Modelo", back_populates="treino")


class Inferencia(Base):
    __tablename__ = "inferencias"

    inf_id = Column(Text, primary_key=True)
    treino_id = Column(Text, ForeignKey("treinos.id"), nullable=False)
    nota = Column(Text, default="")
    seed_mundo = Column(Integer, nullable=False)
    fase_mundo = Column(SmallInteger, nullable=False)
    duracao_steps = Column(Integer, nullable=True)
    metricas = Column(JSONB, nullable=False, default=dict)
    criado_em = Column(TIMESTAMP(timezone=True), nullable=False, server_default="NOW()")

    treino = relationship("Treino", back_populates="inferencias")


class StepLog(Base):
    __tablename__ = "steps_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    treino_id = Column(Text, nullable=False)
    episodio = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    timestamp_ms = Column(BigInteger, nullable=False)
    pos_x = Column(SmallInteger, nullable=False)
    pos_y = Column(SmallInteger, nullable=False)
    angulo = Column(Real, nullable=False)
    energia_actual = Column(Real, nullable=False)
    energia_max = Column(Real, nullable=False)
    visao = Column(JSONB, nullable=False)
    accao = Column(Text, nullable=False)
    modo = Column(Text, nullable=False)
    novidade = Column(Real, nullable=True)
    recompensa = Column(Real, nullable=False)
    bias_explorar = Column(Real, nullable=True)
    bias_exploitar = Column(Real, nullable=True)
    comeu = Column(Boolean, nullable=False, default=False)
    fase_mundo = Column(SmallInteger, nullable=False)


class Modelo(Base):
    __tablename__ = "modelos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    treino_id = Column(Text, ForeignKey("treinos.id"), nullable=False)
    episodio = Column(Integer, nullable=False)
    path_snn = Column(Text, nullable=True)
    checksum_snn = Column(Text, nullable=True)
    path_kan = Column(Text, nullable=True)
    checksum_kan = Column(Text, nullable=True)
    guardado_em = Column(TIMESTAMP(timezone=True), nullable=False, server_default="NOW()")

    treino = relationship("Treino", back_populates="modelos")
