from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import Modelo, StepLog, Treino
from src.schemas import ModeloRead, StepLogRead, TreinoCreate, TreinoRead, TreinoUpdate

router = APIRouter(prefix="/treinos", tags=["treinos"])


@router.get("/", response_model=List[TreinoRead])
def list_treinos(db: Session = Depends(get_db)):
    return db.query(Treino).order_by(Treino.criado_em.desc()).limit(100).all()


@router.post("/", response_model=TreinoRead, status_code=201)
def create_treino(payload: TreinoCreate, db: Session = Depends(get_db)):
    now = datetime.utcnow()
    treino = Treino(
        id=now.isoformat(),
        data_human=now.strftime("%d/%m/%Y %H:%M:%S"),
        nota=payload.nota or "",
        parametros=payload.parametros,
    )
    db.add(treino)
    db.commit()
    db.refresh(treino)
    return treino


@router.get("/{treino_id}", response_model=TreinoRead)
def get_treino(treino_id: str, db: Session = Depends(get_db)):
    treino = db.query(Treino).filter(Treino.id == treino_id).first()
    if treino is None:
        raise HTTPException(status_code=404, detail="Treino not found")
    return treino


@router.patch("/{treino_id}", response_model=TreinoRead)
def update_treino(treino_id: str, payload: TreinoUpdate, db: Session = Depends(get_db)):
    treino = db.query(Treino).filter(Treino.id == treino_id).first()
    if treino is None:
        raise HTTPException(status_code=404, detail="Treino not found")
    if payload.nota is not None:
        treino.nota = payload.nota
    if payload.conclusao is not None:
        treino.conclusao = payload.conclusao
    if payload.estado is not None:
        treino.estado = payload.estado
    if payload.metricas_finais is not None:
        treino.metricas_finais = payload.metricas_finais
    db.commit()
    db.refresh(treino)
    return treino


@router.get("/{treino_id}/steps", response_model=List[StepLogRead])
def list_steps(
    treino_id: str,
    episodio: Optional[int] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(StepLog).filter(StepLog.treino_id == treino_id)
    if episodio is not None:
        query = query.filter(StepLog.episodio == episodio)
    return query.limit(5000).all()


@router.get("/{treino_id}/modelos", response_model=List[ModeloRead])
def list_modelos(treino_id: str, db: Session = Depends(get_db)):
    return db.query(Modelo).filter(Modelo.treino_id == treino_id).all()
