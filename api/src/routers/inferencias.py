from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import Inferencia
from src.schemas import InferenciaCreate, InferenciaRead, InferenciaUpdate

router = APIRouter(prefix="/inferencias", tags=["inferencias"])


@router.get("/", response_model=List[InferenciaRead])
def list_inferencias(db: Session = Depends(get_db)):
    return db.query(Inferencia).order_by(Inferencia.criado_em.desc()).limit(100).all()


@router.post("/", response_model=InferenciaRead, status_code=201)
def create_inferencia(payload: InferenciaCreate, db: Session = Depends(get_db)):
    now = datetime.utcnow()
    inferencia = Inferencia(
        inf_id=f"inf_{now.isoformat()}",
        treino_id=payload.treino_id,
        nota=payload.nota or "",
        seed_mundo=payload.seed_mundo,
        fase_mundo=payload.fase_mundo,
    )
    db.add(inferencia)
    db.commit()
    db.refresh(inferencia)
    return inferencia


@router.get("/{inf_id}", response_model=InferenciaRead)
def get_inferencia(inf_id: str, db: Session = Depends(get_db)):
    inferencia = db.query(Inferencia).filter(Inferencia.inf_id == inf_id).first()
    if inferencia is None:
        raise HTTPException(status_code=404, detail="Inferencia not found")
    return inferencia


@router.patch("/{inf_id}", response_model=InferenciaRead)
def update_inferencia(inf_id: str, payload: InferenciaUpdate, db: Session = Depends(get_db)):
    inferencia = db.query(Inferencia).filter(Inferencia.inf_id == inf_id).first()
    if inferencia is None:
        raise HTTPException(status_code=404, detail="Inferencia not found")
    if payload.duracao_steps is not None:
        inferencia.duracao_steps = payload.duracao_steps
    if payload.metricas is not None:
        inferencia.metricas = payload.metricas
    db.commit()
    db.refresh(inferencia)
    return inferencia
