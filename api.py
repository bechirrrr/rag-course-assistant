"""API du RAG : uvicorn api:app --reload   puis http://localhost:8000/docs"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rag

app = FastAPI(
    title="RAG Course Assistant API",
    description="Pose des questions sur tes supports de cours, avec les pages sources.",
    version="1.0.0",
)


# ---------- Modèles de données (validation automatique) ----------
class Question(BaseModel):
    question: str = Field(..., min_length=3, examples=["Qu'est-ce qu'un curseur ?"])
    course: str | None = Field(None, examples=["tsql"])
    top_k: int = Field(3, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: int
    course: str
    score: float
    extrait: str


class Reponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------- Routes ----------
@app.get("/health")
def health():
    """Vérifie que l'API tourne."""
    return {"status": "ok"}


@app.get("/courses", response_model=list[str])
def courses():
    """Liste les cours indexés."""
    return rag.lister_cours()


@app.post("/ask", response_model=Reponse)
def ask(q: Question):
    """Répond à une question à partir des cours."""
    if q.course and q.course not in rag.lister_cours():
        raise HTTPException(status_code=404, detail=f"Cours inconnu : {q.course}")
    return rag.repondre(q.question, q.course, q.top_k)
