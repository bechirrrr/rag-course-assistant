"""Tests de l'API, sans Ollama ni Chroma : le RAG est remplacé par une fausse version."""
import pytest
from fastapi.testclient import TestClient

import api
import rag


@pytest.fixture
def client(monkeypatch):
    """Un client de test où rag.py est simulé (réponses rapides et prévisibles)."""
    monkeypatch.setattr(rag, "lister_cours", lambda: ["nlp", "tsql"])

    def faux_repondre(question, course=None, top_k=3):
        source = {"source": "tsql_1_intro.pdf", "page": 30, "course": course or "tsql",
                  "score": 0.74, "extrait": "Un curseur permet de parcourir..."}
        return {"answer": f"Réponse à : {question}", "sources": [source] * top_k}

    monkeypatch.setattr(rag, "repondre", faux_repondre)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_courses(client):
    r = client.get("/courses")
    assert r.status_code == 200
    assert r.json() == ["nlp", "tsql"]


def test_ask_reponse_et_sources(client):
    r = client.post("/ask", json={"question": "Qu'est-ce qu'un curseur ?"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Réponse à : Qu'est-ce qu'un curseur ?"
    assert len(data["sources"]) == 3  # top_k par défaut
    assert set(data["sources"][0]) == {"source", "page", "course", "score", "extrait"}


def test_ask_transmet_cours_et_top_k(client):
    r = client.post("/ask", json={"question": "Qu'est-ce qu'une vue ?", "course": "tsql", "top_k": 5})
    assert r.status_code == 200
    assert len(r.json()["sources"]) == 5
    assert all(s["course"] == "tsql" for s in r.json()["sources"])


def test_ask_cours_inconnu(client):
    r = client.post("/ask", json={"question": "Bonjour ?", "course": "maths"})
    assert r.status_code == 404
    assert "maths" in r.json()["detail"]


@pytest.mark.parametrize("corps", [
    {},                                        # question manquante
    {"question": "ab"},                        # trop courte
    {"question": "Bonjour ?", "top_k": 0},     # top_k trop petit
    {"question": "Bonjour ?", "top_k": 11},    # top_k trop grand
    {"question": "Bonjour ?", "top_k": "trois"},  # mauvais type
])
def test_ask_requetes_invalides(client, corps):
    assert client.post("/ask", json=corps).status_code == 422
