"""Le cœur du RAG, partagé par l'API (et réutilisable ailleurs)."""
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from ask import PROMPT

_index = None
_collection = None


def charger():
    """Ouvre Chroma une seule fois, puis réutilise l'index."""
    global _index, _collection
    if _index is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _collection = client.get_collection("courses")
        _index = VectorStoreIndex.from_vector_store(
            ChromaVectorStore(chroma_collection=_collection)
        )
    return _index, _collection


def lister_cours():
    _, collection = charger()
    metadonnees = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["course"] for m in metadonnees})


def repondre(question, course=None, top_k=3):
    index, _ = charger()
    filtres = None
    if course:
        filtres = MetadataFilters(filters=[MetadataFilter(key="course", value=course)])
    moteur = index.as_query_engine(
        similarity_top_k=top_k, text_qa_template=PROMPT, filters=filtres
    )
    reponse = moteur.query(question)
    sources = [
        {
            "source": n.metadata["source"],
            "page": n.metadata["page"],
            "course": n.metadata["course"],
            "score": round(n.score or 0.0, 3),
            "extrait": n.text[:300],
        }
        for n in reponse.source_nodes
    ]
    return {"answer": str(reponse), "sources": sources}
