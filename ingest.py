"""Ingestion : PDF -> pages nettoyées -> chunks -> vecteurs -> Chroma."""
import re
from pathlib import Path

import chromadb
import pymupdf
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config  # configure Settings.llm et Settings.embed_model (Ollama)

DOSSIER_PDF = Path("data/raw")
DOSSIER_CHROMA = config.CHROMA_PATH
NOM_COLLECTION = "courses"

LIGNES_A_SUPPRIMER = {
    "IDS 4/IGL 5",                   # TALN : pied de slide
    "Dr. Asma MEKKI",                # TALN : pied de slide
    "Du déclaratif au procédural:",  # T-SQL : titre générique sur 49 slides sur 52
}


# ---------- 1. Nettoyage ----------
def nettoyer(texte):
    lignes_gardees = []
    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if ligne == "":
            continue
        if ligne in LIGNES_A_SUPPRIMER:
            continue
        if re.fullmatch(r"\d{2}/\d{2}/\d{4}", ligne):
            continue
        if re.fullmatch(r"\d+", ligne):
            continue
        lignes_gardees.append(ligne)
    return "\n".join(lignes_gardees)


# ---------- 2. Lecture des PDF : un Document par page ----------
def charger_documents():
    documents = []
    for chemin in sorted(DOSSIER_PDF.rglob("*.pdf")):
        pdf = pymupdf.open(chemin)
        for numero, page in enumerate(pdf, start=1):
            texte = nettoyer(page.get_text())
            if not texte:
                continue  # page vide après nettoyage
            documents.append(
                Document(
                    text=texte,
                    metadata={
                        "source": chemin.name,
                        "page": numero,
                        "course": chemin.parent.name,
                    },
                )
            )
        print(f"  {chemin.name} : {len(pdf)} pages lues")
    return documents


# ---------- 3. Préparation de Chroma ----------
def preparer_chroma():
    client = chromadb.PersistentClient(path=DOSSIER_CHROMA)
    try:
        client.delete_collection(NOM_COLLECTION)  # repartir de zéro, sinon doublons
    except Exception:
        pass  # la collection n'existait pas encore
    collection = client.create_collection(
        NOM_COLLECTION, configuration={"hnsw": {"space": "cosine"}}
    )
    return collection


# ---------- 4. Programme principal ----------
if __name__ == "__main__":
    print("1) Lecture et nettoyage des PDF...")
    documents = charger_documents()
    print(f"   -> {len(documents)} pages gardées")

    print("2) Préparation de Chroma...")
    collection = preparer_chroma()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("3) Découpage + embeddings (bge-m3) + stockage...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"\nTerminé : {collection.count()} chunks dans Chroma.")
    exemple = collection.get(limit=1, include=["documents", "metadatas"])
    print("Exemple de chunk stocké :")
    meta = exemple["metadatas"][0]
    print("  métadonnées :", {k: meta[k] for k in ("source", "page", "course")})
    print("  texte       :", exemple["documents"][0][:200], "...")
