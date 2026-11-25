"""Interface web de l'assistant : streamlit run app.py"""
import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.chroma import ChromaVectorStore

import config  # configure Ollama (LLM + embeddings)
from ask import PROMPT, TOP_K  # même prompt et même top k que ask.py

st.set_page_config(page_title="Assistant de révision", page_icon="📚")


# ---------- 1. Charger l'index une seule fois ----------
@st.cache_resource
def charger_index():
    collection = chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection("courses")
    index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))
    metadonnees = collection.get(include=["metadatas"])["metadatas"]
    cours = sorted({m["course"] for m in metadonnees})
    return index, cours


index, liste_cours = charger_index()


# ---------- 2. Barre latérale : réglages ----------
with st.sidebar:
    st.header("Réglages")
    choix_cours = st.selectbox("Cours", ["Tous les cours"] + liste_cours)
    top_k = st.slider("Nombre de passages donnés au LLM", 1, 10, TOP_K)
    if st.button("Effacer la conversation"):
        st.session_state.historique = []


def creer_moteur(cours, k):
    filtres = None
    if cours != "Tous les cours":
        filtres = MetadataFilters(filters=[MetadataFilter(key="course", value=cours)])
    return index.as_query_engine(similarity_top_k=k, text_qa_template=PROMPT, filters=filtres)


# ---------- 3. Afficher un message (question ou réponse) ----------
def afficher(message):
    with st.chat_message(message["role"]):
        st.markdown(message["texte"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for s in message["sources"]:
                    st.markdown(f"**{s['source']} — page {s['page']}** · score {s['score']:.3f}")
                    st.caption(s["extrait"])


# ---------- 4. Page principale ----------
st.title("📚 Assistant de révision")
st.caption("Les réponses viennent uniquement de tes supports de cours, avec les pages sources.")

if "historique" not in st.session_state:
    st.session_state.historique = []

for message in st.session_state.historique:
    afficher(message)

question = st.chat_input("Pose ta question sur le cours...")
if question:
    message_question = {"role": "user", "texte": question}
    st.session_state.historique.append(message_question)
    afficher(message_question)

    with st.spinner("Recherche dans le cours..."):
        reponse = creer_moteur(choix_cours, top_k).query(question)

    sources = [
        {
            "source": n.metadata["source"],
            "page": n.metadata["page"],
            "score": n.score or 0.0,
            "extrait": n.text[:300].replace("\n", " ") + "...",
        }
        for n in reponse.source_nodes
    ]
    message_reponse = {"role": "assistant", "texte": str(reponse), "sources": sources}
    st.session_state.historique.append(message_reponse)
    afficher(message_reponse)
