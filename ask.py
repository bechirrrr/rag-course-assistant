"""Poser une question au cours : python ask.py "ta question" """
import sys

import chromadb
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

import config  # configure Ollama (LLM + embeddings)

TOP_K = 3  # nombre de passages donnés au LLM

PROMPT = PromptTemplate(
    "Tu es un assistant qui aide un étudiant à réviser son cours.\n"
    "Réponds en français, uniquement à partir des extraits ci-dessous.\n"
    "Si la réponse n'est pas dans les extraits, dis : "
    "« Je ne trouve pas cette information dans le cours. »\n"
    "\n"
    "Extraits du cours :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Question : {query_str}\n"
    "Réponse : "
)

# 1. Se connecter à la base Chroma déjà remplie par ingest.py
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("courses")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# 2. Créer le moteur de questions
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    text_qa_template=PROMPT,
)

# 3. Poser la question
question = " ".join(sys.argv[1:]) or "Qu'est-ce que la lemmatisation ?"
print("Question :", question)
print()

reponse = query_engine.query(question)
print("Réponse :", reponse)
print()

# 4. Montrer les passages utilisés (la "fenêtre" sur le RAG)
print("Sources utilisées :")
for rang, node in enumerate(reponse.source_nodes, start=1):
    m = node.metadata
    print(f"  {rang}. {m['source']} p.{m['page']}  (score {node.score:.3f})")
    print("     ", node.text[:120].replace("\n", " "), "...")
