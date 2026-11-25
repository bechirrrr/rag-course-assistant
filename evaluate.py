"""Évaluation de la recherche : python evaluate.py
Ajoute --llm pour tester aussi les questions hors cours (plus lent)."""
import json
import sys

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

import config  # configure Ollama

K = 5  # on regarde les 5 premiers résultats

# 1. Charger les questions et se connecter à Chroma
with open("eval/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

collection = chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection("courses")
index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))
retriever = index.as_retriever(similarity_top_k=K)


# 2. Trouver le rang de la première bonne page (None si absente du top K)
def rang_premiere_bonne_page(q, resultats):
    for rang, node in enumerate(resultats, start=1):
        m = node.metadata
        if m["source"] == q["source"] and m["page"] in q["pages"]:
            return rang
    return None


# 3. Évaluer la recherche sur les questions qui ont une réponse dans le cours
lignes = []
for q in questions:
    if not q["pages"]:
        continue  # hors cours : pas de bonne page à trouver
    resultats = retriever.retrieve(q["question"])
    rang = rang_premiere_bonne_page(q, resultats)
    trouves = [f"{n.metadata['source'].removesuffix('.pdf')}:p{n.metadata['page']}" for n in resultats]
    lignes.append({"id": q["id"], "type": q["type"], "rang": rang, "trouves": trouves})
    print(f"Q{q['id']:>2} [{q['type']:<13}] rang={str(rang):<4} "
          f"attendu={q['pages']}  trouvé={trouves}")


# 4. Calculer les métriques
def hit_rate(k):
    return sum(1 for l in lignes if l["rang"] and l["rang"] <= k) / len(lignes)


mrr = sum(1 / l["rang"] for l in lignes if l["rang"]) / len(lignes)

print("\n===== RÉSULTATS DE LA RECHERCHE =====")
print(f"Questions évaluées : {len(lignes)}")
for k in (1, 3, 5):
    print(f"Hit rate@{k} : {hit_rate(k):.0%}")
print(f"MRR@{K}      : {mrr:.3f}")

print("\nPar type :")
for t in sorted({l["type"] for l in lignes}):
    groupe = [l for l in lignes if l["type"] == t]
    hits = sum(1 for l in groupe if l["rang"] and l["rang"] <= 3)
    print(f"  {t:<13} hit@3 = {hits}/{len(groupe)}")


# 5. (Option) Tester les questions hors cours avec le LLM
if "--llm" in sys.argv:
    from ask import PROMPT  # réutilise le même prompt que ask.py
    moteur = index.as_query_engine(similarity_top_k=3, text_qa_template=PROMPT)
    print("\n===== QUESTIONS HORS COURS =====")
    for q in questions:
        if q["pages"]:
            continue
        reponse = str(moteur.query(q["question"]))
        avoue = "ne trouve pas" in reponse.lower()
        print(f"Q{q['id']} {'OK  ' if avoue else 'HALLUCINATION'} : {reponse[:150]}")

# 6. Sauvegarder le détail pour comparer plus tard
with open("eval/resultats.json", "w", encoding="utf-8") as f:
    json.dump({"hit@1": hit_rate(1), "hit@3": hit_rate(3), "hit@5": hit_rate(5),
               "mrr": mrr, "details": lignes}, f, ensure_ascii=False, indent=2)
print("\nDétail sauvegardé dans eval/resultats.json")
