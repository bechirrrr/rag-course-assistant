"""Mesure la taille des pages en tokens, avant l'ingestion : python mesurer_pages.py"""
from llama_index.core.utils import get_tokenizer

from ingest import charger_documents  # même lecture + nettoyage que l'ingestion

CHUNK_SIZE = 512  # doit être la même valeur que dans ingest.py

# 1. Lire et nettoyer toutes les pages
documents = charger_documents()

# 2. Compter les tokens de chaque page (même compteur que SentenceSplitter)
tok = get_tokenizer()
mesures = []
for doc in documents:
    nb_tokens = len(tok(doc.text))
    mesures.append((nb_tokens, doc.metadata["source"], doc.metadata["page"]))

tailles = [m[0] for m in mesures]

# 3. Statistiques
print()
print("Pages :", len(tailles))
print(f"Minimum : {min(tailles)}   "
      f"Moyenne : {sum(tailles) / len(tailles):.0f}   "
      f"Maximum : {max(tailles)}")
print("Total :", sum(tailles), "tokens")

# 4. Les 3 pages les plus longues
print("Plus longues :")
for nb_tokens, source, page in sorted(mesures, reverse=True)[:3]:
    print(f"  {nb_tokens:>4}  {source} p.{page}")

# 5. Combien de pages seront découpées en plusieurs chunks ?
trop_longues = [m for m in mesures if m[0] > CHUNK_SIZE]
print(f"Pages > {CHUNK_SIZE} tokens : {len(trop_longues)}")
for nb_tokens, source, page in trop_longues:
    print(f"  {nb_tokens:>4}  {source} p.{page}")
