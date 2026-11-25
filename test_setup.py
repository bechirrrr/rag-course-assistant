import config
from llama_index.core import Settings

# 1. Test du modèle d'embeddings
vecteur = Settings.embed_model.get_text_embedding("test")
print("Dimension :", len(vecteur))
print("Début du vecteur :", vecteur[:5])

if len(vecteur) == config.EMBED_DIM:
    print("OK")
else:
    print("ERREUR : dimension attendue", config.EMBED_DIM)

# 2. Test du LLM
reponse = Settings.llm.complete("Explique ce qu'est un processus en une phrase.")
print("Réponse du LLM :", reponse)