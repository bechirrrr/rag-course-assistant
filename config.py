import os

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

# Adresse d'Ollama : localhost sur ton PC, "http://ollama:11434" dans Docker
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Dossier de la base Chroma
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")

Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL, request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL)
