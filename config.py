from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)