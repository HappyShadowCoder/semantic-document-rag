from langchain_ollama import OllamaEmbeddings

def get_embedder():
    return OllamaEmbeddings(model="nomic-embed-text")