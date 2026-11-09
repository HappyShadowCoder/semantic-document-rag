import os
import socket

def is_ollama_running(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def get_embedder(openai_key: str = None, gemini_key: str = None):
    """
    Returns the appropriate embeddings model based on available keys or local Ollama.
    Ollama is never mandatory if an OpenAI or Gemini API key is provided.
    """
    key_openai = (openai_key or os.getenv("OPENAI_API_KEY") or "").strip()
    key_gemini = (gemini_key or os.getenv("GOOGLE_API_KEY") or "").strip()

    # 1. Explicit or detected OpenAI Embeddings
    if key_openai:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=key_openai)
        except Exception as e:
            raise RuntimeError(f"OpenAI Embeddings Error: {e}")

    # 2. Explicit or detected Gemini Embeddings
    if key_gemini:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            # Prefer text-embedding-004, fallback to embedding-001
            try:
                return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key_gemini)
            except Exception:
                return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key_gemini)
        except Exception as e:
            raise RuntimeError(f"Gemini Embeddings Error: {e}")

    # 3. Ollama (Local) only if running
    if is_ollama_running():
        try:
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model="nomic-embed-text")
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(model="nomic-embed-text")

    # 4. Neither API key was supplied and Ollama is not running
    raise RuntimeError(
        "No LLM or embedding provider connected. Please open Settings in the sidebar to enter your OpenAI or Google Gemini API Key."
    )
