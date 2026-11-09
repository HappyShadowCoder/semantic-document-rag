from dotenv import load_dotenv
import os
from .embedder import is_ollama_running

load_dotenv()

AVAILABLE_MODELS = [
    "Auto (Fallback Priority)",
    "OpenAI GPT-4o Mini",
    "OpenAI GPT-4o",
    "Gemini 2.0 Flash",
    "Gemini 1.5 Pro",
    "Ollama llama3.1:8b (Local)"
]

def get_openai_llm(model_name: str, api_key: str):
    from langchain_openai import ChatOpenAI
    m = "gpt-4o" if "GPT-4o" in model_name and "Mini" not in model_name else "gpt-4o-mini"
    llm = ChatOpenAI(model=m, api_key=api_key, temperature=0)
    llm.invoke("ping")
    return llm, f"OpenAI {m}"

def get_gemini_llm(model_name: str, api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    m = "gemini-1.5-pro" if "1.5 Pro" in model_name else "gemini-2.0-flash"
    llm = ChatGoogleGenerativeAI(
        model=m,
        google_api_key=api_key,
        temperature=0
    )
    llm.invoke("ping")
    return llm, f"Gemini {m}"

def get_ollama_llm(model_name: str = "llama3.1:8b"):
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.1:8b")
    except ImportError:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3.1:8b")
    llm.invoke("ping")
    return llm, "Ollama llama3.1:8b (Local)"

def route_to_llm(
    selected_model: str = "Auto (Fallback Priority)",
    custom_openai_key: str = None,
    custom_gemini_key: str = None
):
    """
    Routes to the requested or best available LLM.
    Ollama is completely optional and never compulsory if an API key is provided.
    """
    openai_key = (custom_openai_key or os.getenv("OPENAI_API_KEY") or "").strip()
    gemini_key = (custom_gemini_key or os.getenv("GOOGLE_API_KEY") or "").strip()

    # Explicit Model Selections
    if selected_model and "OpenAI" in selected_model:
        if not openai_key:
            raise ValueError("OpenAI API Key is required for this model. Please enter it in Settings.")
        return get_openai_llm(selected_model, openai_key)

    if selected_model and "Gemini" in selected_model:
        if not gemini_key:
            raise ValueError("Google Gemini API Key is required for this model. Please enter it in Settings.")
        return get_gemini_llm(selected_model, gemini_key)

    if selected_model and "Ollama" in selected_model:
        if not is_ollama_running():
            raise RuntimeError("Ollama is not running on port 11434. Please start Ollama or select OpenAI/Gemini in Settings.")
        return get_ollama_llm()

    # Auto Fallback Chain
    last_error = None
    if openai_key:
        try:
            return get_openai_llm("gpt-4o-mini", openai_key)
        except Exception as e:
            last_error = f"OpenAI error: {e}"

    if gemini_key:
        try:
            return get_gemini_llm("gemini-2.0-flash", gemini_key)
        except Exception as e:
            last_error = f"Gemini error: {e}"

    # Fallback to Ollama only if it is actually running
    if is_ollama_running():
        try:
            return get_ollama_llm()
        except Exception as e:
            last_error = f"Ollama error: {e}"

    if last_error:
        raise RuntimeError(f"Configured provider failed: {last_error}")

    raise RuntimeError(
        "No LLM connected. Please open Settings in the sidebar to enter your OpenAI or Google Gemini API Key."
    )
