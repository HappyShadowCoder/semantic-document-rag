from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama  # ✅ fixed: was `ollama` (lowercase)
from dotenv import load_dotenv
import os

load_dotenv()

def route_to_llm():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if openai_key:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
            llm.invoke("ping")
            print("✅ Using: OpenAI GPT-4o Mini")
            return llm, "OpenAI GPT-4o Mini"
        except Exception as e:
            print(f"⚠️ OpenAI failed: {e}")

    if gemini_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=gemini_key,  # ✅ fixed: was `api_key`, should be `google_api_key`
                temperature=0
            )
            llm.invoke("ping")
            print("✅ Using: Gemini 2.0 Flash")
            return llm, "Gemini 2.0 Flash"
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}")

    try:
        llm = Ollama(model="llama3.1:8b")  # ✅ fixed: was llama3:8b, your model is llama3.1:8b
        llm.invoke("ping")
        print("✅ Using: Ollama llama3.1:8b (local)")
        return llm, "Ollama llama3.1:8b (Local)"
    except Exception as e:
        raise RuntimeError(f"❌ All LLMs failed. Is Ollama running? Error: {e}")