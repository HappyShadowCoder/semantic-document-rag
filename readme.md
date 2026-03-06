# 🧠 RAG Document Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask questions about them. Built with LangChain, Qdrant, and Streamlit — with an intelligent LLM fallback chain.

---

## ✨ Features

- 📄 Upload any PDF and chat with it instantly
- 🔁 Smart LLM fallback: **OpenAI → Gemini → Ollama (local)**
- 🗄️ Persistent vector storage with **Qdrant**
- 🔍 Semantic search using **nomic-embed-text** embeddings (runs locally via Ollama)
- 💬 Clean dark-themed chat UI built with Streamlit
- 📚 Source citations showing which page each answer came from

---

## 🏗️ Project Structure

```
rag/
├── app.py                   # Streamlit UI + chain orchestration
├── rag_pdf/
│   ├── __init__.py
│   ├── llm_router.py        # OpenAI → Gemini → Ollama fallback
│   ├── embedder.py          # nomic-embed-text via Ollama
│   └── vector_store.py      # Qdrant vector DB + PDF ingestion
├── docker-compose.yml       # Qdrant container
├── .env                     # API keys (not committed)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| LLM | OpenAI / Gemini / Ollama llama3.1:8b |
| Embeddings | nomic-embed-text (Ollama) |
| Vector DB | Qdrant (Docker) |
| RAG Framework | LangChain (LCEL) |
| PDF Loader | PyPDFLoader |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- [Docker](https://docker.com) (for Qdrant)
- [Ollama](https://ollama.ai) (for local LLM + embeddings)

### 2. Clone & Install

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Pull Ollama Models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Optional — app works without these using Ollama as fallback
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

### 5. Start Qdrant

```bash
docker compose up -d
```

Qdrant dashboard will be available at `http://localhost:6333/dashboard`

### 6. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔁 LLM Fallback Chain

The app automatically detects which LLMs are available and picks the best one:

```
Start
  │
  ├─ OPENAI_API_KEY set & valid?  ──► ✅ Use OpenAI GPT-3.5
  │
  ├─ GOOGLE_API_KEY set & valid?  ──► ✅ Use Gemini 2.0 Flash
  │
  └─ Ollama running locally?      ──► ✅ Use llama3.1:8b
                                       ❌ Error: check Ollama
```

The active LLM is shown as a badge in the sidebar.

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-openai
langchain-google-genai
langchain-qdrant
langchain-ollama
langchain-text-splitters
qdrant-client
pypdf
python-dotenv
openai
google-generativeai
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 🧠 How It Works

1. **Upload PDF** → parsed and split into overlapping chunks
2. **Chunks embedded** → converted to vectors using `nomic-embed-text`
3. **Stored in Qdrant** → persisted in Docker container
4. **User asks question** → query embedded and top-k similar chunks retrieved
5. **LLM answers** → retrieved chunks passed as context to the active LLM
6. **Answer displayed** → with source page references

---

## 🛠️ Configuration

| Setting | Location | Default |
|---|---|---|
| Chunk size | `vector_store.py` | 2000 |
| Chunk overlap | `vector_store.py` | 400 |
| Top-k retrieval | `app.py` (build_chain) | 20 |
| Embedding model | `embedder.py` | nomic-embed-text |
| Fallback LLM | `llm_router.py` | llama3.1:8b |
| Qdrant port | `docker-compose.yml` | 6333 |

---

## 📌 Roadmap

| Version | Feature | Status |
|---|---|---|
| 1.0.0 | Core RAG + LLM fallback chain | ✅ Released |
| 1.1.0 | MMR Retrieval | 🔧 Planned |
| 1.2.0 | MultiQueryRetriever | 📋 Backlog |
| 1.3.0 | Multi-document support | 📋 Backlog |
| 1.4.0 | Chat memory / conversation history | 📋 Backlog |
| 1.5.0 | Deploy to HuggingFace Spaces | 📋 Backlog |

---

## 📄 License

MIT License