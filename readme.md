# Enterprise Semantic Document RAG

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)
![MCP](https://img.shields.io/badge/Protocol-Model_Context_Protocol_(MCP)-8A2BE2?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A hardened, production-grade 8-Layer Enterprise AI Agent Stack for semantic document retrieval, extraction, and synthesis.**

[Architecture](#system-architecture) • [Data Flow](#end-to-end-execution-flow) • [Features](#the-8-layer-production-stack) • [Quickstart](#getting-started) • [MCP Server](#mcp-server-integration) • [Telemetry](#observability--telemetry)

</div>

---

```
  ____                             _   _        ____      _     ____ 
 / ___|  ___ _ __ ___   __ _ _ __ | |_(_) ___  |  _ \    / \   / ___|
 \___ \ / _ \ '_ ` _ \ / _` | '_ \| __| |/ __| | |_) |  / _ \ | |  _ 
  ___) |  __/ | | | | | (_| | | | | |_| | (__  |  _ <  / ___ \| |_| |
 |____/ \___|_| |_| |_|\__,_|_| |_|\__|_|\___| |_| \_\/_/   \_\\____|
```

---

## Overview

**Enterprise Semantic Document RAG** is a resilient document intelligence platform built on top of the complete **8-Layer AI Agent Stack**. It enables high-accuracy question answering, multi-page synthesis, and structural table extraction over complex PDF documents.

Key operational characteristics:
- **Zero-Friction Ingestion:** Documents are parsed with `pdfplumber` (preserving tables & layout) and vectorized immediately upon upload.
- **Provider Agnostic:** Run locally with Ollama or in the cloud with OpenAI or Google Gemini. Ollama is **never required** when an API key is present.
- **Embedded Resilience:** 3-tier vector storage automatically routes between Docker Qdrant, local disk persistence (`./qdrant_data`), and in-memory fallback.
- **Agentic Extensibility:** Ships with an official Model Context Protocol (MCP) server for integration into Claude Desktop, Cursor, or custom multi-agent swarms.
- **Enterprise Hardened:** Integrated input injection defense, output PII redaction, sliding-window rate limiting, and sub-second cosine semantic caching.

---

## System Architecture

The platform is designed around 8 decoupled, testable layers:

```mermaid
flowchart TD
    subgraph UI_Layer["User Experience Layer"]
        User(["Client / User"])
        StreamlitUI["Streamlit Dark UI\n(+ PDF Inline Popover, Chat Bar, Modal Settings)"]
        CLITool["Terminal CLI Interface\n(cli.py)"]
        MCPEndpoint["MCP Server\n(mcp/server.py)"]
    end

    subgraph Layer08["Layer 08 — Observability"]
        LangSmith["LangSmith Distributed Tracing"]
        Telemetry["In-App Query Telemetry Card\n(Latency, Retr. Time, Chunks, Sources)"]
    end

    subgraph Layer07["Layer 07 — Cache & Smart Router"]
        SemanticCache{"Semantic Cache\n(Cosine Sim >= 0.96)"}
        ComplexityRouter["Smart Complexity Router\n(Fast vs Deep Reasoning Path)"]
    end

    subgraph Layer06["Layer 06 — Guardrails"]
        InputGuard["Input Guardrail\n(Prompt Injection & Jailbreak Detector)"]
        OutputGuard["Output Guardrail\n(PII Masking: Email, Phone, SSN)"]
    end

    subgraph Layer05["Layer 05 — Access Control"]
        PasscodeGate["Passcode Gate (APP_ACCESS_KEY)"]
        RateLimiter["Sliding-Window Rate Limiter"]
        TenantIsolation["Session-Isolated Collections"]
    end

    subgraph Layer04["Layer 04 — Tools & Protocols"]
        SearchTool["Tool: search_document"]
        InspectTool["Tool: inspect_page"]
    end

    subgraph Layer03["Layer 03 — RAG & Memory"]
        PDFParser["pdfplumber Table & Text Parser"]
        VectorEngine["Qdrant Vector Store\n(Docker -> Local Disk -> Memory)"]
        MMR["Maximal Marginal Relevance (MMR)"]
        MultiTurnMemory["Sliding-Window Conversation Memory"]
    end

    subgraph Layer02["Layer 02 — Dynamic System Prompts"]
        PersonaSelector["Persona Selector\n(Strict Fact | Executive | Detailed Explainer)"]
        MetaInjector["Dynamic Document Metadata Injection"]
    end

    subgraph Layer01["Layer 01 — Model Router"]
        ModelRouter["LLM Router & Fallback Engine"]
        OpenAICloud["OpenAI (GPT-4o / GPT-4o-mini)"]
        GeminiCloud["Google Gemini (2.0 Flash / 1.5 Pro)"]
        OllamaLocal["Ollama (llama3.1:8b - Optional)"]
    end

    User --> StreamlitUI
    User --> CLITool
    MCPEndpoint -.-> Layer04

    StreamlitUI --> Layer05
    Layer05 --> Layer06
    InputGuard --> Layer07
    
    SemanticCache -- "Cache Hit (sub-second)" --> OutputGuard
    SemanticCache -- "Cache Miss" --> ComplexityRouter

    ComplexityRouter --> Layer03
    Layer03 --> Layer02
    Layer02 --> Layer01

    ModelRouter --> OpenAICloud
    ModelRouter --> GeminiCloud
    ModelRouter --> OllamaLocal

    Layer01 --> OutputGuard
    OutputGuard --> Telemetry
    Telemetry --> LangSmith
    OutputGuard --> StreamlitUI
```

---

## End-to-End Execution Flow

### 1. Document Ingestion Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Operator
    participant UI as Streamlit UI (+ PDF Popover)
    participant Parser as pdfplumber Extractor
    participant Embed as Embedder Factory
    participant DB as Qdrant Vector Store

    User->>UI: Upload Document (.pdf)
    Note over UI: Triggers immediate processing (no manual action needed)
    UI->>Parser: Extract structured text & Markdown tables
    Parser-->>UI: Structured chunks with page metadata
    UI->>Embed: Embed chunks (OpenAI / Gemini / Ollama)
    Embed-->>DB: Index vectors in session collection
    DB-->>UI: Indexing confirmed (collection active)
    UI-->>User: Document loaded & metadata banner displayed
```

### 2. Query Answering & Synthesis Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant Guard as Layer 06: Guardrails
    participant Cache as Layer 07: Semantic Cache
    participant Router as Layer 07: Smart Router
    participant RAG as Layer 03: RAG & MMR Retriever
    participant Prompt as Layer 02: Dynamic Persona
    participant LLM as Layer 01: Model Router
    participant Telemetry as Layer 08: Observability

    User->>Guard: Submit query
    Guard->>Guard: Validate against prompt injection patterns
    alt Injected / Malicious Query
        Guard-->>User: Refusal response (Blocked)
    else Clean Query
        Guard->>Cache: Lookup query embedding
        alt Cache Hit (Similarity >= 0.96)
            Cache-->>Guard: Return cached answer
            Guard->>User: Stream cached result (0.05s latency)
        else Cache Miss
            Cache->>Router: Classify complexity (simple lookup vs analytical)
            Router->>RAG: Retrieve context (MMR diversity search, k=4)
            RAG-->>Prompt: Assemble retrieved context + conversation memory
            Prompt->>LLM: Dispatch tailored prompt to selected model
            LLM-->>Guard: Raw generated answer
            Guard->>Guard: Redact sensitive PII (emails, phones, SSNs)
            Guard->>Cache: Save query + answer pair to semantic cache
            Guard->>Telemetry: Record trace, latency, chunk count, tokens
            Telemetry-->>User: Stream grounded response + Telemetry card
        end
    end
```

---

## The 8-Layer Production Stack

| Layer | Component | Implementation | Production Value |
|---|---|---|---|
| **01** | **Model Router** | `rag_pdf/llm_router.py` | Multi-cloud fallback across OpenAI, Gemini, and Ollama. Dynamic key loading with zero hard dependencies. |
| **02** | **System Prompts** | `rag_pdf/prompts.py` | 3 distinct personas (*Strict Fact-Based*, *Executive Summary*, *Detailed Explainer*) with dynamic document metadata injection. |
| **03** | **RAG & Memory** | `rag_pdf/retriever.py`<br>`rag_pdf/memory.py`<br>`rag_pdf/vector_store.py` | Maximal Marginal Relevance (MMR) retrieval, `pdfplumber` table extraction, and sliding-window conversation memory. |
| **04** | **Tools & MCP** | `rag_pdf/tools.py`<br>`mcp/server.py` | LangChain tools (`search_document`, `inspect_page`) and standard MCP stdio endpoint for external AI agents. |
| **05** | **Access Control** | `rag_pdf/auth.py` | Optional passcode gate, per-session sliding-window rate limiting, and per-user collection isolation. |
| **06** | **Guardrails** | `rag_pdf/guardrails.py` | Regex-based prompt injection and jailbreak detection on inputs; regex-based PII redaction on outputs. |
| **07** | **Cache & Router** | `rag_pdf/cache.py`<br>`rag_pdf/smart_router.py` | Cosine similarity semantic cache delivering sub-second response on duplicate/similar questions; complexity classifier. |
| **08** | **Observability** | `rag_pdf/observability.py` | Native LangSmith distributed tracing and real-time UI telemetry cards detailing response time, retrieval time, and chunk sources. |

---

## LLM State & Routing Logic

```mermaid
stateDiagram-v2
    [*] --> CheckProvider
    
    state CheckProvider {
        [*] --> CheckOpenAI: Has OPENAI_API_KEY?
        CheckOpenAI --> OpenAIReady: Yes
        CheckOpenAI --> CheckGemini: No
        
        CheckGemini --> GeminiReady: Has GOOGLE_API_KEY?
        CheckGemini --> CheckOllama: No
        
        CheckOllama --> OllamaReady: Is port 11434 alive?
        CheckOllama --> NoProvider: No
    }
    
    OpenAIReady --> QueryDispatch
    GeminiReady --> QueryDispatch
    OllamaReady --> QueryDispatch
    NoProvider --> PromptSettingsModal: Prompt user in Settings modal
    
    state QueryDispatch {
        [*] --> AnalyzeComplexity
        AnalyzeComplexity --> FastModel: Low Complexity (Simple Lookup)
        AnalyzeComplexity --> DeepModel: High Complexity (Multi-step Reasoning)
    }
    
    FastModel --> [*]
    DeepModel --> [*]
```

---

## User Interface & Ergonomics

The application interface is styled with a sleek, minimalist dark theme and strictly zero emojis:

- **Inline Upload Popover:** A compact `+ PDF` button is positioned directly adjacent to the chat input bar, keeping document attachment intuitive and immediate.
- **Automatic Ingestion:** Documents parse and vectorize the moment they are selected—no redundant "Process Document" button.
- **Modal Settings Dialog:** Configure OpenAI or Google Gemini API keys and select models inside an overlay dialog without resetting active conversation state.
- **Query Telemetry Cards:** Each assistant response provides an expandable metadata breakdown detailing execution duration, vector search latency, retrieved chunks, and source pages.

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/swastiek/semantic-document-rag.git
cd semantic-document-rag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

You can provide API keys via a `.env` file or enter them directly inside the app's **Settings** modal:

```env
# Cloud Model Providers (provide at least one)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional: Access Gate (Layer 05)
APP_ACCESS_KEY=your_secret_passcode

# Optional: LangSmith Tracing (Layer 08)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=semantic-document-rag
```

### 3. Launch Streamlit Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## MCP Server Integration

This repository includes a standard **Model Context Protocol (MCP)** server (`mcp/server.py`), allowing external agents (e.g. Claude Desktop, Cursor, or LangGraph swarms) to query your indexed documents.

### Claude Desktop Configuration

Add the following to your `claude_desktop_config.json` (located at `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "semantic-rag": {
      "command": "python3",
      "args": [
        "/absolute/path/to/semantic-document-rag/mcp/server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key-here",
        "GOOGLE_API_KEY": "your-key-here"
      }
    }
  }
}
```

The server exposes two tools:
- `search_document(query, k=4)`: Semantically queries the vector database using MMR.
- `inspect_page(page_number)`: Retrieves the raw full text and tables for a given page.

---

## CLI Interface

For headless environments or automated scripting, query documents directly from the terminal:

```bash
# Ingest and query from command line
python3 cli.py --pdf path/to/document.pdf --query "Summarize the primary conclusions"
```

---

## Automated Test Suite

The 8-layer architecture includes a comprehensive unit test suite:

```bash
python3 -m unittest tests/test_stack.py
```

Test coverage includes:
- Multi-provider fallback and lazy import handling
- Dynamic system prompt construction and metadata injection
- Cosine semantic cache insertion and retrieval thresholds
- Prompt injection detection and PII redaction filters
- Sliding-window conversation memory tracking
- Sliding-window rate limiter enforcement
- LangSmith query telemetry encapsulation

---

## Project Structure

```
semantic-document-rag/
├── app.py                         # Streamlit UI & 8-Layer Orchestrator
├── cli.py                         # Terminal CLI query runner
├── requirements.txt               # Project dependencies
├── docker-compose.yml             # Optional Qdrant container definition
├── .gitignore                     # Git exclusion rules (safeguarding data/keys)
├── readme.md                      # Architecture, specifications & guide
├── rag_pdf/
│   ├── __init__.py                # Package facade
│   ├── llm_router.py              # Layer 01: Cloud/local model router
│   ├── prompts.py                 # Layer 02: Personas & dynamic metadata
│   ├── retriever.py               # Layer 03: MMR retrieval & citations
│   ├── memory.py                  # Layer 03: Conversation sliding window
│   ├── vector_store.py            # Layer 03: Qdrant indexing & table parsing
│   ├── tools.py                   # Layer 04: Document search LangChain tools
│   ├── auth.py                    # Layer 05: Passcode, limiter & isolation
│   ├── guardrails.py              # Layer 06: Injection defense & PII mask
│   ├── cache.py                   # Layer 07: Cosine semantic vector cache
│   ├── smart_router.py            # Layer 07: Query complexity classifier
│   ├── observability.py           # Layer 08: LangSmith tracing & telemetry
│   └── embedder.py                # Embedding model factory
├── mcp/
│   ├── server.py                  # Standard MCP stdio server
│   └── claude_desktop_config.json # Integration template for Claude Desktop
└── tests/
    └── test_stack.py              # Automated 9-point unit test suite
```

---

## Security & Privacy Guarantee

- **Zero Data Leakage:** PDFs, vector storage files (`qdrant_data/`), and environment secrets (`.env`) are strictly excluded from version control via `.gitignore`.
- **Session Isolation:** Documents and vector embeddings are stored in isolated collections per user session.
- **Client-Side Storage:** When cloud providers are not configured, local execution with Ollama and local disk Qdrant runs 100% offline.

---

## License

This project is licensed under the [MIT License](LICENSE).
