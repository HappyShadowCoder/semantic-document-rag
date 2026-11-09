import streamlit as st
import tempfile
import os
import uuid
import time
from dotenv import load_dotenv

from rag_pdf.vector_store import build_vectorstore
from rag_pdf.llm_router import route_to_llm
from rag_pdf.prompts import get_rag_prompt, PERSONA_PROMPTS
from rag_pdf.guardrails import Guardrails
from rag_pdf.cache import SemanticCache
from rag_pdf.smart_router import SmartRouter
from rag_pdf.memory import ConversationManager
from rag_pdf.auth import AccessController
from rag_pdf.observability import setup_langsmith, QueryTrace
from rag_pdf.retriever import retrieve_with_metrics, format_docs_with_sources
from rag_pdf.tools import create_rag_tools

load_dotenv()
setup_langsmith()

# Reload .env keys into env if present
_env_openai = os.getenv("OPENAI_API_KEY", "")
_env_gemini = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(
    page_title="Semantic Document RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS without emojis
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background-color: #0d0d0d; color: #e8e8e8; }
    .stChatMessage { background-color: #1a1a1a !important; border: 1px solid #2a2a2a; border-radius: 12px; margin-bottom: 8px; }
    .stChatInputContainer { background-color: #1a1a1a !important; border-top: 1px solid #2a2a2a; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
    
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        background: #1b2838;
        color: #66c0f4;
        border: 1px solid #2a475e;
        margin-bottom: 8px;
    }
    
    .guardrail-alert {
        background: #2a1515;
        border-left: 3px solid #d9534f;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 13px;
        color: #f2dede;
        margin-bottom: 8px;
    }
    
    .telemetry-card {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #777;
        background: #141414;
        padding: 6px 10px;
        border-radius: 6px;
        margin-top: 6px;
        border: 1px solid #222;
    }
    
    .history-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #bbb;
        padding: 8px 10px;
        background: #181818;
        border: 1px solid #252525;
        border-radius: 6px;
        margin-bottom: 6px;
        word-break: break-word;
    }

    .doc-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        background: #1a1a1a;
        color: #999;
        border: 1px solid #2e2e2e;
        margin-bottom: 12px;
    }
    
    h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
    .stButton > button {
        background: #222;
        color: #e0e0e0;
        font-weight: 600;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background: #2d2d2d;
        color: #fff;
        border-color: #555;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Layer 05: Access & Session Initialization
# ----------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "auth_controller" not in st.session_state:
    st.session_state.auth_controller = AccessController()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = not st.session_state.auth_controller.is_auth_enabled()

if not st.session_state.authenticated:
    st.markdown("## Access Gate")
    st.markdown("Authentication is required to access this application.")
    passcode = st.text_input("Access Key:", type="password")
    if st.button("Authenticate"):
        if st.session_state.auth_controller.verify_key(passcode):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid access key.")
    st.stop()

# ----------------------------------------------------
# Global Resources (Cache, Router, Memory)
# ----------------------------------------------------
@st.cache_resource
def get_semantic_cache():
    return SemanticCache(similarity_threshold=0.92)

@st.cache_resource
def get_router():
    return SmartRouter()

semantic_cache = get_semantic_cache()
router = get_router()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationManager(max_turns=5)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Auto (Fallback Priority)"

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GOOGLE_API_KEY", "")

# Always sync session keys into os.environ so embedder/LLM can read them
if st.session_state.get("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
if st.session_state.get("gemini_api_key"):
    os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key

if "persona" not in st.session_state:
    st.session_state.persona = "Strict Fact-Based"

if "enable_multiquery" not in st.session_state:
    st.session_state.enable_multiquery = False

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None

# Initialize LLM
if "llm" not in st.session_state:
    try:
        llm, llm_name, _ = router.route_query("init")
        st.session_state.llm = llm
        st.session_state.llm_name = llm_name
    except Exception as e:
        st.session_state.llm = None
        st.session_state.llm_name = "Not Configured"

# ----------------------------------------------------
# Settings Dialog (Modal - does not interrupt running chat)
# ----------------------------------------------------
@st.dialog("Settings")
def open_settings_dialog():
    st.markdown("### System Configuration")
    st.caption("Configure model selection, API credentials, and response persona.")

    st.markdown("#### Model Selection")
    model_options = [
        "Auto (Fallback Priority)",
        "OpenAI GPT-4o Mini",
        "OpenAI GPT-4o",
        "Gemini 2.0 Flash",
        "Gemini 1.5 Pro",
        "Ollama llama3.1:8b (Local)"
    ]
    cur_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    chosen_model = st.selectbox(
        "Active Model:",
        options=model_options,
        index=cur_idx
    )

    st.markdown("#### API Keys")
    input_openai = st.text_input(
        "OpenAI API Key:",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="sk-..."
    )
    input_gemini = st.text_input(
        "Google Gemini API Key:",
        type="password",
        value=st.session_state.gemini_api_key,
        placeholder="AIza..."
    )

    # Immediately track entered keys in session state and environment
    if input_openai and input_openai.strip():
        st.session_state.openai_api_key = input_openai.strip()
        os.environ["OPENAI_API_KEY"] = input_openai.strip()
    if input_gemini and input_gemini.strip():
        st.session_state.gemini_api_key = input_gemini.strip()
        os.environ["GOOGLE_API_KEY"] = input_gemini.strip()

    st.markdown("#### Response Persona")
    new_persona = st.selectbox(
        "Tone and Grounding Rules:",
        options=list(PERSONA_PROMPTS.keys()),
        index=list(PERSONA_PROMPTS.keys()).index(st.session_state.persona)
    )

    st.markdown("#### Retrieval Options")
    new_multiquery = st.checkbox(
        "Enable Multi-Query Expansion",
        value=st.session_state.enable_multiquery,
        help="Expands queries into variations to increase retrieval recall."
    )

    if st.button("Save Changes", use_container_width=True):
        try:
            clean_openai = input_openai.strip()
            clean_gemini = input_gemini.strip()

            router.update_model(
                selected_model=chosen_model,
                openai_key=clean_openai,
                gemini_key=clean_gemini
            )
            st.session_state.selected_model = chosen_model
            st.session_state.openai_api_key = clean_openai
            st.session_state.gemini_api_key = clean_gemini
            st.session_state.persona = new_persona
            st.session_state.enable_multiquery = new_multiquery
            st.session_state.llm = router.primary_llm
            st.session_state.llm_name = router.llm_name

            # Persist to .env so keys survive app reloads
            env_lines = []
            if clean_openai:
                env_lines.append(f"OPENAI_API_KEY={clean_openai}\n")
            if clean_gemini:
                env_lines.append(f"GOOGLE_API_KEY={clean_gemini}\n")
            if env_lines:
                with open(".env", "w") as f:
                    f.writelines(env_lines)

            # Re-trigger processing for any currently uploaded file
            st.session_state.processed_file_id = None

            st.toast("Changes saved")
            st.success(f"Changes saved. Active provider: {router.llm_name}")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save changes: {e}")

    st.markdown("---")
    st.markdown("#### Cache Maintenance")
    if st.button("Clear Semantic Cache", use_container_width=True):
        semantic_cache.clear()
        st.toast("Semantic cache cleared")
        st.success("Semantic cache cleared.")

# ----------------------------------------------------
# Sidebar: ONLY TWO THINGS (Conversation History & Settings)
# ----------------------------------------------------
with st.sidebar:
    # 1. Settings Button
    if st.button("Settings", use_container_width=True):
        open_settings_dialog()

    st.markdown("---")

    # 2. Conversation History
    st.markdown("### Conversation History")
    history_items = st.session_state.memory.get_raw_history()

    if not history_items:
        st.caption("No messages yet.")
    else:
        # Display past user queries as conversation history items
        user_queries = [m["content"] for m in history_items if m["role"] == "user"]
        for idx, q in enumerate(user_queries, 1):
            short_q = q if len(q) <= 45 else q[:42] + "..."
            st.markdown(f'<div class="history-item">{idx}. {short_q}</div>', unsafe_allow_html=True)

        if st.button("Clear History", use_container_width=True):
            st.session_state.memory.clear()
            st.rerun()

# ----------------------------------------------------
# Main Content Area
# ----------------------------------------------------
st.markdown("# Semantic Document RAG")
st.caption("Production document intelligence stack.")

# Active Document Status Banner
if st.session_state.get("doc_name"):
    st.markdown(
        f'<div class="doc-pill">Active Document: <b>{st.session_state.doc_name}</b> | '
        f'{st.session_state.get("num_pages", 0)} pages | '
        f'{st.session_state.get("num_chunks", 0)} chunks indexed</div>',
        unsafe_allow_html=True
    )

# Display Chat Messages
for msg in st.session_state.memory.get_raw_history():
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if not st.session_state.vectorstore:
    st.info("Use the upload button next to the message bar below to select a PDF and begin.")

# ----------------------------------------------------
# Document Upload — sits flush next to the chat input
# using st.chat_input's upload_btn companion layout
# ----------------------------------------------------
# Streamlit renders chat_input at the bottom of the page.
# We place the popover uploader in a slim bar directly above it
# so it visually reads as one unified input row.
# ----------------------------------------------------

# Slim upload row pinned just above the chat bar
with st.container():
    up_col, _ = st.columns([1, 11])
    with up_col:
        with st.popover("+ PDF", use_container_width=True):
            st.markdown("**Upload PDF Document**")
            st.caption("File is indexed immediately on selection.")
            uploaded_file = st.file_uploader(
                "Select PDF",
                type="pdf",
                key="pdf_uploader",
                label_visibility="collapsed"
            )

# Immediate processing logic: triggered right when file is uploaded
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.processed_file_id != file_id:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.getbuffer())
                tmp_path = f.name
            try:
                user_collection = AccessController.get_user_collection_name(st.session_state.session_id)
                vs, num_chunks, num_pages = build_vectorstore(
                    tmp_path,
                    collection_name=user_collection,
                    doc_name=uploaded_file.name,
                    openai_key=st.session_state.get("openai_api_key"),
                    gemini_key=st.session_state.get("gemini_api_key")
                )
                st.session_state.vectorstore = vs
                st.session_state.doc_name = uploaded_file.name
                st.session_state.num_chunks = num_chunks
                st.session_state.num_pages = num_pages
                st.session_state.processed_file_id = file_id
                st.session_state.tools = create_rag_tools(vs)
                semantic_cache.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Chat Input Bar
if user_prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.vectorstore:
        st.warning("Please upload a PDF document before asking questions.")
        st.stop()

    if not st.session_state.get("llm"):
        st.warning("No language model is active. Please open Settings in the sidebar to configure an API key or start Ollama.")
        st.stop()

    # Rate Limiting
    allowed, rate_msg = st.session_state.auth_controller.check_rate_limit(st.session_state.session_id)
    if not allowed:
        st.error(rate_msg)
        st.stop()

    # Observability Trace
    trace = QueryTrace(query=user_prompt, model_name=st.session_state.llm_name)

    # Render User Query
    st.session_state.memory.add_user_message(user_prompt)
    with st.chat_message("user"):
        st.write(user_prompt)

    # Input Guardrails
    is_safe, guardrail_error = Guardrails.validate_input(user_prompt)
    if not is_safe:
        trace.guardrail_blocked = True
        trace.guardrail_reason = guardrail_error
        trace.complete(error=guardrail_error)
        with st.chat_message("assistant"):
            st.markdown(f'<div class="guardrail-alert">[Guardrail Blocked] {guardrail_error}</div>', unsafe_allow_html=True)
            st.session_state.memory.add_ai_message(f"[Blocked]: {guardrail_error}")
        st.stop()

    # Semantic Cache Check
    cached_result = semantic_cache.get(
        user_prompt,
        doc_name=st.session_state.get("doc_name", ""),
        openai_key=st.session_state.get("openai_api_key"),
        gemini_key=st.session_state.get("gemini_api_key")
    )
    if cached_result:
        cached_answer, sim_score = cached_result
        trace.complete(cached=True)

        with st.chat_message("assistant"):
            st.markdown(f'<div class="status-badge">Semantic Cache Hit (Similarity: {sim_score:.2f})</div>', unsafe_allow_html=True)
            st.write(cached_answer)
            st.markdown(
                f'<div class="telemetry-card">Total Latency: {trace.total_latency:.2f}s | Source: Cache</div>',
                unsafe_allow_html=True
            )
            st.session_state.memory.add_ai_message(cached_answer)
        st.stop()

    # Smart Router
    active_llm, active_llm_name, complexity = router.route_query(user_prompt)

    # RAG Retrieval
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25}
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving document context and generating answer..."):
            docs, ret_latency, pages = retrieve_with_metrics(
                retriever,
                user_prompt,
                llm=active_llm,
                enable_multiquery=st.session_state.enable_multiquery
            )
            formatted_context = format_docs_with_sources(docs)
            trace.record_retrieval(latency=ret_latency, count=len(docs), pages=pages)

            # System Prompt
            prompt_template = get_rag_prompt(
                persona=st.session_state.persona,
                doc_name=st.session_state.get("doc_name", ""),
                total_pages=st.session_state.get("num_pages", 0)
            )

            # Chain execution
            from langchain_core.output_parsers import StrOutputParser
            chain = prompt_template | active_llm | StrOutputParser()

            try:
                raw_response = chain.invoke({
                    "context": formatted_context,
                    "input": user_prompt,
                    "chat_history": st.session_state.memory.get_messages()[:-1]
                })

                # Output Guardrails
                sanitized_response = Guardrails.filter_output(raw_response)

                # Cache update
                semantic_cache.put(
                    user_prompt,
                    sanitized_response,
                    doc_name=st.session_state.get("doc_name", ""),
                    openai_key=st.session_state.get("openai_api_key"),
                    gemini_key=st.session_state.get("gemini_api_key")
                )
                trace.complete()

                # Render Response
                st.write(sanitized_response)

                # Telemetry Card (No emojis)
                pages_str = ", ".join(map(str, trace.source_pages)) if trace.source_pages else "N/A"
                st.markdown(
                    f'<div class="telemetry-card">Total Latency: {trace.total_latency:.2f}s | '
                    f'Retrieval Latency: {trace.retrieval_latency:.2f}s | '
                    f'Retrieved Chunks: {trace.retrieved_chunks_count} (Pages: {pages_str}) | '
                    f'Model: {active_llm_name} [{complexity.upper()}]</div>',
                    unsafe_allow_html=True
                )

                st.session_state.memory.add_ai_message(sanitized_response)

            except Exception as e:
                trace.complete(error=str(e))
                err_str = str(e)
                if "Connection refused" in err_str or "11434" in err_str or "Errno 61" in err_str:
                    st.error("Connection Refused: Local Ollama service is not running on port 11434. Please start Ollama (`ollama serve` or launch Ollama), or enter an OpenAI / Gemini API Key in Settings.")
                else:
                    st.error(f"Generation error: {e}")
