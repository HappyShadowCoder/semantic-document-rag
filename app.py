import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from rag_pdf.vector_store import build_vectorstore
from rag_pdf.llm_router import route_to_llm

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background-color: #0d0d0d; color: #e8e8e8; }
    .stChatMessage { background-color: #1a1a1a !important; border: 1px solid #2a2a2a; border-radius: 12px; margin-bottom: 8px; }
    .stChatInputContainer { background-color: #1a1a1a !important; border-top: 1px solid #2a2a2a; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
    .llm-badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 600; background: linear-gradient(135deg, #00ff87, #00c9ff); color: #000; margin-bottom: 12px; }
    .source-card { background: #1e1e1e; border-left: 3px solid #00ff87; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #aaa; }
    h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
    .stButton > button { background: linear-gradient(135deg, #00ff87, #00c9ff); color: #000; font-weight: 700; border: none; border-radius: 8px; font-family: 'Syne', sans-serif; }
    .stButton > button:hover { opacity: 0.85; color: #000; }
    .upload-hint { font-size: 13px; color: #555; font-family: 'JetBrains Mono', monospace; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_llm():
    return route_to_llm()


def build_chain(vectorstore, llm):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document question-answering assistant.

You must answer ONLY using the provided context.

Rules:
- If the answer is present in the context, return it exactly.
- If the context contains lists or tables, return ALL items completely.
- Never use outside knowledge.
- If the answer is not present in the context, reply exactly:

"I don't have enough information in this document."

Do not guess or infer beyond the context.
"""),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])

    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 20, "fetch_k": 50})

    def format_docs(docs):
        return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

st.markdown("# 🧠 RAG Document Chatbot")
st.markdown("Upload a PDF and ask anything about it.")
st.divider()

with st.sidebar:
    st.caption("Version 1.0.0")
    st.markdown("### ⚙️ Setup")

    with st.spinner("Detecting LLM..."):
        try:
            if "llm" not in st.session_state:
                llm, llm_name = load_llm()
                st.session_state.llm = llm
                st.session_state.llm_name = llm_name
            st.markdown(f'<div class="llm-badge">🟢 {st.session_state.llm_name}</div>', unsafe_allow_html=True)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    st.markdown("---")
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Chunking & embedding..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name
                try:
                    vs, num_chunks = build_vectorstore(tmp_path)
                    st.session_state.chain = build_chain(vs, st.session_state.llm)
                    st.session_state.messages = []
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.num_chunks = num_chunks
                    st.success(f"✅ Ready! {num_chunks} chunks indexed.")
                except Exception as e:
                    st.error(f"Failed to process: {e}")
                finally:
                    os.unlink(tmp_path)

    if "doc_name" in st.session_state:
        st.markdown(f"**Active doc:** `{st.session_state.doc_name}`")
        st.markdown(f'<p class="upload-hint">{st.session_state.num_chunks} chunks in Qdrant</p>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**LLM Priority**")
    st.markdown("""<p class="upload-hint">1️⃣ OpenAI GPT-4o Mini<br>2️⃣ Gemini 2.0 Flash<br>3️⃣ Ollama llama3.1:8b</p>""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if "chain" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar and click **Process Document** to get started.")

if prompt := st.chat_input("Ask anything about your document..."):
    if "chain" not in st.session_state:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(prompt)

            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })