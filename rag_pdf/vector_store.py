from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from .embedder import get_embedder
import pdfplumber
import os

DEFAULT_COLLECTION = "pdf_chunks"
_QDRANT_CLIENT = None

def get_qdrant_client():
    """
    Returns a singleton QdrantClient.
    Priority:
    1. Reuses existing client singleton to prevent file lock conflicts.
    2. Tries Docker Qdrant on localhost:6333.
    3. Falls back to local embedded on-disk storage (./qdrant_data).
    4. Falls back to in-memory mode if on-disk is locked by another process.
    """
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    try:
        client = QdrantClient(host="localhost", port=6333, timeout=1.0)
        client.get_collections()
        _QDRANT_CLIENT = client
        return _QDRANT_CLIENT
    except Exception:
        pass

    storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qdrant_data"))
    os.makedirs(storage_dir, exist_ok=True)
    try:
        _QDRANT_CLIENT = QdrantClient(path=storage_dir)
        return _QDRANT_CLIENT
    except Exception:
        # Fallback to in-memory mode to guarantee zero concurrency lock collisions
        _QDRANT_CLIENT = QdrantClient(location=":memory:")
        return _QDRANT_CLIENT

def load_pdf_with_pdfplumber(pdf_path, doc_name=""):
    from langchain_core.documents import Document
    docs = []
    source_name = doc_name or os.path.basename(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # also extract tables as text
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    text += "\n" + " | ".join(cell or "" for cell in row)
            docs.append(Document(page_content=text, metadata={"page": i+1, "source": source_name}))
    return docs

def build_vectorstore(
    pdf_path: str,
    collection_name: str = DEFAULT_COLLECTION,
    doc_name: str = "",
    openai_key: str = None,
    gemini_key: str = None
):
    embedder = get_embedder(openai_key=openai_key, gemini_key=gemini_key)
    sample_vec = embedder.embed_query("test")
    vector_size = len(sample_vec)

    docs = load_pdf_with_pdfplumber(pdf_path, doc_name=doc_name)

    for doc in docs:
        doc.page_content = doc.page_content.replace("  ", " ")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    client = get_qdrant_client()

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedder,
    )
    vectorstore.add_documents(chunks)
    return vectorstore, len(chunks), len(docs)
