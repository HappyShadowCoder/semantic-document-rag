from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from .embedder import get_embedder
import pdfplumber

embedder = get_embedder()

COLLECTION_NAME = "pdf_chunks"
VECTOR_SIZE = len(embedder.embed_query("test"))

def load_pdf_with_pdfplumber(pdf_path):
    from langchain_core.documents import Document
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # also extract tables as text
            for table in page.extract_tables():
                for row in table:
                    text += "\n" + " | ".join(cell or "" for cell in row)
            docs.append(Document(page_content=text, metadata={"page": i+1}))
    return docs

def build_vectorstore(pdf_path: str):
    docs = load_pdf_with_pdfplumber(pdf_path)

    for doc in docs:
        doc.page_content = doc.page_content.replace("  ", " ")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    client = QdrantClient(host="localhost", port=6333)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
    )
    vectorstore.add_documents(chunks)
    return vectorstore, len(chunks)