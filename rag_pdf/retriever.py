import time
from typing import List, Tuple, Any
from langchain_core.documents import Document

def format_docs_with_sources(docs: List[Document]) -> str:
    """Formats retrieved document chunks with page metadata for prompt injection."""
    formatted = []
    for doc in docs:
        page_num = doc.metadata.get("page", "N/A")
        doc_source = doc.metadata.get("source", "Doc")
        header = f"[Source: {doc_source} | Page: {page_num}]"
        formatted.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(formatted)

def retrieve_with_metrics(
    retriever: Any,
    query: str,
    llm: Any = None,
    enable_multiquery: bool = False
) -> Tuple[List[Document], float, List[int]]:
    """
    Executes retrieval and tracks retrieval latency and unique pages retrieved.
    If enable_multiquery is True and an LLM is provided, generates multiple query variations
    to improve retrieval recall.
    """
    start_time = time.time()

    all_docs: List[Document] = []
    seen_contents = set()

    queries = [query]
    if enable_multiquery and llm:
        try:
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate(
                input_variables=["question"],
                template=(
                    "Generate 3 different search queries to look up information related to this question.\n"
                    "Provide each query on a new line without numbers or prefixes.\n\n"
                    "Question: {question}\n\n"
                    "Search Queries:"
                )
            )
            chain = prompt | llm
            raw_queries = chain.invoke({"question": query})
            text_queries = raw_queries.content if hasattr(raw_queries, "content") else str(raw_queries)
            for line in text_queries.strip().split("\n"):
                clean = line.strip().lstrip("0123456789.-) ")
                if clean and len(clean) > 5 and clean not in queries:
                    queries.append(clean)
        except Exception:
            pass

    for q in queries:
        try:
            docs = retriever.invoke(q)
            for d in docs:
                if d.page_content not in seen_contents:
                    seen_contents.add(d.page_content)
                    all_docs.append(d)
        except Exception:
            continue

    latency = time.time() - start_time

    pages = []
    for d in all_docs:
        p = d.metadata.get("page")
        if p and p not in pages:
            pages.append(p)

    return all_docs, latency, pages
