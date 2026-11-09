from langchain_core.tools import tool
from typing import Optional, Any
from .retriever import format_docs_with_sources

def create_rag_tools(vectorstore: Any):
    """Factory creating LangChain tools linked to the active document vectorstore."""

    @tool
    def search_document(query: str) -> str:
        """Search the indexed PDF document for specific keywords, topics, or excerpts."""
        if not vectorstore:
            return "No document is currently indexed in the vector store."
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "No matching content found for the query."
        return format_docs_with_sources(docs)

    @tool
    def inspect_page(page_number: int) -> str:
        """Inspect all indexed text chunks from a specific page number in the document."""
        if not vectorstore:
            return "No document is currently indexed."
        # Retrieve all chunks belonging to the given page
        client = vectorstore.client
        collection_name = vectorstore.collection_name
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            res = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="metadata.page", match=MatchValue(value=page_number))]
                ),
                limit=20
            )
            points, _ = res
            if not points:
                return f"No text found on page {page_number}."
            texts = [p.payload.get("page_content", "") for p in points if "page_content" in p.payload]
            return f"--- Content from Page {page_number} ---\n\n" + "\n\n".join(texts)
        except Exception as e:
            return f"Unable to inspect page {page_number}: {e}"

    return [search_document, inspect_page]

