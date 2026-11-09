#!/usr/bin/env python3
"""
Terminal CLI for Semantic Document RAG.
Test the full 8-layer AI stack directly from your terminal.
"""
import sys
import os
import argparse
from dotenv import load_dotenv

from rag_pdf.vector_store import build_vectorstore
from rag_pdf.llm_router import route_to_llm
from rag_pdf.prompts import get_rag_prompt
from rag_pdf.guardrails import Guardrails
from rag_pdf.cache import SemanticCache
from rag_pdf.smart_router import SmartRouter
from rag_pdf.memory import ConversationManager
from rag_pdf.observability import QueryTrace, setup_langsmith
from rag_pdf.retriever import retrieve_with_metrics, format_docs_with_sources
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
setup_langsmith()

def main():
    parser = argparse.ArgumentParser(description="Query documents using the 8-layer RAG stack.")
    parser.add_argument("--pdf", type=str, help="Path to PDF document to index")
    parser.add_argument("--persona", type=str, default="Strict Fact-Based", help="Prompt persona")
    parser.add_argument("--multiquery", action="store_true", help="Enable multi-query retrieval expansion")
    args = parser.parse_args()

    print("Initializing Semantic Document RAG Stack...")
    cache = SemanticCache()
    router = SmartRouter()
    memory = ConversationManager(max_turns=5)

    llm, llm_name, _ = router.route_query("ping")
    print(f"Active Model: {llm_name}")

    vectorstore = None
    doc_name = ""
    num_pages = 0

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: File not found: {args.pdf}")
            sys.exit(1)
        print(f"Indexing PDF: {args.pdf}...")
        doc_name = os.path.basename(args.pdf)
        vectorstore, chunks, num_pages = build_vectorstore(args.pdf, doc_name=doc_name)
        print(f"Indexed {num_pages} pages into {chunks} vector chunks in Qdrant.")

    print("\n" + "=" * 60)
    print("Type your questions below. Type 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye.")
            break

        # Layer 06: Guardrails check
        is_safe, error_msg = Guardrails.validate_input(query)
        if not is_safe:
            print(f"\n[Guardrail Blocked]: {error_msg}")
            continue

        # Layer 08: Observability trace
        trace = QueryTrace(query=query, model_name=llm_name)

        # Layer 07: Semantic cache check
        cached_result = cache.get(query, doc_name=doc_name)
        if cached_result:
            answer, score = cached_result
            trace.complete(cached=True)
            print(f"\n[Cache Hit - Similarity: {score:.2f}]:")
            print(answer)
            continue

        if not vectorstore:
            print("\nPlease provide a PDF using `--pdf <file.pdf>` to ask document questions.")
            continue

        # Layer 07: Smart router
        active_llm, active_llm_name, complexity = router.route_query(query)

        # Layer 03: RAG Retrieval
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 25})
        docs, ret_lat, pages = retrieve_with_metrics(
            retriever,
            query,
            llm=active_llm,
            enable_multiquery=args.multiquery
        )
        trace.record_retrieval(latency=ret_lat, count=len(docs), pages=pages)

        # Layer 02: Dynamic prompt
        prompt = get_rag_prompt(persona=args.persona, doc_name=doc_name, total_pages=num_pages)
        chain = prompt | active_llm | StrOutputParser()

        memory.add_user_message(query)
        try:
            formatted_context = format_docs_with_sources(docs)
            response = chain.invoke({
                "context": formatted_context,
                "input": query,
                "chat_history": memory.get_messages()[:-1]
            })

            # Layer 06: Output guardrails
            sanitized = Guardrails.filter_output(response)
            cache.put(query, sanitized, doc_name=doc_name)
            memory.add_ai_message(sanitized)
            trace.complete()

            print(f"\nAssistant ({active_llm_name} [{complexity}]):")
            print(sanitized)
            print(f"\nLatency: {trace.total_latency:.2f}s (Retrieval: {trace.retrieval_latency:.2f}s, Chunks: {len(docs)}, Pages: {pages})")

        except Exception as e:
            trace.complete(error=str(e))
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
