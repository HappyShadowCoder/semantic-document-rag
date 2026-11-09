"""
Model Context Protocol (MCP) Server for Semantic Document RAG.
Allows any MCP-compatible agent or client to query indexed documents and retrieve relevant chunks.
"""
import sys
import json
from rag_pdf.vector_store import get_qdrant_client, DEFAULT_COLLECTION
from rag_pdf.embedder import get_embedder

COLLECTION_NAME = DEFAULT_COLLECTION

def handle_request(line: str) -> dict:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON"}

    method = req.get("method")
    req_id = req.get("id")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "search_document",
                        "description": "Search semantically indexed document chunks in Qdrant.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query or concept"}
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        }

    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "search_document":
            query = args.get("query", "")
            try:
                embedder = get_embedder()
                client = get_qdrant_client()
                vec = embedder.embed_query(query)
                results = client.search(collection_name=COLLECTION_NAME, query_vector=vec, limit=5)
                formatted = [
                    f"[Page {p.payload.get('metadata', {}).get('page', 'N/A')}] {p.payload.get('page_content', '')}"
                    for p in results
                ]
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "\n\n".join(formatted) or "No matches."}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(e)}
                }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"}
    }

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle_request(line)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()

