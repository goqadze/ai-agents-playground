"""
RAG as a REST API.

Endpoints:
  POST /ingest   — add text documents to the knowledge base
  POST /query    — ask a question and get a grounded answer
  GET  /documents — list all indexed document IDs

Run:
  python rag_api.py
  # or: uvicorn rag_api:app --reload

Examples:
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d '{"documents": [{"id": "1", "text": "The Eiffel Tower is in Paris."}]}'

  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "Where is the Eiffel Tower?"}'
"""

import os
import chromadb
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="RAG API", description="Minimal RAG with ChromaDB + OpenAI")

# ---------------------------------------------------------------------------
# Shared state — a single in-memory vector store for the whole server lifetime
# ---------------------------------------------------------------------------
_chroma = chromadb.EphemeralClient()
_collection = _chroma.get_or_create_collection(name="knowledge_base")
_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Document(BaseModel):
    id: str
    text: str


class IngestRequest(BaseModel):
    documents: list[Document]


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3  # how many chunks to retrieve


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[str]
    question: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/ingest", summary="Add documents to the knowledge base")
def ingest(request: IngestRequest):
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    _collection.add(
        ids=[doc.id for doc in request.documents],
        documents=[doc.text for doc in request.documents],
    )

    return {
        "message": f"Indexed {len(request.documents)} document(s)",
        "ids": [doc.id for doc in request.documents],
    }


@app.post("/query", response_model=QueryResponse, summary="Ask a question")
def query(request: QueryRequest):
    count = _collection.count()
    if count == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Call /ingest first.")

    # Step 1 — Retrieve
    top_k = min(request.top_k, count)
    results = _collection.query(query_texts=[request.question], n_results=top_k)
    chunks: list[str] = results["documents"][0]

    # Step 2 — Augment + Generate
    context = "\n\n---\n\n".join(chunks)

    response = _openai.chat.completions.create(
        model="gpt-4.1-nano",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY the context "
                    "provided below. If the answer is not in the context, say 'I don't have enough "
                    "information in the knowledge base to answer that.'\n\n"
                    f"CONTEXT:\n{context}"
                ),
            },
            {"role": "user", "content": request.question},
        ],
    )

    answer = response.choices[0].message.content

    return QueryResponse(
        answer=answer,
        retrieved_chunks=chunks,
        question=request.question,
    )


@app.get("/documents", summary="List all document IDs in the knowledge base")
def list_documents():
    all_docs = _collection.get()
    return {
        "count": len(all_docs["ids"]),
        "ids": all_docs["ids"],
    }


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
