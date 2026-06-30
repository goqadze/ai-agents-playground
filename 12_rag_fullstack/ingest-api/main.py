"""
Ingest API — adds documents to the pgvector knowledge base.

Endpoints:
  POST /ingest/text   — paste raw text and index it
  POST /ingest/file   — upload a .txt file and index it
  GET  /documents     — count indexed chunks
  GET  /health        — readiness probe

The LangGraph RAG graph reads from the same pgvector collection,
so anything ingested here is immediately available for retrieval.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="RAG Ingest API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def _vector_store() -> PGVector:
    return PGVector(
        connection=os.environ["DATABASE_URL"],
        embeddings=_embeddings,
        collection_name="documents",
    )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TextIngestRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"


class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(body: TextIngestRequest):
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    chunks = _splitter.split_text(body.text)
    docs = [
        Document(page_content=chunk, metadata={"source": body.source})
        for chunk in chunks
    ]
    _vector_store().add_documents(docs)
    return IngestResponse(chunks_indexed=len(docs), source=body.source)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded")

    chunks = _splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"source": file.filename})
        for chunk in chunks
    ]
    _vector_store().add_documents(docs)
    return IngestResponse(chunks_indexed=len(docs), source=file.filename)


@app.get("/documents")
def list_documents():
    """Return the total number of chunks stored in the vector store."""
    vs = _vector_store()
    # Use the underlying SQLAlchemy session to count rows
    with vs._make_sync_session() as session:
        collection = vs.get_collection(session)
        if collection is None:
            return {"count": 0}
        from langchain_postgres.vectorstores import EmbeddingStore
        count = (
            session.query(EmbeddingStore)
            .filter(EmbeddingStore.collection_id == collection.uuid)
            .count()
        )
    return {"count": count}


@app.get("/health")
def health():
    return {"status": "ok"}
