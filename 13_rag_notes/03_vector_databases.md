# Vector Databases

A vector database stores embeddings and lets you search them by similarity. This is what powers retrieval in RAG.

---

## What It Does

```
Index time:
  "Returns accepted within 30 days" → embed → [0.12, -0.44, ...] → store with metadata

Query time:
  "What is the return window?" → embed → [0.11, -0.42, ...] → find nearest vectors → return text
```

Regular databases store and filter exact values. Vector databases find approximate nearest neighbors in high-dimensional space — fast.

---

## pgvector (Start Here)

PostgreSQL extension. If you're already using PostgreSQL, this requires no extra infrastructure.

```sql
-- Enable
CREATE EXTENSION vector;

-- Store
CREATE TABLE docs (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB
);

-- Index (HNSW = fast approximate search)
CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);

-- Search
SELECT content, 1 - (embedding <=> '[0.1, -0.4, ...]'::vector) AS score
FROM docs
ORDER BY embedding <=> '[0.1, -0.4, ...]'::vector
LIMIT 5;
```

**With LangChain:**
```python
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

vs = PGVector(
    connection="postgresql+psycopg://user:pass@localhost/db",
    embeddings=OpenAIEmbeddings(),
    collection_name="documents",
)

# Add
vs.add_texts(["Returns within 30 days.", "Shipping takes 3-5 days."])

# Search
results = vs.similarity_search("return policy", k=3)
```

**When to use:** You're already on PostgreSQL. Single infra. ACID transactions. SQL joins + vector search in one query.  
**Limitation:** Slower than dedicated DBs at 1M+ vectors.

---

## Qdrant

Purpose-built vector DB. Self-hosted or cloud. Best developer experience.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create collection
client.create_collection(
    "docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Upsert
client.upsert("docs", points=[
    PointStruct(id=1, vector=embedding, payload={"text": "...", "source": "manual.pdf"}),
])

# Search with metadata filter
results = client.search(
    "docs",
    query_vector=query_embedding,
    limit=5,
    query_filter={"must": [{"key": "source", "match": {"value": "manual.pdf"}}]},
)
```

**With LangChain:**
```python
from langchain_qdrant import QdrantVectorStore

vs = QdrantVectorStore.from_documents(docs, embedding=OpenAIEmbeddings(), url="http://localhost:6333", collection_name="docs")
```

**When to use:** Self-hosted production. Need hybrid search (dense + sparse built in). Rich filtering.  
**Docker:** `docker run -p 6333:6333 qdrant/qdrant`

---

## Weaviate

Vector DB with GraphQL API and built-in hybrid search.

```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("Document")

# Hybrid search (vector + BM25 in one call)
results = collection.query.hybrid(
    query="return policy",
    alpha=0.5,  # 0 = pure BM25, 1 = pure vector
    limit=3,
)
```

**When to use:** Need multimodal (text + image). GraphQL API. Built-in hybrid is convenient.  
**Docker:** `docker run -p 8080:8080 semitechnologies/weaviate`

---

## Comparison

| | pgvector | Qdrant | Weaviate |
|--|---------|--------|---------|
| Setup | Easy (PG extension) | Docker | Docker |
| Scale | Medium | Large | Large |
| Hybrid search | Manual | Built-in | Built-in |
| SQL joins | Yes | No | No |
| Extra infra | No | Yes | Yes |
| Best for | PG users | Production self-hosted | Multimodal |

---

## HNSW vs IVFFlat (pgvector index choice)

```sql
-- HNSW: faster queries, more memory, good for < 1M vectors
CREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);

-- IVFFlat: less memory, slightly slower, good for large datasets
CREATE INDEX ON docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

For most learning projects: use HNSW.
