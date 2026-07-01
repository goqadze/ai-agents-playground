# Hybrid Search

Hybrid search combines **vector search** (semantic) with **keyword search** (BM25). Most production RAG systems use it because each method catches what the other misses.

---

## Why Pure Vector Search Isn't Enough

```
Query: "OpenAI GPT-4o pricing"

Vector search finds:  "AI model cost comparison for enterprise" ✓ (semantically close)
                      "LLM pricing guide 2024" ✓
                      Misses: doc with exact string "GPT-4o" but different phrasing

BM25 finds:           Exact matches for "GPT-4o" and "pricing" ✓
                      Misses: doc that says "latest OpenAI model costs" (no exact match)

Hybrid:               Gets both → better recall
```

**Vector search** is great for: meaning-based queries, paraphrasing, concepts.  
**BM25** is great for: exact terms, product names, codes, abbreviations, proper nouns.

---

## How BM25 Works (Brief)

BM25 is a keyword scoring algorithm. It ranks documents by term frequency, adjusted for document length.

```
Query: "password reset"
Doc A: "password reset link" → high score (both terms, short doc)
Doc B: "security password management system reset procedures" → lower score (terms diluted)
Doc C: "account recovery" → 0 score (neither term appears)
```

No embeddings involved — purely lexical matching.

---

## Hybrid Score

Combine both scores with a weighting factor (alpha):

```
hybrid_score = alpha × vector_score + (1 - alpha) × bm25_score

alpha = 1.0  → pure vector search
alpha = 0.0  → pure BM25
alpha = 0.5  → equal weight (common default)
```

Tune alpha based on your use case:
- Lots of exact product names/codes → lower alpha (more BM25)
- Conceptual/general queries → higher alpha (more vector)

---

## Implementation Options

### Qdrant (built-in hybrid)
```python
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVectorParams, SparseIndexParams

# Collection needs both dense and sparse vectors
client.create_collection(
    "docs",
    vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
)

# Query — Qdrant handles fusion internally
results = client.query_points(
    "docs",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(query=SparseVector(indices=[12, 45], values=[0.4, 0.8]), using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
    limit=5,
)
```

### pgvector + pg_trgm (PostgreSQL)
```sql
-- Enable both extensions
CREATE EXTENSION vector;
CREATE EXTENSION pg_trgm;

-- Search combining both
SELECT content,
    (0.5 * (1 - (embedding <=> $1))) +
    (0.5 * similarity(content, $2)) AS hybrid_score
FROM docs
ORDER BY hybrid_score DESC
LIMIT 5;
```

### LangChain + EnsembleRetriever (any vector DB)
```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.vectorstores import Chroma

# BM25 retriever (keyword)
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# Vector retriever (semantic)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine with Reciprocal Rank Fusion
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("OpenAI GPT-4o pricing")
```

---

## Reciprocal Rank Fusion (RRF)

Instead of combining raw scores (which are on different scales), RRF merges ranked lists:

```
Vector results:   [A(#1), C(#2), B(#3), D(#4)]
BM25 results:     [B(#1), A(#2), E(#3), C(#4)]

RRF score = Σ 1/(rank + k)   where k=60 (constant)

A: 1/61 + 1/62 = 0.0325  ← highest
B: 1/63 + 1/61 = 0.0321
C: 1/62 + 1/64 = 0.0317
```

RRF is more robust than score-averaging because it doesn't require normalizing scores from different systems.

---

## When Hybrid Search Matters Most

| Content type | Why hybrid helps |
|-------------|-----------------|
| Product catalogs | Exact SKU/model names (BM25) + description similarity (vector) |
| Legal/medical docs | Exact terms like "Section 4.2(b)" or drug names (BM25) |
| Code documentation | Function names, error codes (BM25) + concept search (vector) |
| Customer support | Common phrases + semantic variants of the same question |

**General rule:** If your documents contain any proper nouns, codes, or technical terms that users might query exactly → use hybrid search.
