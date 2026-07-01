# Re-ranking

Re-ranking is a second-pass scoring step that takes your initial retrieval results and reorders them by true relevance. It often gives a bigger quality boost than switching to a better LLM.

---

## The Problem with First-Stage Retrieval

Vector similarity is fast but approximate. It scores each chunk independently against the query using a dot product. It doesn't model the interaction between query and document.

```
Query: "Can I return a damaged item?"

Retrieved (by vector similarity):
  #1 score 0.89 — "Items can be returned within 30 days"  ← doesn't mention damage
  #2 score 0.87 — "Damaged goods policy: damaged items may be returned..."  ← correct!
  #3 score 0.85 — "Return shipping label instructions"

After reranking:
  #1 — "Damaged goods policy: damaged items may be returned..."  ← correctly ranked first
  #2 — "Items can be returned within 30 days"
  #3 — "Return shipping label instructions"
```

---

## How Re-ranking Works

A **cross-encoder** model sees the query and document together in one forward pass — so it models their interaction directly.

```
Bi-encoder (retrieval):          Cross-encoder (reranking):
  query → [vector]                 (query, doc) → relevance score
  doc   → [vector]
  score = dot product              Much more accurate, but slow
  (fast, no interaction)
```

This is why you do it in two stages:
1. Retrieve top-20 fast (bi-encoder)
2. Rerank top-20 accurately (cross-encoder)
3. Keep top-3–5 for the LLM

---

## Pipeline

```
User Question
      │
      ▼
Vector Search (fast)
  → retrieve top 20 chunks
      │
      ▼
Re-ranker (slower, accurate)
  → score each of 20 chunks against query
  → sort by new score
      │
      ▼
Keep top 5
      │
      ▼
LLM
```

---

## Implementation

### Cohere Rerank (API, easiest)
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5,
)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)

# Usage is identical to a normal retriever
docs = retriever.invoke("Can I return a damaged item?")
```

### BGE Reranker (local, free)
```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
reranker = CrossEncoderReranker(model=model, top_n=5)

retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

### Direct API call (Cohere)
```python
import cohere

co = cohere.Client("your-api-key")

results = co.rerank(
    query="Can I return a damaged item?",
    documents=[doc.page_content for doc in retrieved_docs],
    model="rerank-english-v3.0",
    top_n=5,
)

reranked_docs = [retrieved_docs[r.index] for r in results.results]
```

---

## Reranker Options

| Model | Type | Cost | Speed | Quality |
|-------|------|------|-------|---------|
| `cohere rerank-v3.0` | API | Paid | Fast | Excellent |
| `BAAI/bge-reranker-large` | Local | Free | Medium | Very Good |
| `cross-encoder/ms-marco-MiniLM-L12` | Local | Free | Fast | Good |
| `Jina Reranker v2` | API/Local | Free tier | Fast | Good |

---

## How Many to Retrieve vs Keep

```
Retrieve: 20–50 candidates
Rerank:   score all of them
Keep:     top 3–5 for LLM

Why not retrieve 5 directly?
  → You might miss the best chunk if it ranks #6 by vector similarity
  → Reranker finds it in the top-20 pool
```

**Rule of thumb:** Retrieve 4–10x what you'll send to the LLM.

---

## Cost Consideration

- Cohere Rerank: ~$0.001 per 1000 docs reranked (very cheap)
- Local models: free but adds latency (~50–200ms for 20 docs on CPU)

Reranking 20 docs with Cohere costs less than 1/100th of an LLM call. It's almost always worth it.
