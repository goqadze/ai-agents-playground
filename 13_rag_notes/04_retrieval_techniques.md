# Retrieval Techniques

Retrieval is the step where you find the right chunks to send to the LLM. Most RAG failures happen here.

---

## 1. Top-K Retrieval

The default. Return the K most similar chunks.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

```
Query → embed → find 5 nearest vectors → return those 5 chunks
```

**Problem:** Always returns K results even if none are actually relevant. If your KB has nothing useful, you still get 5 garbage chunks → LLM hallucinates from them.

---

## 2. Similarity Threshold Retrieval

Only return chunks above a similarity score. Return nothing if no good match exists.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75, "k": 5},
)
```

```
Query → embed → search → score 0.91 ✓, 0.88 ✓, 0.61 ✗ (below 0.75)
→ return only 2 chunks
→ if 0 results: LLM says "I don't have that information"
```

**This is one of the most impactful changes you can make.** Prevents the LLM from answering questions that aren't in your knowledge base.

Good threshold range: **0.70–0.80** depending on your embedding model.

---

## 3. Metadata Filtering

Narrow the search space before running similarity search.

```python
# Only search within a specific document or category
results = vectorstore.similarity_search(
    "return policy",
    filter={"source": "policy_2024.pdf"},
    k=5,
)
```

```python
# Filter by date
results = vectorstore.similarity_search(
    "pricing",
    filter={"year": {"$gte": 2024}},
    k=5,
)
```

**Use cases:**
- Multi-tenant apps (filter by `user_id`)
- Date-sensitive queries (filter by `updated_after`)
- Source-specific queries ("search only in HR docs")

Metadata must be stored at index time:
```python
docs = [Document(page_content="...", metadata={"source": "policy.pdf", "year": 2024})]
vectorstore.add_documents(docs)
```

---

## 4. Multi-Query Retrieval

Generate N variations of the query, retrieve for each, merge results. Improves recall for ambiguous queries.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(),
)
```

```
Original: "How do I reset my password?"

LLM generates 3 variants:
  → "password recovery process"
  → "forgot account credentials"
  → "change login password steps"

Retrieve 5 per variant → deduplicate → union of up to 15 unique chunks
```

**Good for:** Ambiguous or domain-specific language. When users phrase things differently than your docs.  
**Cost:** 1 extra LLM call per query + 3x retrieval calls.

---

## 5. Contextual Compression

After retrieval, compress each chunk to only the part relevant to the query. Reduces noise sent to the LLM.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

compressor = LLMChainExtractor.from_llm(ChatOpenAI())
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)
```

```
Retrieved chunk (500 tokens):
  "Our company was founded in 1995. We sell electronics.
   Returns are accepted within 30 days of purchase date.
   Our CEO is John Smith. We have 200 employees..."

After compression (for query "return window"):
  "Returns are accepted within 30 days of purchase date."
```

**Good for:** Long chunks with mixed content. Reduces token cost for the LLM.  
**Cost:** Extra LLM call per chunk.

---

## Retrieval Quality: Precision vs Recall

```
Precision = of what was retrieved, how much was actually relevant
Recall    = of all relevant content in KB, how much was retrieved

High K (k=20): high recall, low precision  → LLM gets lots of noise
Low K (k=3):   low recall, high precision  → LLM might miss relevant content
```

**Solution:** retrieve high K (20+), then rerank to keep only best 3–5. See `06_reranking.md`.

---

## Quick Decision Guide

| Situation | Technique |
|-----------|-----------|
| Default setup | Top-K (k=5) |
| Prevent hallucination on out-of-scope questions | Similarity threshold |
| Multi-tenant / date-sensitive | Metadata filtering |
| Users phrase things differently from docs | Multi-query |
| Chunks have lots of irrelevant content | Contextual compression |
| Need both precision and recall | Top-K + Reranking |
