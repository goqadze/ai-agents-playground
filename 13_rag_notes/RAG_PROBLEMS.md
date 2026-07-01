# RAG Problems, Failure Modes & What to Learn

## The Core Problem Map

```
User Question
     │
     ├── [Retrieval Fails]  ──► wrong or no chunks returned
     │         │
     │         ├── Semantic mismatch (query ≠ chunk wording)
     │         ├── Chunk too large (relevant part diluted)
     │         ├── Chunk too small (missing context)
     │         ├── Wrong top-k (relevant chunk ranked 10th)
     │         └── Knowledge gap (document never ingested)
     │
     ├── [Context Fails]    ──► right chunks, wrong answer
     │         │
     │         ├── Lost in the middle (LLM ignores middle chunks)
     │         ├── Context window overflow (too many chunks)
     │         ├── Conflicting chunks (different docs disagree)
     │         └── Stale data (document is outdated)
     │
     └── [Generation Fails] ──► good context, bad output
               │
               ├── Hallucination (LLM ignores context anyway)
               ├── Unfaithful summary (distorts chunk meaning)
               ├── Over-refusal ("I don't know" when answer exists)
               └── Verbose / irrelevant answer
```

---

## Problem 1 — Chunking Issues

### Too Large
Chunk contains the answer + irrelevant noise. The embedding averages both, so the vector drifts away from the query.

```
Query: "What is the return policy?"

Chunk (800 tokens):
  "...the company was founded in 1995. Our CEO is John Smith.
   The return policy allows 30 days... We also sell gift cards..."

Embedding = average of all topics → poor match to "return policy"
```

### Too Small
Each chunk lacks enough context to be meaningful.

```
Chunk: "The returns are 30 days."

LLM gets no info about: from what date? what condition? what format?
```

### Cut at Wrong Boundary
Splitting mid-sentence or mid-paragraph destroys meaning.

```
Chunk 1: "You can return items if they meet the following"
Chunk 2: "conditions: unopened, within 30 days, with receipt."

→ Chunk 1 alone is useless. Chunk 2 alone loses the subject.
```

**What to learn:** Semantic chunking, sentence-window chunking, parent-child chunking, RAPTOR.

---

## Problem 2 — Retrieval Failures

### Vocabulary Mismatch
User says "car" but document says "automobile". Embeddings help but don't fully solve this.

```
Query:    "How do I fix my automobile?"  → vector A
Document: "Car repair guide"             → vector B

Cosine(A, B) = 0.71  (good)
vs
Document: "Vehicle maintenance"          → vector C
Cosine(A, C) = 0.69  (also good, but different document)
```

### Wrong Distance Metric
Using dot product when vectors aren't normalized → wrong rankings.

### Low Top-k Recall
The correct chunk is #8 but you only retrieve top-3.

```
Solution: retrieve top-20, then rerank to top-3.
```

### Multi-hop Questions
"Who founded the company that acquired OpenAI's biggest competitor?"
→ requires multiple retrieval steps, not one.

**What to learn:** Hybrid search (BM25 + dense), reranking, multi-query retrieval, HyDE.

---

## Problem 3 — The "Lost in the Middle" Problem

Research shows LLMs perform worst on context that appears in the **middle** of a long prompt. They pay most attention to the beginning and end.

```
Prompt with 10 chunks:
  [Chunk 1 - LLM attends well]
  [Chunk 2]
  [Chunk 3]
  [Chunk 4 ← ANSWER IS HERE]   ← LLM often ignores this
  [Chunk 5]
  ...
  [Chunk 10 - LLM attends well]
```

**Fixes:**
- Put most relevant chunk first or last
- Use fewer, better chunks (quality > quantity)
- Rerank before assembly

---

## Problem 4 — Hallucination Despite Retrieval

Even with correct context, LLMs can ignore it and generate from training data.

```
Context: "Our product costs $49/month."
Query:   "How much does it cost?"
Output:  "Based on industry standards, similar products cost around $30-80/month." ← hallucinated
```

**Causes:**
- Model didn't "trust" the context (poorly formatted prompt)
- Context contradicts strong training priors
- System prompt not strict enough

**What to learn:** Faithfulness evaluation, RAGAS, grounding prompts, citation forcing.

---

## Problem 5 — Stale Knowledge

Documents ingested 6 months ago are now wrong. RAG has no notion of document freshness by default.

```
Ingested: "CEO is Alice Johnson" (January)
Reality:  "CEO is Bob Chen" (July, after leadership change)
Query:    "Who is the CEO?"
RAG answers: "Alice Johnson" ← confidently wrong
```

**Fixes:** Metadata timestamps, TTL-based re-indexing, date filtering at retrieval.

---

## Problem 6 — No Multi-Turn Memory

Naive RAG treats every question independently.

```
Turn 1: "Tell me about the refund policy."
Turn 2: "How long does it take?"  ← "it" = refund? delivery? unclear

Naive RAG embeds "How long does it take?" → retrieves random chunks about time
```

**Fix:** Maintain conversation history, query rewriting to include prior context.

---

## Problem 7 — Retrieval of Irrelevant Chunks

When no good match exists, the vector store still returns its "best" match, which may be completely unrelated.

```
Query: "What's your return policy for digital downloads?"
→ No such document exists
→ Returns: best match = "Physical item return policy" (irrelevant)
→ LLM hallucinates from this noise
```

**Fix:** Score thresholding — if similarity < 0.75, return "I don't know".

---

## Problem 8 — Scalability

| Scale | Problem |
|-------|---------|
| 100K+ documents | Index build time, memory |
| Millions of queries/day | Vector search latency |
| Frequent updates | Re-embedding cost, index refresh |
| Many users | Thread/session isolation |

---

## What to Learn (Learning Path)

### Beginner
- [ ] How embeddings work (cosine similarity, vector space)
- [ ] Basic chunking strategies (fixed-size, overlap)
- [ ] Simple RAG pipeline with LangChain
- [ ] ChromaDB / pgvector setup

### Intermediate
- [ ] Hybrid search: BM25 (sparse) + dense embeddings
- [ ] Reranking with cross-encoders (Cohere Rerank, BGE)
- [ ] HyDE — Hypothetical Document Embeddings
- [ ] Multi-query retrieval (generate N query variants)
- [ ] Metadata filtering (filter by date, source, type)
- [ ] Parent-child chunking (retrieve child, send parent)
- [ ] RAGAS evaluation framework

### Advanced
- [ ] Agentic RAG with LangGraph (query planning, self-reflection)
- [ ] Corrective RAG (CRAG) — grade retrieved docs, re-retrieve if bad
- [ ] Self-RAG — model decides when to retrieve
- [ ] GraphRAG — knowledge graph + vector search
- [ ] RAPTOR — recursive abstractive processing for tree-organized retrieval
- [ ] ColBERT / late interaction models
- [ ] RAG evaluation: faithfulness, answer relevance, context precision/recall
