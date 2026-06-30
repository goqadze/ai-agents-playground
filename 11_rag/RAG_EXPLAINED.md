# RAG — Retrieval-Augmented Generation

## What is RAG?

RAG is a technique that gives a language model access to **external knowledge at query time**, without retraining it.

Instead of relying only on what the model learned during training, RAG:
1. **Retrieves** the most relevant documents for the user's question from a knowledge base
2. **Augments** the LLM prompt with those documents as extra context
3. **Generates** an answer that is grounded in the retrieved content

---

## Why RAG?

| Problem with plain LLMs | How RAG solves it |
|--------------------------|-------------------|
| Knowledge cutoff — the model doesn't know recent events | Your knowledge base can contain fresh data |
| Hallucinations — model invents facts confidently | Answers are grounded in real retrieved documents |
| No access to private/internal data | You control the knowledge base |
| Expensive to fine-tune for every domain | No retraining needed — just update documents |
| Can't cite sources | Retrieval gives you the exact source chunks |

---

## RAG Flow (Step by Step)

### Phase 1 — Indexing (done once, or on document update)

```
Your Documents (PDFs, TXTs, web pages…)
         │
         ▼
   ┌─────────────┐
   │   Chunking  │  ← Split long docs into ~500-token overlapping chunks
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │    Embedding    │  ← Convert each chunk to a numeric vector (e.g. 1536 dimensions)
   │     Model       │    "What is RAG?" → [0.12, -0.43, 0.87, ...]
   └──────┬──────────┘
          │
          ▼
   ┌─────────────────┐
   │  Vector Store   │  ← Store vectors + original text (ChromaDB, Pinecone, pgvector…)
   └─────────────────┘
```

### Phase 2 — Querying (happens on every user question)

```
User: "How does RAG reduce hallucinations?"
         │
         ▼
   ┌─────────────────┐
   │    Embedding    │  ← Embed the query using the SAME embedding model
   │     Model       │
   └──────┬──────────┘
          │  query vector
          ▼
   ┌─────────────────┐
   │  Vector Store   │  ← Find top-k chunks whose vectors are closest (cosine similarity)
   │   Similarity    │
   │    Search       │
   └──────┬──────────┘
          │  top-3 relevant chunks
          ▼
   ┌──────────────────────────────────────┐
   │           Prompt Assembly            │
   │                                      │
   │  System: "Answer only using context" │
   │  Context:                            │
   │    [Chunk 1 text]                    │
   │    [Chunk 2 text]                    │
   │    [Chunk 3 text]                    │
   │  User: "How does RAG reduce..."      │
   └──────┬───────────────────────────────┘
          │
          ▼
   ┌─────────────────┐
   │      LLM        │  ← Claude, GPT-4, etc.
   │   (Claude)      │
   └──────┬──────────┘
          │
          ▼
   Answer: "RAG reduces hallucinations because the LLM is instructed
            to answer only from the provided context chunks, which
            are real retrieved documents…"
```

---

## Key Concepts

### Chunking
Breaking documents into smaller pieces. Two important parameters:
- **chunk_size** — how many tokens per chunk (typically 256–1024)
- **chunk_overlap** — how many tokens overlap between adjacent chunks (prevents cutting context at boundaries)

```
Document: "The cat sat on the mat. The mat was old. The old mat belonged to a witch."

chunk_size=5, chunk_overlap=2:
  Chunk 1: "The cat sat on the"
  Chunk 2: "on the mat The mat"
  Chunk 3: "The mat was old The"
  Chunk 4: "old The old mat belonged"
```

### Embeddings
A way to turn text into a numeric vector that captures semantic meaning. Similar texts → similar vectors.

```
"dog"  → [0.8, 0.1, -0.3, ...]
"puppy"→ [0.7, 0.2, -0.2, ...]   ← close to "dog"
"car"  → [-0.4, 0.9, 0.6, ...]   ← far from "dog"
```

### Vector Similarity
Finding how "close" two vectors are. The most common metric:

**Cosine Similarity** = cos(θ) between two vectors
- 1.0 = identical meaning
- 0.0 = unrelated
- -1.0 = opposite

### Vector Store
A database optimized for similarity search over millions of vectors in milliseconds.

Popular choices:
| Store | Best for |
|-------|----------|
| **ChromaDB** | Local / small projects (this project uses it) |
| **Pinecone** | Managed cloud, large scale |
| **pgvector** | Already using PostgreSQL |
| **Weaviate** | Open-source, graph + vector |
| **FAISS** | High-performance, local, no persistence |

---

## Use Cases

### 1. Customer Support Bot
```
Knowledge base: product manuals, FAQs, support tickets
Query: "My device won't turn on"
RAG retrieves: troubleshooting guide section → LLM generates step-by-step answer
```

### 2. Internal Knowledge Assistant
```
Knowledge base: Confluence pages, Notion docs, Slack archives
Query: "What's our vacation policy?"
RAG retrieves: HR handbook chunk → LLM answers accurately
```

### 3. Legal / Medical Document Q&A
```
Knowledge base: case law, medical journals
Query: "What precedents exist for data breach liability?"
RAG retrieves: relevant case excerpts → LLM synthesizes with citations
```

### 4. Code Documentation Search
```
Knowledge base: GitHub repos, API docs
Query: "How do I paginate results in our API?"
RAG retrieves: relevant README / docstring chunks → LLM explains with examples
```

### 5. Research Assistant
```
Knowledge base: uploaded PDFs / papers
Query: "What did the 2023 study say about sleep and memory?"
RAG retrieves: abstract + methodology chunks → LLM summarizes findings
```

---

## RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|--|-----|-------------|
| **Updates knowledge** | Yes — add documents anytime | No — requires retraining |
| **Cost** | Low | High (GPU training) |
| **Hallucination control** | Good (grounded in retrieved text) | Moderate |
| **Teaches new reasoning styles** | No | Yes |
| **Best for** | Domain knowledge, fresh data | Style, tone, specialized tasks |

**Rule of thumb:** Use RAG for "what the model knows", use fine-tuning for "how the model behaves".

---

## What This Project Implements

```
11_rag/
├── RAG_EXPLAINED.md     ← You are here
├── requirements.txt
├── .env.example
├── rag_demo.py          ← Simple CLI demo (start here)
└── rag_api.py           ← FastAPI REST API version
```

### Quick Start
```bash
# 1. Set API key
cp .env.example .env
# edit .env → OPENAI_API_KEY=your_key

# 2a. Run with Docker (recommended)
docker compose up --build
# API available at http://localhost:8000/docs

# 2b. OR run locally (CLI demo)
pip install -r requirements.txt
python rag_demo.py

# 2c. OR run the API locally
pip install -r requirements.txt
python rag_api.py
# then POST http://localhost:8000/ingest  and  POST http://localhost:8000/query
```
