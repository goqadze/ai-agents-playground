# RAG Optimizations

## Optimization Map

```
                        RAG Pipeline
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   [Indexing]           [Retrieval]          [Generation]
   Chunking             Hybrid Search        Prompt Design
   Embedding Model      Reranking            Context Order
   Metadata             HyDE                 Faithfulness
   Hierarchy            Multi-Query          Caching
```

---

## 1. Chunking Optimizations

### Fixed-Size (Baseline — avoid in production)
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```
Problem: cuts sentences arbitrarily.

### Semantic Chunking
Split at topic boundaries, not character count. Groups sentences with similar embeddings together.
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(OpenAIEmbeddings())
chunks = splitter.split_text(document)
# Result: chunks end where the topic changes, not at 500 chars
```

### Sentence-Window Chunking
Index single sentences, but retrieve surrounding window as context.
```
Index:    [S1] [S2] [S3*] [S4] [S5]
Match:     S3 (most relevant)
Retrieve:  S2 + S3 + S4  (window of 3)
```
Benefit: precise matching + rich context.

### Parent-Child Chunking (Best for most cases)
```
Parent chunk (2000 tokens):  Full section of a document
   └── Child chunk (200 tokens): Single paragraph

Index:    child chunks (small → precise embedding)
Retrieve: parent chunk (large → full context for LLM)
```
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
```

### Document Summary Index (RAPTOR-lite)
Store a summary of each document alongside its chunks. Retrieve summaries first to filter, then chunks.

---

## 2. Embedding Optimizations

### Choosing the Right Model

| Model | Dims | Best for | Cost |
|-------|------|----------|------|
| `text-embedding-3-small` | 1536 | General, fast | Low |
| `text-embedding-3-large` | 3072 | High precision | Medium |
| `text-embedding-ada-002` | 1536 | Legacy, widely supported | Low |
| `cohere-embed-v3` | 1024 | Multilingual, enterprise | Medium |
| `BAAI/bge-large-en` | 1024 | Open source, on-prem | Free |
| `nomic-embed-text` | 768 | Open source, fast | Free |

### Matryoshka Embeddings
`text-embedding-3-*` supports truncation — you can store 256-dim vectors (cheaper, faster) and upgrade to 1536-dim only when needed.

```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=256  # smaller index, lower cost, ~5% accuracy drop
)
```

---

## 3. Retrieval Optimizations

### Hybrid Search (Most Impactful)
Combine **dense** (semantic) + **sparse** (keyword/BM25) search. Catches both semantic and exact matches.

```
Query: "OpenAI GPT-4 pricing"

Dense only:  retrieves docs about "AI model costs" (semantic match)
Sparse only: retrieves docs with exact string "GPT-4 pricing"
Hybrid:      gets both → better recall

Score = α × dense_score + (1 - α) × sparse_score   (α ≈ 0.5)
```

Supported by: pgvector (with pg_trgm), Weaviate, Qdrant, Elasticsearch, Pinecone.

### Reranking (Cross-Encoder)
First-stage retrieval is fast but imprecise. Second-stage reranking is slow but accurate.

```
Stage 1: Retrieve top-20 chunks (fast, approximate)
Stage 2: Rerank top-20 with cross-encoder → return top-3 (slow, precise)

Cross-encoder sees (query, chunk) pair together → much better relevance signal
```

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

### HyDE — Hypothetical Document Embeddings
Instead of embedding the query, ask the LLM to generate a hypothetical answer, then embed that.

```
Query:              "What causes inflation?"
                           ↓
LLM generates:      "Inflation is caused by excessive money supply,
                     supply chain disruptions, demand pull factors..."
                           ↓
Embed hypothetical doc → search vector store
```

Why it works: the hypothetical answer is in the same style/vocabulary as real documents in the index.

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(),
    embeddings=OpenAIEmbeddings(),
    prompt_key="web_search",
)
```

### Multi-Query Retrieval
Generate N variations of the query → retrieve for each → deduplicate → union of results.

```
Original: "How do I reset my password?"
Variants:
  - "password recovery steps"
  - "forgot password process"
  - "account access lost"

→ Retrieve 3 per variant = 9 candidates → deduplicate → top-5
```

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(),
)
```

### Score Thresholding
Reject low-quality matches instead of returning garbage.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75, "k": 5},
)
# Returns empty list if best match < 0.75 → LLM says "I don't know"
```

### Metadata Filtering
Narrow the search space before similarity search.

```python
results = vectorstore.similarity_search(
    query="return policy",
    filter={"source": "policy_manual", "year": {"$gte": 2024}},
    k=5,
)
```

---

## 4. Context Assembly Optimizations

### Order Matters (Lost in the Middle)
Put the most relevant chunk **first**, not in the middle.

```python
# Sort by relevance score descending before assembling prompt
chunks = sorted(retrieved_chunks, key=lambda x: x.score, reverse=True)
```

### Context Compression
Compress each retrieved chunk to only the relevant sentence(s) before sending to LLM.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
# Each 500-token chunk → compressed to 1-2 relevant sentences
```

---

## 5. Generation Optimizations

### Strict Grounding Prompt
```
System: You are a helpful assistant. Answer ONLY using the provided context.
        If the answer is not in the context, say exactly:
        "I don't have information about that in my knowledge base."
        Do NOT use your training knowledge. Do NOT speculate.

Context:
{context}
```

### Forced Citations
```
System: After your answer, list the source(s) you used in this format:
        Sources: [chunk_id_1], [chunk_id_2]
```

### Response Caching
Cache LLM responses for identical (query + context) pairs.

```python
from langchain.cache import InMemoryCache
import langchain
langchain.llm_cache = InMemoryCache()
```

---

## 6. Architecture-Level Optimizations

### Agentic RAG
Instead of one fixed retrieve → generate loop, the agent decides:
- Do I need to retrieve? (maybe the answer is in history)
- Which query to use?
- Should I retrieve again if the first result was poor?

See `RAG_FLOW_LANGGRAPH.md` for implementation.

### Corrective RAG (CRAG)
Grade retrieved documents → if grade is poor, fall back to web search.

```
Retrieve → Grade (good/bad) → if bad: web search → generate
```

### GraphRAG (Microsoft)
Build a knowledge graph from documents. Traverse relationships for multi-hop questions.

```
"Who manages the team responsible for the product with the highest NPS?"

Vector RAG:  might fail (3 hops)
GraphRAG:    Person → manages → Team → owns → Product → has metric → NPS ✓
```

---

## Quick Win Priority Order

1. **Parent-child chunking** — biggest retrieval quality improvement for most use cases
2. **Score thresholding** — eliminates garbage retrieval causing hallucinations  
3. **Reranking** — cheap API call, significant precision boost
4. **Hybrid search** — catches exact-match queries dense search misses
5. **Multi-query** — improves recall for ambiguous queries
6. **Semantic chunking** — better than fixed-size for structured documents
