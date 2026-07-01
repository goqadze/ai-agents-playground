# RAG Tools, Frameworks & Databases

## Full Stack Overview

```
┌─────────────────────────────────────────────────────┐
│                    Your Application                  │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────▼──────────────┐
           │    Orchestration Layer    │
           │  LangChain / LlamaIndex   │
           │  LangGraph (agentic RAG)  │
           └───────────┬──────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  [Embedding]    [Vector DB]    [LLM]
  OpenAI         pgvector       GPT-4
  Cohere         Pinecone       Claude
  local model    Qdrant         Gemini
                 Weaviate       local

        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  [Reranker]     [Sparse]       [Eval]
  Cohere         Elasticsearch  RAGAS
  BGE            pgvector       LangSmith
  cross-encoder  Typesense      DeepEval
```

---

## Vector Databases

### pgvector (PostgreSQL extension)
**Best for:** teams already on PostgreSQL, transactional + vector in one DB.

```sql
CREATE EXTENSION vector;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB
);
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Search
SELECT content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

| Pro | Con |
|-----|-----|
| No extra infra if using PostgreSQL | Not purpose-built for vectors |
| ACID transactions | Slower at 1M+ vectors than dedicated DBs |
| SQL filtering, joins | Requires tuning (HNSW vs IVFFlat) |
| Open source | |

---

### Pinecone
**Best for:** managed cloud, large scale, no ops.

```python
import pinecone

pc = pinecone.Pinecone(api_key="...")
index = pc.Index("my-rag-index")

# Upsert
index.upsert(vectors=[
    {"id": "doc1", "values": embedding, "metadata": {"source": "manual.pdf"}},
])

# Query
results = index.query(vector=query_embedding, top_k=5, filter={"source": "manual.pdf"})
```

| Pro | Con |
|-----|-----|
| Fully managed, scales to billions | Paid (no free tier for production) |
| Real-time updates | Vendor lock-in |
| Metadata filtering | No SQL joins |
| Low latency globally | Data leaves your infra |

---

### Qdrant
**Best for:** self-hosted or cloud, fast, feature-rich.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)
client.create_collection("docs", vectors_config=VectorParams(size=1536, distance=Distance.COSINE))

# Upsert
client.upsert("docs", points=[{"id": 1, "vector": embedding, "payload": {"text": "..."}}])

# Query with filter
client.search("docs", query_vector=embedding, limit=5,
              query_filter={"must": [{"key": "year", "match": {"value": 2024}}]})
```

| Pro | Con |
|-----|-----|
| HNSW index (fast) | Newer ecosystem |
| Sparse + dense (hybrid) | Smaller community than Pinecone |
| Open source + cloud option | |
| Payload (metadata) filtering | |

---

### Weaviate
**Best for:** semantic + keyword + GraphQL API, multimodal.

```python
import weaviate

client = weaviate.connect_to_local()
collection = client.collections.get("Document")

# Query
results = collection.query.near_text(query="return policy", limit=3)
# Also supports: near_vector, bm25, hybrid
```

| Pro | Con |
|-----|-----|
| Built-in hybrid search | More complex setup |
| GraphQL + REST APIs | Heavier resource usage |
| Multimodal (text + image) | |
| Open source + managed | |

---

### ChromaDB
**Best for:** local development, prototyping.

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")

collection.add(documents=["text..."], ids=["id1"], embeddings=[embedding])
results = collection.query(query_embeddings=[q_emb], n_results=3)
```

| Pro | Con |
|-----|-----|
| Zero config, runs local | Not production-ready at scale |
| Python-native | No native hybrid search |
| Great for learning | Limited metadata filtering |

---

### FAISS (Meta)
**Best for:** high-performance local search, no persistence needed.

```python
import faiss
import numpy as np

index = faiss.IndexFlatIP(1536)      # inner product (cosine if normalized)
index = faiss.IndexIVFFlat(quantizer, 1536, 100)  # faster approximate

index.add(np.array(embeddings, dtype=np.float32))
D, I = index.search(np.array([query_emb], dtype=np.float32), k=5)
```

| Pro | Con |
|-----|-----|
| Extremely fast | No persistence (must save/load manually) |
| Runs entirely in memory | No metadata filtering |
| Used in production at scale | No updates (must rebuild index) |

---

### Comparison Table

| DB | Scale | Managed | Hybrid | SQL | License | Best for |
|----|-------|---------|--------|-----|---------|----------|
| pgvector | Medium | Self | Partial | Yes | OSS | Existing PG users |
| Pinecone | Very Large | Yes | Yes | No | Paid | Production, no ops |
| Qdrant | Large | Both | Yes | No | OSS | Self-hosted production |
| Weaviate | Large | Both | Yes | No | OSS | Multimodal |
| ChromaDB | Small | Self | No | No | OSS | Dev/prototyping |
| FAISS | Very Large | Self | No | No | OSS | Research, high-perf |
| Milvus | Very Large | Both | Yes | No | OSS | Enterprise scale |

---

## Embedding Models

### OpenAI
```python
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536-dim
emb = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072-dim
```
**Cost:** ~$0.02 per 1M tokens (3-small) | **Quality:** Very good | **Speed:** Fast API

### Cohere
```python
from langchain_cohere import CohereEmbeddings

emb = CohereEmbeddings(model="embed-english-v3.0")
```
**Best for:** multilingual, enterprise | Supports input type (search_document vs search_query)

### Open Source (local, free)
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
emb = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```
**Cost:** Free | **Speed:** Depends on hardware | **Privacy:** Data stays local

---

## Orchestration Frameworks

### LangChain
The most popular RAG framework. High-level chains and retrievers.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vectorstore.as_retriever(),
)
result = chain.invoke("What is the return policy?")
```

Pros: huge ecosystem, many integrations | Cons: abstraction can hide complexity

### LlamaIndex
Data-framework focused. Better for complex document ingestion, indexing strategies.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the refund policy?")
```

Pros: excellent for document ingestion, more index types | Cons: smaller community than LangChain

### LangGraph
For **agentic RAG** — when the retrieval loop itself needs to be stateful and conditional.
See `RAG_FLOW_LANGGRAPH.md` for full patterns.

---

## Reranking Models

| Model | Type | Cost | Use |
|-------|------|------|-----|
| `cohere rerank-v3` | API | Paid | Production |
| `BAAI/bge-reranker-large` | Local | Free | Self-hosted |
| `cross-encoder/ms-marco-MiniLM` | Local | Free | Lightweight |
| `Jina Reranker` | API/Local | Free tier | Fast |

---

## Evaluation Tools

### RAGAS
Evaluate RAG pipeline without human labels.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
# faithfulness:       does answer match retrieved context?
# answer_relevancy:   does answer address the question?
# context_precision:  are retrieved chunks actually useful?
```

### LangSmith
Trace every RAG call — see which chunks were retrieved, what the LLM was given, latency.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
# All LangChain calls are now traced automatically
```

### DeepEval
Unit-test your RAG pipeline.

```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is the return policy?",
    actual_output="30 days from purchase",
    retrieval_context=["Returns are accepted within 30 days of purchase."],
)
metric = FaithfulnessMetric(threshold=0.8)
metric.measure(test_case)
```
