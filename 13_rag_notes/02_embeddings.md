# Embeddings

An embedding is a list of numbers (vector) that represents the meaning of text. Similar meaning → similar vectors.

---

## What Embeddings Actually Are

```
"dog"   → [0.81, 0.12, -0.33, ...]   (1536 numbers)
"puppy" → [0.79, 0.14, -0.31, ...]   ← very close
"car"   → [-0.42, 0.88, 0.61, ...]   ← far away
```

The position in this high-dimensional space encodes semantic meaning. The model learned this from billions of text examples.

---

## Cosine Similarity

The standard way to measure how close two vectors are.

```
similarity = cos(θ) between two vectors

1.0  = identical meaning
0.5  = related
0.0  = unrelated
-1.0 = opposite meaning
```

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim = cosine_similarity(embed("dog"), embed("puppy"))  # → ~0.91
sim = cosine_similarity(embed("dog"), embed("car"))    # → ~0.12
```

Vector stores compute this for every stored vector against your query — returning the top-k closest.

---

## Similarity Search

```
Query: "How do I reset my password?"
         │
         ▼
    embed query → [0.12, -0.44, 0.87, ...]
         │
         ▼
    compare against all stored vectors
         │
         ▼
    return top-3 closest chunks:
      0.94 — "To reset your password, go to Settings > Security..."
      0.88 — "Forgot password? Click the link on the login page..."
      0.71 — "Account security settings allow you to change credentials..."
```

---

## Embedding Model Selection

### OpenAI (API, paid)
```python
from langchain_openai import OpenAIEmbeddings

# Fast, cheap, good quality
emb = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims

# Higher quality, more expensive
emb = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072 dims
```

### Open Source (local, free)
```python
from langchain_huggingface import HuggingFaceEmbeddings

# Best open-source general model
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Lightweight, fast
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Model Comparison

| Model | Dims | Quality | Cost | Speed |
|-------|------|---------|------|-------|
| `text-embedding-3-small` | 1536 | Good | Low | Fast |
| `text-embedding-3-large` | 3072 | Best | Medium | Fast |
| `text-embedding-ada-002` | 1536 | OK | Low | Fast |
| `BAAI/bge-large-en` | 1024 | Good | Free | Medium |
| `all-MiniLM-L6-v2` | 384 | OK | Free | Very Fast |

---

## Key Rules

**Same model for indexing and querying.** If you index with `text-embedding-3-small`, you must query with it too. Different models produce incompatible vector spaces.

**Embedding model ≠ LLM.** The embedding model only converts text to vectors. It doesn't generate answers. Use a small, fast model for embeddings.

**Bigger dimensions ≠ always better.** `all-MiniLM-L6-v2` at 384 dims often beats `ada-002` at 1536 dims on domain-specific tasks, because it was trained more carefully.

---

## Comparing Models (Experiment)

```python
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

docs = [
    "Returns are accepted within 30 days of purchase.",
    "Our CEO founded the company in 2010.",
    "Items must be unopened to qualify for a refund.",
]
query = "What is the return window?"

for model_name, emb in [
    ("OpenAI small", OpenAIEmbeddings(model="text-embedding-3-small")),
    ("MiniLM", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")),
]:
    doc_vecs = emb.embed_documents(docs)
    q_vec = emb.embed_query(query)
    scores = [np.dot(q_vec, d) / (np.linalg.norm(q_vec) * np.linalg.norm(d)) for d in doc_vecs]
    print(f"\n{model_name}:")
    for doc, score in zip(docs, scores):
        print(f"  {score:.3f} — {doc[:50]}")
```

Expected: both should rank doc[0] and doc[2] highest (return-related), doc[1] lowest.
