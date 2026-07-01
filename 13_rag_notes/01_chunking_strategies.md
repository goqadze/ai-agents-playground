# Chunking Strategies

Chunking is how you split documents before embedding. Wrong chunking = wrong retrieval, regardless of how good your LLM is.

---

## Why It Matters

The chunk is what gets embedded and stored. Its vector must represent a focused idea — not too broad, not too narrow.

```
Too large (1000 tokens):
  "...company history... return policy... CEO bio... shipping times..."
  → embedding is an average of all topics → bad match for any specific query

Too small (50 tokens):
  "Returns are accepted within"
  → no meaning without context
```

---

## 1. Fixed-Size Chunking

Split by character/token count, regardless of content.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(document)
```

**Good:** Simple, fast, predictable.  
**Bad:** Cuts mid-sentence. Loses semantic boundaries.  
**Use when:** Quick prototype, structured data with uniform sections.

---

## 2. Recursive Chunking

Tries to split at natural boundaries first: `\n\n` → `\n` → `.` → ` ` → character.

```python
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""],
    chunk_size=500,
    chunk_overlap=50,
)
```

**Good:** Respects paragraph/sentence structure better than fixed-size.  
**Bad:** Still size-based, just smarter about where to cut.  
**Use when:** Most general-purpose cases. This is the default starting point.

---

## 3. Semantic Chunking

Groups sentences with similar meaning together. Splits when topic changes.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
chunks = splitter.split_text(document)
```

**Good:** Chunks align with actual topics. Much better embeddings.  
**Bad:** Slower (embeds every sentence). Chunk sizes are unpredictable.  
**Use when:** Long documents with multiple distinct sections (reports, manuals).

---

## 4. Parent-Child Chunking

Index small chunks (precise embedding), but retrieve the larger parent when matched.

```
Parent (2000 tokens): full policy section
  └── Child (200 tokens): single paragraph  ← indexed
  └── Child (200 tokens): next paragraph    ← indexed
  └── Child (200 tokens): ...

Query matches a child → LLM gets the parent (full context)
```

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
```

**Good:** Best of both — precise matching + rich context for generation.  
**Bad:** More complex setup. Requires a document store alongside the vector store.  
**Use when:** Production. This is the most reliable general strategy.

---

## 5. Document Hierarchy Chunking

Preserve document structure: Title → Section → Subsection → Paragraph.

```
Document
├── Chapter 1: "Return Policy"
│   ├── Section 1.1: "Standard Returns"
│   │   └── Paragraph: "Items returned within 30 days..."
│   └── Section 1.2: "Sale Items"
│       └── Paragraph: "Sale items are final sale..."
```

Each chunk carries metadata about its position in the hierarchy:
```python
{"source": "policy.pdf", "chapter": "Return Policy", "section": "Sale Items"}
```

**Good:** Enables metadata filtering by section. LLM gets structural context.  
**Bad:** Requires parsing document structure (PDF headers, Markdown, HTML).  
**Use when:** Well-structured documents (legal, technical docs, manuals).

---

## Chunk Size Trade-offs

| Size | Embedding Quality | Context for LLM | Retrieval |
|------|------------------|-----------------|-----------|
| 100-200 tokens | Very precise | Too little context | High precision, low recall |
| 300-500 tokens | Good | Enough | Balanced |
| 800-1000 tokens | Diluted | Rich | Low precision, high recall |

**chunk_overlap:** Set to ~10-15% of chunk_size. Prevents important context from being cut at a boundary.

```
chunk_size=500, chunk_overlap=50

Chunk 1: tokens 0–500
Chunk 2: tokens 450–950   ← 50-token overlap
Chunk 3: tokens 900–1400
```

---

## Decision Guide

```
Starting out?          → Recursive chunking (chunk_size=500, overlap=50)
Production system?     → Parent-child chunking
Long structured docs?  → Semantic or hierarchy chunking
Need max precision?    → Small child chunks (150–200 tokens) + parent retrieval
```
