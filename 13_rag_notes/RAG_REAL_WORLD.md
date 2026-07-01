# RAG Real-World Examples & Production Patterns

## Example 1 — Customer Support Bot (E-commerce)

**Company:** Online retailer with 50K product SKUs and complex return policies.

**Problem before RAG:** Support agents spend 60% of time answering questions already in the manual. Chatbot based on fine-tuned model gives wrong answers when policies change.

### Architecture

```
Customer Chat
     │
     ▼
┌────────────────────────────────────────────────────────┐
│                    RAG Support Bot                      │
│                                                         │
│  Query: "Can I return a sale item bought 45 days ago?" │
│                    │                                    │
│                    ▼                                    │
│  ┌─────────────────────────────┐                       │
│  │  Multi-Query Retriever       │                       │
│  │  → "sale item return policy" │                       │
│  │  → "return window duration"  │                       │
│  │  → "exceptions to 30 day rule"                      │
│  └──────────────┬──────────────┘                       │
│                 │                                       │
│                 ▼                                       │
│  ┌─────────────────────────────┐                       │
│  │  Metadata Filter             │                       │
│  │  source IN (policy_2024.pdf) │                       │
│  │  type = "return_policy"      │                       │
│  └──────────────┬──────────────┘                       │
│                 │                                       │
│                 ▼                                       │
│  ┌─────────────────────────────┐                       │
│  │  Cohere Reranker             │ top-20 → top-3       │
│  └──────────────┬──────────────┘                       │
│                 │                                       │
│                 ▼                                       │
│  ┌─────────────────────────────┐                       │
│  │  GPT-4o with citations       │                       │
│  │  "According to our Return    │                       │
│  │  Policy (updated Jan 2024),  │                       │
│  │  sale items..."              │                       │
│  └─────────────────────────────┘                       │
└────────────────────────────────────────────────────────┘
```

### Knowledge Base
```
├── policy_2024.pdf        → return & refund policies
├── product_catalog.json   → product specs, categories
├── faq_2024.md            → top 200 customer questions
└── support_tickets.csv    → past resolved tickets (anonymized)
```

### Key Design Decisions
- **Metadata filtering** by `source` and `updated_at` prevents stale answers
- **Multi-query** handles different phrasings of the same question
- **Score threshold = 0.72** — if no match, escalate to human agent
- **Citation required** in system prompt — builds trust, enables audit

### Results
- 40% reduction in support ticket volume
- Average handle time: 8 min → 2 min
- Customer satisfaction: +12 NPS points

---

## Example 2 — Internal Legal Knowledge Assistant (Law Firm)

**Problem:** Junior associates spend hours searching case law and past contracts. Senior partners can't bill this time.

### Architecture

```
Attorney Query: "Find precedents for force majeure in SaaS contracts post-2020"
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│                   Legal RAG Agent (LangGraph)               │
│                                                             │
│  [query_analyzer] → extracts: topic, jurisdiction, date     │
│         │                                                   │
│         ▼                                                   │
│  [parallel_retrieval] ─────────────────────────────────────│
│    ├── Case Law DB (pgvector, 500K cases)                   │
│    ├── Contract Templates DB                                │
│    └── Firm's Past Contracts DB                             │
│         │                                                   │
│         ▼                                                   │
│  [reranker] → BGE cross-encoder, keeps top 5               │
│         │                                                   │
│         ▼                                                   │
│  [generate_memo] → GPT-4o with forced citations             │
│         │          "See Smith v. Jones (2021), § 8.2..."   │
│         ▼                                                   │
│  [human_review] ← partner approves before sending          │
└────────────────────────────────────────────────────────────┘
```

### Special Considerations
- **Local embedding model** (BGE) — client data never leaves firm servers
- **Self-hosted Qdrant** — on-prem, not cloud
- **Human-in-the-loop** (LangGraph `interrupt`) before any output goes to client
- **Audit log** — every retrieval + generation logged with timestamps

---

## Example 3 — Medical Literature Q&A (Healthcare)

**Problem:** Clinicians need fast access to latest drug interaction studies. PubMed has 35M papers.

### Architecture

```
Query: "Interactions between metformin and SGLT2 inhibitors in elderly patients"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Medical RAG Pipeline                       │
│                                                             │
│  [date_filter] → published_after = 2020                    │
│  [mesh_filter] → MeSH terms: "drug interactions", "T2DM"   │
│  [hybrid_search]                                            │
│      dense:  semantic embeddings (BioBERT)                  │
│      sparse: BM25 on medical terminology (exact drug names) │
│         │                                                   │
│         ▼                                                   │
│  [evidence_grader] → grades each paper: RCT > review > case│
│         │                                                   │
│         ▼                                                   │
│  [generate_summary]                                         │
│      "Based on 3 RCTs (2021-2023), combination therapy...  │
│       SOURCE: NEJM 2022 DOI:10.1056/..."                   │
│         │                                                   │
│         ▼                                                   │
│  [disclaimer_appended] → "Not clinical advice. Consult..."  │
└─────────────────────────────────────────────────────────────┘
```

### Special Considerations
- **Domain-specific embedding model** (BioBERT, PubMedBERT) — general embeddings miss medical terminology
- **BM25 for drug names** — "metformin" must exact-match, not semantically approximate
- **Evidence grading** — RCT > systematic review > case study
- **HIPAA compliance** — no patient data in prompts

---

## Example 4 — Code Documentation Search (Developer Tool)

**Problem:** Developer at a large tech company can't find how to use internal APIs across 1000+ repos.

### Architecture

```
Dev Query: "How do I paginate the user search endpoint in the auth service?"
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│                   Code RAG System                             │
│                                                              │
│  Sources indexed:                                            │
│    ├── OpenAPI specs (YAML → parsed per endpoint)            │
│    ├── README files                                          │
│    ├── Code comments (AST-extracted docstrings)              │
│    └── Internal Confluence / Notion                          │
│                                                              │
│  Chunking strategy:                                          │
│    OpenAPI:    one chunk per endpoint definition             │
│    Code:       one chunk per function/class with docstring   │
│    Docs:       semantic chunking by section                  │
│                                                              │
│  [hybrid_search]                                             │
│      dense: semantic (what does it do?)                      │
│      sparse: BM25 (exact function/endpoint names)            │
│         │                                                    │
│         ▼                                                    │
│  [generate_with_example]                                     │
│      "The /users/search endpoint supports cursor pagination. │
│       Example: GET /users/search?cursor=eyJpZCI6MTAwfQ==    │
│       SOURCE: auth-service/openapi.yaml#/paths/users/search" │
└──────────────────────────────────────────────────────────────┘
```

---

## Example 5 — Multi-Modal RAG (Insurance)

**Problem:** Claims adjusters need to search policy documents + damage photos together.

```
Claim: Photo of water-damaged ceiling + "Is this covered?"
     │
     ├── [vision_model] → extracts: "water damage, ceiling, staining, ~20% area"
     │
     └── [text_query] → "water damage ceiling coverage"
                │
                ▼
         [Weaviate multi-modal]
             dense text vectors + CLIP image vectors
                │
                ▼
         [retrieved] → relevant policy sections + similar past claims
                │
                ▼
         [generate] → "Based on section 4.2 (Water Damage), and similar
                       claims #45231 and #45889, this appears covered.
                       Estimated payout: $2,400–$4,100."
```

---

## Production Checklist

### Before Launch
- [ ] Evaluate with RAGAS — faithfulness > 0.8, relevancy > 0.75
- [ ] Set similarity score threshold (no garbage retrieval)
- [ ] Test edge cases: empty KB, out-of-domain questions, adversarial inputs
- [ ] Implement rate limiting on embedding API calls
- [ ] Add query logging for offline analysis

### Monitoring
- [ ] Track retrieval latency (p50, p95, p99)
- [ ] Monitor embedding API costs per query
- [ ] Alert on high "I don't know" rate (may indicate KB gap)
- [ ] Alert on high latency (retrieval or generation spike)
- [ ] Log which chunks are retrieved most (helps identify KB gaps)

### Knowledge Base Management
- [ ] Document versioning — keep old chunks for audit
- [ ] Re-indexing pipeline when source docs update
- [ ] Deduplication — same content in multiple docs wastes tokens
- [ ] Access control — per-user or per-role filtering (e.g., HR docs only for HR)

### Cost Control
- [ ] Cache embeddings for repeated queries (Redis, exact-match)
- [ ] Cache LLM responses for identical (query + context) pairs
- [ ] Use smaller embedding model (3-small vs 3-large) — 10% quality for 4x cheaper
- [ ] Compress context before sending to LLM (removes boilerplate)

---

## Architecture Template (Production)

```
                          User
                           │
                     [API Gateway]
                    rate limit, auth
                           │
                    [Query Service]
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        [Cache Hit]   [Retriever]  [Router]
          Redis        Qdrant       multi-source
              │            │
              │       [Reranker]
              │        Cohere
              │            │
              └────────────▼
                    [LLM Service]
                    GPT-4o / Claude
                           │
                    [Response + Sources]
                           │
                    [Eval & Logging]
                      LangSmith
```
