# Advanced RAG

These patterns go beyond the basic retrieve → generate loop. Use them when basic RAG isn't giving good enough results.

---

## 1. Agentic RAG

Instead of a fixed pipeline, an agent decides dynamically: should I retrieve? what query should I use? is the result good enough?

```
User Query
    │
    ▼
[Agent decides]
    ├── Answer from memory (no retrieval needed)
    ├── Retrieve once
    ├── Retrieve → grade → retrieve again with better query
    └── Retrieve from multiple sources → synthesize
```

```python
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(),
    name="search_docs",
    description="Search the knowledge base for relevant information.",
)

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[retriever_tool],
)

# Agent decides when and how to call the retriever
result = agent.invoke({"messages": [("human", "What is the shipping policy for Alaska?")]})
```

**Use when:** Queries vary a lot in complexity. Some need retrieval, some don't. Some need multiple retrievals.

---

## 2. Self-Correcting RAG (CRAG)

Grade the retrieved documents. If they're bad, fall back to web search or a broader query.

```
Retrieve
   │
   ▼
Grade docs: GOOD / BAD / AMBIGUOUS
   │
   ├── GOOD → Generate → Done
   ├── BAD  → Web search → Generate
   └── AMBIGUOUS → Combine KB + web → Generate
```

```python
from pydantic import BaseModel
from typing import Literal

class DocumentGrade(BaseModel):
    score: Literal["yes", "no"]

grader = ChatOpenAI().with_structured_output(DocumentGrade)

def grade_documents(state):
    question = state["question"]
    docs = state["documents"]
    good_docs = []
    needs_web = False
    for doc in docs:
        result = grader.invoke(
            f"Is this document relevant?\nQuestion: {question}\nDoc: {doc.page_content}"
        )
        if result.score == "yes":
            good_docs.append(doc)
        else:
            needs_web = True
    return {"documents": good_docs, "web_search": needs_web}
```

**Use when:** Your KB has gaps. You need a fallback when KB doesn't have the answer.

---

## 3. Query Decomposition

Break complex questions into simpler sub-questions. Answer each, then combine.

```
Complex: "Compare the return policies for electronics and clothing, and tell me which is more flexible."

Decomposed:
  Q1: "What is the return policy for electronics?"
  Q2: "What is the return policy for clothing?"
  Q3: [combine answers] → "Electronics: 15 days. Clothing: 30 days. Clothing is more flexible."
```

```python
decomposition_prompt = """Break this question into 2-3 simpler sub-questions:
Question: {question}
Output as a JSON list of strings."""

decomposer = ChatOpenAI().with_structured_output({"questions": list[str]})

def decompose_and_answer(question):
    sub_questions = decomposer.invoke(decomposition_prompt.format(question=question))
    answers = [rag_chain.invoke(q) for q in sub_questions["questions"]]
    return synthesize(question, answers)
```

**Use when:** Users ask multi-part or comparison questions.

---

## 4. Multi-Hop Retrieval

Some questions require chaining multiple retrievals — the answer to the first retrieval informs the second query.

```
Question: "Who is responsible for approving the budget for the team that handles returns?"

Hop 1: "who handles returns?" → "The Customer Experience team"
Hop 2: "who approves budget for Customer Experience team?" → "VP of Operations, Sarah Lee"

Answer: "Sarah Lee (VP of Operations)"
```

```python
def multi_hop_retrieve(question, max_hops=3):
    context = []
    current_query = question

    for _ in range(max_hops):
        docs = retriever.invoke(current_query)
        context.extend(docs)

        # Ask LLM: do I have enough to answer? If not, what should I search next?
        next_query = llm.invoke(
            f"Based on context so far, what additional info do I need?\n"
            f"Original question: {question}\n"
            f"Context: {format(context)}\n"
            f"Reply with: DONE or SEARCH: <next query>"
        )
        if next_query.content.startswith("DONE"):
            break
        current_query = next_query.content.replace("SEARCH: ", "")

    return generate(question, context)
```

**Use when:** Questions about relationships between entities that span multiple documents.

---

## 5. Graph RAG

Build a knowledge graph from your documents. Use it alongside (or instead of) vector search for relationship queries.

```
Documents → extract entities & relationships → Knowledge Graph

"Alice manages the Returns team."
"The Returns team owns the refund policy."
"The refund policy was updated in 2024."

Graph:
  Alice ──manages──► Returns Team ──owns──► Refund Policy ──updated──► 2024

Query: "Who is responsible for the 2024 refund policy?"
Graph traversal: 2024 ←── Refund Policy ←── Returns Team ←── Alice → Answer: Alice
```

**Tools:**
- **Microsoft GraphRAG** — open-source, builds community graphs from documents
- **LlamaIndex Property Graph** — integrates with Neo4j, Nebula
- **Neo4j + LangChain** — store graph in Neo4j, query with Cypher

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="...")
chain = GraphCypherQAChain.from_llm(ChatOpenAI(), graph=graph)
result = chain.invoke("Who manages the returns team?")
```

**Use when:** Data is naturally relational. Multi-hop questions about entities. Org charts, product hierarchies, legal case networks.

---

## 6. HyDE — Hypothetical Document Embeddings

Instead of embedding the query, ask the LLM to write a hypothetical answer and embed that. The hypothetical answer is in the same style as your documents.

```
Query:              "What causes inflation?"
                           ↓
LLM generates:      "Inflation is typically caused by increased money supply,
                     demand-pull factors, and supply chain disruptions..."
                           ↓
Embed this text → search vector store (better match than the short query)
```

```python
from langchain.chains import HypotheticalDocumentEmbedder

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(),
    embeddings=OpenAIEmbeddings(),
    prompt_key="web_search",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever.embeddings = hyde_embeddings
```

**Use when:** Queries are very short. Domain has specific vocabulary different from how users phrase questions.

---

## When to Use What

| Problem | Solution |
|---------|---------|
| Basic RAG gives inconsistent results | Agentic RAG (agent retries with better query) |
| KB has gaps, some questions unanswerable | CRAG (fallback to web search) |
| Complex multi-part questions | Query decomposition |
| Questions about relationships between entities | Multi-hop or Graph RAG |
| Short queries don't match doc vocabulary | HyDE |
| Questions require following chains of facts | Multi-hop retrieval |

---

## Progression Path

```
Start here:   Simple RAG (retrieve → generate)
       ↓
Add:          Threshold + reranking
       ↓
Add:          Multi-query retrieval
       ↓
Graduate to:  Agentic RAG (LangGraph)
       ↓
If needed:    CRAG / self-reflection / graph
```

Don't jump to advanced patterns before you've maximized the basics. Most production quality improvements come from better chunking and reranking, not from complex agent architectures.
