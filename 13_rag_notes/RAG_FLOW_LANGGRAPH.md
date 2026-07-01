# RAG Flow with LangChain & LangGraph

## 1. Simple RAG Chain (LangChain)

The baseline — no state, single pass, no memory.

```
User Query
    │
    ▼
Embed Query ──► Vector Search ──► Top-k Chunks
                                       │
                                       ▼
                              Assemble Prompt
                              (system + context + query)
                                       │
                                       ▼
                                      LLM
                                       │
                                       ▼
                                    Answer
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

vectorstore = PGVector(connection="postgresql+psycopg://...", embeddings=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context below. If not in context, say you don't know.

Context: {context}
Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = chain.invoke("What is the refund policy?")
```

**Limitations:** No memory, no multi-turn, no retry on bad retrieval.

---

## 2. Conversational RAG (LangChain + History)

Adds chat history and query rewriting for follow-up questions.

```
Turn 1: "What is the return policy?"          → direct query
Turn 2: "How long does it take?"              → ambiguous, needs history

                   ┌─────────────────────┐
Chat History ─────►│  Query Rewriter LLM  │──► "How long does a return take?"
                   └─────────────────────┘
                                │
                                ▼
                         Vector Search
                                │
                                ▼
                               LLM
                                │
                                ▼
                             Answer
```

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder

# Step 1: Rewrite query using history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and the latest user question, "
               "rewrite it as a standalone question. Don't answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

# Step 2: Answer with context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY this context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Full chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Usage
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []
response = rag_chain.invoke({"input": "What is the return policy?", "chat_history": chat_history})
chat_history.extend([HumanMessage("What is the return policy?"), AIMessage(response["answer"])])

response2 = rag_chain.invoke({"input": "How long does it take?", "chat_history": chat_history})
# "it" is resolved to "return" from history → correct retrieval
```

---

## 3. Basic RAG Graph (LangGraph)

Stateful, explicit nodes — same result as #1 but as a graph for extensibility.

```
START
  │
  ▼
[retrieve]  ──► embed query → similarity_search → add chunks to state
  │
  ▼
[generate]  ──► build prompt from state.context → call LLM → add response to state
  │
  ▼
END
```

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: list[str]

def retrieve(state: RAGState) -> dict:
    query = state["messages"][-1].content
    vs = PGVector(connection="postgresql+psycopg://...", embeddings=OpenAIEmbeddings())
    docs = vs.similarity_search(query, k=3)
    return {"context": [d.page_content for d in docs]}

def generate(state: RAGState) -> dict:
    context = "\n\n".join(state["context"])
    system = f"Answer ONLY using this context:\n\n{context}"
    messages = [SystemMessage(content=system)] + list(state["messages"])
    response = ChatOpenAI(model="gpt-4o-mini").invoke(messages)
    return {"messages": [response]}

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()

# Single question
result = app.invoke({"messages": [("human", "What is the return policy?")]})

# Multi-turn — LangGraph persists state across turns via checkpointer
from langgraph.checkpoint.memory import MemorySaver

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-123"}}

app.invoke({"messages": [("human", "What is the return policy?")]}, config)
app.invoke({"messages": [("human", "How long does it take?")]}, config)
# State (including context from turn 1) is preserved
```

---

## 4. Agentic RAG with Self-Reflection (LangGraph)

The agent decides whether retrieved docs are good enough. If not, it rewrites the query and retrieves again.

```
START
  │
  ▼
[retrieve]
  │
  ▼
[grade_docs] ──── all docs irrelevant? ──► [rewrite_query] ──► [retrieve] (loop)
  │                                                                   ↑
  │ good docs                                                         │
  ▼                                                                   │
[generate] ──── hallucination? ──────────────────────────────────────┘
  │
  │ good answer
  ▼
END
```

```python
from typing import Literal
from pydantic import BaseModel, Field

class GradeDoc(BaseModel):
    score: Literal["yes", "no"] = Field(description="Is this doc relevant to the query?")

grader_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(GradeDoc)

def grade_documents(state):
    question = state["messages"][-1].content
    docs = state["documents"]
    relevant = []
    for doc in docs:
        grade = grader_llm.invoke(f"Question: {question}\nDocument: {doc.page_content}")
        if grade.score == "yes":
            relevant.append(doc)
    return {"documents": relevant, "web_search_needed": len(relevant) == 0}

def decide_after_grade(state) -> Literal["generate", "rewrite"]:
    return "rewrite" if state["web_search_needed"] else "generate"

def rewrite_query(state):
    question = state["messages"][-1].content
    better = ChatOpenAI().invoke(f"Rewrite this query to improve retrieval: {question}")
    return {"messages": [("human", better.content)]}

# Build graph with conditional edges
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite_query)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", decide_after_grade)
graph.add_edge("rewrite", "retrieve")  # loop back
graph.add_edge("generate", END)
```

---

## 5. Corrective RAG — CRAG (LangGraph)

If retrieved docs score poorly, fall back to web search.

```
[retrieve]
    │
    ▼
[grade] ──► score < threshold? ──► [web_search]
    │                                    │
    │ score OK                           │
    ▼                                    ▼
[generate] ◄─────────────────────────────
```

```python
from langchain_community.tools import TavilySearchResults

web_search = TavilySearchResults(max_results=3)

def web_search_node(state):
    query = state["messages"][-1].content
    results = web_search.invoke(query)
    web_docs = [Document(page_content=r["content"]) for r in results]
    return {"documents": web_docs}

def route_after_grade(state) -> Literal["generate", "web_search"]:
    return "web_search" if state["web_search_needed"] else "generate"

graph.add_conditional_edges("grade_documents", route_after_grade)
graph.add_edge("web_search", "generate")
```

---

## 6. Multi-Agent RAG (LangGraph)

Different agents for different knowledge sources — routes query to the right agent.

```
                     ┌──────────────┐
                     │    Router    │
                     │    Agent     │
                     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        [Policy RAG]  [Product RAG]  [HR RAG]
        policy docs   product docs   hr docs
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                         [Synthesizer]
                            │
                            ▼
                          Answer
```

```python
def route_question(state) -> Literal["policy", "product", "hr"]:
    router_llm = ChatOpenAI().with_structured_output(RouteQuery)
    source = router_llm.invoke(state["messages"][-1].content)
    return source.datasource  # "policy" | "product" | "hr"

graph.add_conditional_edges(START, route_question)
```

---

## LangGraph vs LangChain RAG — When to Use What

| Scenario | Use |
|----------|-----|
| Simple Q&A, no history | LangChain LCEL chain |
| Multi-turn chat | LangChain conversational RAG |
| Retry on bad retrieval | LangGraph |
| Self-reflection / grading | LangGraph |
| Multiple knowledge sources | LangGraph multi-agent |
| Human-in-the-loop approval | LangGraph with interrupt |
| Production with persistence | LangGraph + PostgreSQL checkpointer |

---

## Checkpointer Options (LangGraph State Persistence)

```python
from langgraph.checkpoint.memory import MemorySaver        # dev only, in-memory
from langgraph.checkpoint.sqlite import SqliteSaver        # single-process
from langgraph.checkpoint.postgres import PostgresSaver    # production

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
app = graph.compile(checkpointer=checkpointer)

# Every thread_id = separate conversation with full state history
config = {"configurable": {"thread_id": "user-456-session-1"}}
```
