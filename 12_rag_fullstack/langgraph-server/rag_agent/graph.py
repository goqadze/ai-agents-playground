"""
RAG Graph — LangGraph implementation of Retrieval-Augmented Generation.

Graph flow:
  START → retrieve → generate → END

  retrieve: embed the last user message → search pgvector → return top-k chunks
  generate: build a context-aware system prompt → call LLM → return AI response

The graph state is persisted by the LangGraph server in PostgreSQL,
so every thread (conversation) remembers its full message history.
"""

import os
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    # add_messages merges new messages into the list instead of replacing it
    messages: Annotated[list[BaseMessage], add_messages]
    # retrieved chunks injected as context for the generate step
    context: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vector_store() -> PGVector:
    return PGVector(
        connection=os.environ["DATABASE_URL"],
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="documents",
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> dict:
    """Embed the latest user query and fetch the most relevant chunks from pgvector."""
    query = state["messages"][-1].content
    try:
        docs = _vector_store().similarity_search(query, k=3)
        context = [doc.page_content for doc in docs]
    except Exception:
        context = []
    return {"context": context}


def generate(state: RAGState) -> dict:
    """Build a RAG prompt from the retrieved context and generate an answer."""
    chunks = state.get("context", [])

    if chunks:
        context_text = "\n\n---\n\n".join(chunks)
        system_content = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the context provided below. If the answer is not in the context, say "
            "'I don't have information about that in the knowledge base.'\n\n"
            f"CONTEXT:\n{context_text}"
        )
    else:
        system_content = (
            "You are a helpful assistant. No documents have been added to the "
            "knowledge base yet. Politely ask the user to add documents via the "
            "Ingest panel on the left before asking questions."
        )

    llm = ChatOpenAI(model="gpt-4.1-nano", streaming=True)
    messages = [SystemMessage(content=system_content)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
