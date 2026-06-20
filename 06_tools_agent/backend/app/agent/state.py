"""
Shared state passed between every node in the LangGraph graph.

In LangGraph, the state is a TypedDict — just a plain Python dict
with typed keys. Every node receives the full state and returns a
dict of only the keys it wants to update.

New in 06 vs 05:
  - agent_messages: the working message list for the ReAct tool-calling loop.
    Uses add_messages so new messages are APPENDED (not replaced) each time
    a node returns {"agent_messages": [new_msg]}.
"""

from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(dict):
    """
    history        : conversation history loaded from the DB (context only)
    question       : the current user question
    intent         : "simple" | "complex" — set by the classify node
    analysis       : deep analysis text — set by the analyze node (complex only)
    agent_messages : working messages for the agent ↔ tools ReAct loop
                     add_messages reducer automatically appends new messages
    answer         : final answer text — set by the extract node
    """
    history:        list[BaseMessage]
    question:       str
    intent:         str
    analysis:       str
    agent_messages: Annotated[list[BaseMessage], add_messages]
    answer:         str
