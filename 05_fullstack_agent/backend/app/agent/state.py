from typing import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: list[BaseMessage]   # conversation history
    question: str                 # current user question
    intent: str                   # "simple" | "complex"
    analysis: str                 # deep analysis for complex questions
    answer: str                   # final answer
