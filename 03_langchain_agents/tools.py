"""
Custom tools the agent can use.

In LangChain, a "tool" is just a Python function decorated with @tool.
The decorator does two things:
  1. Reads the function name and docstring → tells the LLM what the tool does
     and WHEN to call it.
  2. Wraps the function so LangChain can call it automatically during the
     agent loop.

The LLM never runs code itself — it just decides WHICH tool to call and with
WHAT arguments. LangChain then calls the real Python function and feeds the
result back to the LLM.
"""

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use this when the user asks to add numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this when the user asks to multiply numbers."""
    return a * b


@tool
def word_count(text: str) -> int:
    """Count the number of words in a given text string."""
    return len(text.split())


@tool
def reverse_text(text: str) -> str:
    """Reverse the characters in a text string."""
    return text[::-1]


# Collect all tools in a list — we pass this to the agent
ALL_TOOLS = [add, multiply, word_count, reverse_text]
