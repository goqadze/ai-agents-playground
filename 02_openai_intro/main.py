"""
Simple LangChain + OpenAI demo.

LangChain is a framework that sits on top of LLM APIs (like OpenAI) and gives
you reusable building blocks: models, prompts, chains, memory, tools, etc.

This file shows the three most important building blocks:
  1. ChatOpenAI      — the LangChain wrapper around OpenAI's chat model
  2. ChatPromptTemplate — a reusable prompt with placeholders
  3. Chain (pipe |)  — connecting prompt → model → output in one line
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 1. Load API key from .env ────────────────────────────────────────────────

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "your-api-key-here":
    raise ValueError(
        "No API key found. Open .env and replace 'your-api-key-here' with your real key."
    )

# ── 2. Create the model ──────────────────────────────────────────────────────
#
# ChatOpenAI is LangChain's wrapper for OpenAI chat models.
# temperature controls creativity: 0 = focused/deterministic, 1 = more creative

model = ChatOpenAI(
    model="gpt-4.1-nano",   # small and cheap — great for learning
    temperature=0,
    api_key=api_key,
)

# ── 3. Create a prompt template ──────────────────────────────────────────────
#
# A prompt template is a reusable prompt with named placeholders like {topic}.
# You fill them in later with .invoke({"topic": "..."}). This keeps prompts
# clean and reusable across different inputs.
#
# ChatPromptTemplate.from_messages() takes a list of (role, text) tuples:
#   "system"  = instructions for the AI
#   "human"   = the user's message (same as "user" in raw OpenAI)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful teacher. Explain things simply and clearly."),
    ("human", "Explain {topic} in 2-3 sentences, like I'm a beginner."),
])

# ── 4. Build a chain ─────────────────────────────────────────────────────────
#
# The pipe operator | connects steps left-to-right:
#   prompt → model → parser
#
# StrOutputParser just extracts the plain text string from the model's response
# object, so we don't have to do response.content ourselves.

chain = prompt | model | StrOutputParser()

# ── 5. Run examples ──────────────────────────────────────────────────────────

def explain(topic: str) -> str:
    """Fill in the prompt template and run the chain."""
    # .invoke() runs the full chain and returns the final output
    return chain.invoke({"topic": topic})


def main():
    print("=== LangChain + OpenAI demo ===\n")

    # Example 1 — hardcoded topic
    topic1 = "what LangChain is"
    print(f"Topic: {topic1}")
    print(explain(topic1))
    print()

    # Example 2 — hardcoded topic
    topic2 = "what a prompt template is"
    print(f"Topic: {topic2}")
    print(explain(topic2))
    print()

    # Example 3 — interactive
    user_topic = input("Enter your own topic to explain (or press Enter to skip): ").strip()
    if user_topic:
        print(explain(user_topic))


if __name__ == "__main__":
    main()
