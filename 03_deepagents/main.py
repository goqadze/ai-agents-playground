"""
Deep Agents + LangChain + OpenAI demo.

─────────────────────────────────────────────────────────────────────────────
WHAT IS DEEP AGENTS?
─────────────────────────────────────────────────────────────────────────────
Deep Agents is a Python library built on top of LangChain + LangGraph.
It adds extra capabilities that plain agents don't have:

  ┌─────────────────────────────────────────────────────────────┐
  │  Plain LangChain agent       │  Deep Agent                  │
  ├─────────────────────────────────────────────────────────────┤
  │  Calls tools in a loop       │  Same PLUS:                  │
  │                              │  • Task planning (breaks big │
  │                              │    tasks into steps first)   │
  │                              │  • Virtual file system       │
  │                              │    (agent can write/read     │
  │                              │    files to store context)   │
  │                              │  • Subagent delegation       │
  │                              │    (spawn specialist agents) │
  │                              │  • Human-in-the-loop         │
  │                              │  • Long-term memory          │
  └─────────────────────────────────────────────────────────────┘

─────────────────────────────────────────────────────────────────────────────
HOW THE EXECUTION PIPELINE WORKS
─────────────────────────────────────────────────────────────────────────────
  1. Task Planning  — agent breaks your request into steps
  2. Tool Invocation — executes tools for each step
  3. Context Management — stores intermediate results (virtual files)
  4. Subagent Spawning — delegates specialist work when needed
  5. Synthesis — compiles everything into a final answer

─────────────────────────────────────────────────────────────────────────────
MODEL STRING FORMAT
─────────────────────────────────────────────────────────────────────────────
Deep Agents uses the format  "provider:model-name"
  "openai:gpt-4.1-nano"        ← OpenAI provider, gpt-4.1-nano model
  "anthropic:claude-sonnet-4"  ← Anthropic provider
  "google:gemini-2.0-flash"    ← Google provider
"""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from tools import ALL_TOOLS

# ── 1. Load API key ───────────────────────────────────────────────────────────

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your-api-key-here":
    raise ValueError(
        "No API key found. Open .env and replace 'your-api-key-here' with your real key."
    )

os.environ["OPENAI_API_KEY"] = api_key  # Deep Agents reads this from the environment

# ── 2. Create the deep agent ─────────────────────────────────────────────────
#
# create_deep_agent() is the main entry point.
#
# Parameters:
#   model        : "provider:model-name" — tells Deep Agents which LLM to use
#   tools        : list of plain Python functions the agent can call
#   system_prompt: optional instructions that shape the agent's personality

agent = create_deep_agent(
    model="openai:gpt-4.1-nano",
    tools=ALL_TOOLS,
    system_prompt=(
        "You are a helpful assistant. "
        "For any task given to you, first make a plan, then execute it step by step."
    ),
)

# ── 3. Helper: run the agent ──────────────────────────────────────────────────

def run(question: str) -> None:
    """
    Send a question to the agent and print the final answer.

    agent.invoke() runs the full agent loop (plan → tools → synthesize)
    and returns the final state dict.

    The final answer is in:
        result["messages"][-1].content
    The last message in the list is always the agent's final reply.
    """
    print(f"\n{'─'*60}")
    print(f"Question: {question}")
    print(f"{'─'*60}")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # The result is a dict with a "messages" key — list of all messages exchanged
    final_message = result["messages"][-1]
    print(f"Answer: {final_message.content}")


# ── 4. Demo ───────────────────────────────────────────────────────────────────

def main():
    print("=== Deep Agents + LangChain + OpenAI Demo ===\n")

    # Simple math — agent uses tools
    run("What is 123 multiplied by 456?")

    # Multi-step — agent plans: add first, then multiply
    run("Add 50 and 75, then multiply the result by 4.")

    # Text tool
    run("How many words are in: 'Deep Agents is a powerful framework for building AI agents'?")

    # Weather tool
    run("What is the weather like in Tbilisi?")

    # No tools needed — agent answers directly
    run("What is the capital of Georgia (the country)?")

    # Interactive
    print(f"\n{'─'*60}")
    user_q = input("Ask the agent anything (or press Enter to quit): ").strip()
    if user_q:
        run(user_q)


if __name__ == "__main__":
    main()
