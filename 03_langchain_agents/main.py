"""
LangChain ReAct Agent demo.

─────────────────────────────────────────────────────────────────────────────
WHAT IS AN AGENT?
─────────────────────────────────────────────────────────────────────────────
A regular LangChain chain runs in a straight line:
    prompt → model → output   (one shot, done)

An AGENT is different. It runs in a loop:
    1. The LLM reads the question and decides: "Do I need a tool?"
    2. If YES  → call the tool, get the result, loop back to step 1
    3. If NO   → produce the final answer and stop

This loop is called the ReAct pattern:
    RE-ason → ACT (call tool) → observe result → RE-ason again → ...

─────────────────────────────────────────────────────────────────────────────
WHAT IS LANGGRAPH?
─────────────────────────────────────────────────────────────────────────────
LangGraph is LangChain's library for building agents as graphs.
Each "node" in the graph is a step (call LLM, call tool, etc.).
The edges decide which node runs next.

create_react_agent() from LangGraph builds a ready-made ReAct graph for us:
    ┌─────────┐     tool call?    ┌──────────┐
    │   LLM   │ ──────────────▶  │   Tool   │
    │  node   │ ◀──────────────  │   node   │
    └─────────┘    tool result   └──────────┘
         │
         │ no tool call (final answer)
         ▼
       DONE
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS

# ── 1. Load API key ───────────────────────────────────────────────────────────

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "your-api-key-here":
    raise ValueError(
        "No API key found. Open .env and replace 'your-api-key-here' with your real key."
    )

# ── 2. Create the model ───────────────────────────────────────────────────────

model = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,    # 0 = deterministic — best for agents that do calculations
    api_key=api_key,
)

# ── 3. Create the agent ───────────────────────────────────────────────────────
#
# create_react_agent() wires up the ReAct loop for us:
#   - model  : the LLM that reasons and decides which tool to call
#   - tools  : the list of tools the LLM is allowed to use
#
# Internally it builds a LangGraph state machine:
#   agent node → (if tool call) → tool node → back to agent node → repeat

agent = create_react_agent(model=model, tools=ALL_TOOLS)


# ── 4. Helper: run the agent and print each step ─────────────────────────────

def run(question: str) -> None:
    """
    Run the agent on a question and print every step so you can see the loop.

    agent.stream() yields events one by one as the agent processes them.
    Each event is a dict like:
        {"agent": {"messages": [AIMessage(...)]}}   ← LLM thinking / tool call
        {"tools": {"messages": [ToolMessage(...)]}} ← tool result
    """
    print(f"\n{'─'*60}")
    print(f"Question: {question}")
    print(f"{'─'*60}")

    for step in agent.stream(
        {"messages": [("user", question)]},
        stream_mode="updates",   # one event per graph node update
    ):
        # Each step is a dict: {node_name: {messages: [...]}}
        node_name = list(step.keys())[0]          # "agent" or "tools"
        messages  = step[node_name]["messages"]

        for msg in messages:
            msg_type = type(msg).__name__          # AIMessage / ToolMessage

            if msg_type == "AIMessage":
                # The LLM produced either a tool call or the final answer
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  🔧 LLM wants to call tool: {tc['name']}")
                        print(f"     with args: {tc['args']}")
                else:
                    # No tool call → this is the final answer
                    print(f"\n  ✅ Final answer: {msg.content}")

            elif msg_type == "ToolMessage":
                # The tool ran and returned a result
                print(f"  📦 Tool result ({msg.name}): {msg.content}")


# ── 5. Demo questions ─────────────────────────────────────────────────────────

def main():
    print("=== LangChain ReAct Agent Demo ===")
    print("Watch the agent REASON → ACT → OBSERVE loop in action.\n")

    # This needs one tool call
    run("What is 42 multiplied by 7?")

    # This needs two tool calls chained together
    run("Add 15 and 27, then multiply the result by 3.")

    # This uses a text tool
    run("How many words are in the sentence: 'The quick brown fox jumps over the lazy dog'?")

    # This uses no tools — the agent answers directly
    run("What is the capital of France?")

    print(f"\n{'─'*60}")
    user_q = input("Ask the agent anything (or press Enter to quit): ").strip()
    if user_q:
        run(user_q)


if __name__ == "__main__":
    main()
