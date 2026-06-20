"""
LANGGRAPH SIMPLE EXAMPLE — A conversational AI agent
=====================================================

Key concepts:
  - State      : A shared dictionary that flows through the graph. Every node reads from it and writes to it.
  - Node       : A Python function that does one job (e.g. call the LLM, run a tool).
  - Edge       : A connection between nodes. Can be fixed or conditional (branch on logic).
  - Graph      : The whole system — nodes + edges + state, compiled into a runnable app.

Flow in this example:
  User input → [chatbot node] → LLM response → end
  (The LLM can call a tool. If it does, a conditional edge routes to [tool node], then back to [chatbot node].)

  ┌──────────┐     always     ┌──────────────┐
  │ START    │──────────────► │   chatbot    │◄─────┐
  └──────────┘                └──────┬───────┘      │
                                     │               │
                          used tool? │ no → END      │
                                     │ yes           │
                                     ▼               │
                              ┌──────────────┐       │
                              │    tools     │───────┘
                              └──────────────┘
"""

import os
from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


# ── 1. STATE ──────────────────────────────────────────────────────────────────
# The state is just a typed dictionary that gets passed around.
# `add_messages` is a special reducer: instead of overwriting the list,
# it *appends* new messages — so conversation history is preserved automatically.

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ── 2. TOOLS ──────────────────────────────────────────────────────────────────
# Tools are plain Python functions decorated with @tool.
# The LLM decides when (and whether) to call them based on the docstring.

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Fake data — replace with a real API call if you like.
    weather_data = {
        "london": "Cloudy, 15°C",
        "tokyo":  "Sunny, 28°C",
        "paris":  "Rainy, 12°C",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 2' or '10 * 5'."""
    try:
        # eval is fine here for a demo; in production use a safe math parser.
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


tools = [get_weather, calculate]


# ── 3. LLM ────────────────────────────────────────────────────────────────────
# Bind the tools to the model so the model knows it can call them.
# The model won't call tools automatically — it decides based on the user's message.

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
llm_with_tools = llm.bind_tools(tools)


# ── 4. NODES ──────────────────────────────────────────────────────────────────
# A node is just a function: it receives the current state and returns an update.
# Returning {"messages": [...]} appends those messages to state["messages"].

def chatbot(state: State) -> dict:
    """Send the conversation to the LLM and get a reply."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode is a prebuilt node that:
#   1. Reads the last AI message to find which tool(s) the LLM requested.
#   2. Calls those tools.
#   3. Returns the results as ToolMessages appended to state["messages"].
tool_node = ToolNode(tools)


# ── 5. GRAPH ──────────────────────────────────────────────────────────────────
# Build the graph by adding nodes, then connecting them with edges.

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

# Always start at the chatbot node.
builder.add_edge(START, "chatbot")

# `tools_condition` is a prebuilt conditional function:
#   - If the last message contains tool_calls → route to "tools"
#   - Otherwise → route to END
builder.add_conditional_edges("chatbot", tools_condition)

# After running tools, always go back to chatbot so the LLM can process the result.
builder.add_edge("tools", "chatbot")

# Compile turns the builder into a runnable app.
graph = builder.compile()


# ── 6. RUN ────────────────────────────────────────────────────────────────────

def chat(user_message: str) -> str:
    """Run one turn through the graph and return the final AI reply."""
    result = graph.invoke({"messages": [("user", user_message)]})
    # result["messages"] is the full conversation; the last item is the AI reply.
    return result["messages"][-1].content


if __name__ == "__main__":
    print("LangGraph agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = chat(user_input)
        print(f"AI:  {response}\n")
