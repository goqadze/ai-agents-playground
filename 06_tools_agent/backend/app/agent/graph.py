"""
LangGraph graph for project 06 — same classify/analyze routing as 05,
but the "respond" node is replaced by a full ReAct tool-calling loop.

Graph layout:

  START
    │
    ▼
  classify ──────────────────────────────────────┐
    │                                             │
    │ [route_by_intent]                           │
    │                                             │
    ├─ "simple"  ──────────────┐                  │
    │                          ▼                  │
    └─ "complex" ──→ analyze ──→ setup_agent       │
                                    │              │
                                    ▼              │
                               agent_node  ◀───────┘ (tool loop)
                                    │
                         [should_continue]
                                    │
                    ┌───────────────┴───────────────┐
                    │ "tools"                       │ "extract"
                    ▼                               ▼
               tools_node                    extract_answer
               (runs the tool)                     │
                    │                              END
                    └──── back to agent_node ──────┘
                          (ReAct loop)

Key difference from 05:
  - "respond" node (one LLM call) is replaced by:
      setup_agent → agent ↔ tools loop → extract_answer
  - The agent can call tools multiple times before giving a final answer.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    classify_node,
    analyze_node,
    setup_agent_node,
    agent_node,
    tools_node,       # ToolNode instance from nodes.py
    should_continue,
    extract_answer_node,
)


def route_by_intent(state: AgentState) -> str:
    """Router after classify — sends simple questions to setup_agent directly."""
    return state.get("intent", "simple")


def build_graph():
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("classify",     classify_node)
    graph.add_node("analyze",      analyze_node)
    graph.add_node("setup_agent",  setup_agent_node)
    graph.add_node("agent",        agent_node)
    graph.add_node("tools",        tools_node)        # prebuilt ToolNode
    graph.add_node("extract",      extract_answer_node)

    # Entry point
    graph.set_entry_point("classify")

    # After classify: route based on intent
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "simple":  "setup_agent",   # skip analysis
            "complex": "analyze",       # do deep analysis first
        }
    )

    # Complex path: analyze → setup_agent
    graph.add_edge("analyze", "setup_agent")

    # Both paths converge at setup_agent → agent
    graph.add_edge("setup_agent", "agent")

    # ReAct loop: agent decides whether to call a tool or finish
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools":   "tools",    # tool call → run the tool → back to agent
            "extract": "extract",  # no tool call → extract the final answer
        }
    )

    # After running tools, always go back to the agent for the next reasoning step
    graph.add_edge("tools", "agent")

    # After extracting the answer, we're done
    graph.add_edge("extract", END)

    return graph.compile()


agent_graph = build_graph()
