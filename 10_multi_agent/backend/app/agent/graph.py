"""
The multi-agent pipeline graph.

Each node is a separate specialized agent.  Data flows through the shared
ResearchState — each agent reads what it needs and writes its output back.

  START
    │
    ▼
  planner        ← breaks topic into 4 research questions
    │
    ▼
  researcher     ← answers all 4 questions with detailed findings
    │
    ▼
  writer         ← synthesizes findings into a polished article
    │
    ▼
  END

This is the simplest multi-agent topology: a linear pipeline.
More advanced patterns (supervisor, parallel fan-out, debate loop)
build on this same foundation.
"""

from langgraph.graph import StateGraph, START, END
from .state import ResearchState
from .nodes import planner_node, researcher_node, writer_node


def build_graph():
    graph = StateGraph(ResearchState)

    # Register the three agent nodes
    graph.add_node("planner",    planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer",     writer_node)

    # Wire them in sequence
    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer",     END)

    return graph.compile()


# Single shared graph instance — compiled once at startup
research_graph = build_graph()
