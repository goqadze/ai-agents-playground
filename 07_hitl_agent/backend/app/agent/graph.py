"""
Graph for the HITL agent.

LINEAR flow — no branching needed here:

  START
    │
    ▼
  intro_node       ← LLM streams a 1-sentence acknowledgment
    │
    ▼
  clarify_node     ← calls interrupt() → graph PAUSES here
    │                 (execution resumes when user picks an option)
    ▼
  respond_node     ← LLM streams the full answer in the chosen format
    │
    ▼
   END

The critical difference from all previous projects:
  graph.compile(checkpointer=memory)

The checkpointer (MemorySaver) saves the full graph state whenever the graph
pauses at an interrupt(). This saved state is what allows the graph to be
resumed later from exactly where it stopped — with all state intact.

Without a checkpointer, interrupt() would have nowhere to save state and
resuming would be impossible.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import intro_node, clarify_node, respond_node

# MemorySaver stores graph state in memory (resets on server restart).
# For production use SqliteSaver or a Postgres-backed checkpointer.
memory = MemorySaver()


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intro",   intro_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("intro")
    graph.add_edge("intro",   "clarify")
    graph.add_edge("clarify", "respond")
    graph.add_edge("respond", END)

    # checkpointer= is what enables interrupt() to work
    return graph.compile(checkpointer=memory)


agent_graph = build_graph()
