from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import classify_node, analyze_node, respond_node


def route_by_intent(state: AgentState) -> str:
    return state.get("intent", "simple")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {"simple": "respond", "complex": "analyze"}
    )
    graph.add_edge("analyze", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


agent_graph = build_graph()
