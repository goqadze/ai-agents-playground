"""
Graph nodes — each function is one step in the agent pipeline.

New in 06 vs 05:
  - llm_with_tools : the LLM is bound to ALL_TOOLS so it can call them
  - setup_agent_node : initialises agent_messages from the question + optional analysis
  - agent_node : calls llm_with_tools on agent_messages (one ReAct step)
  - tools_node : executes whatever tool the LLM asked for (from langgraph.prebuilt)
  - extract_answer_node : pulls the final text out of agent_messages into answer
  - should_continue : router — decides if we loop back for another tool call or stop

  classify_node and analyze_node are identical to project 05.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .tools import ALL_TOOLS

# ── LLM setup ─────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# bind_tools tells the LLM about our tools.
# It serialises each tool's name, docstring, and parameter schema into
# the system prompt automatically — the LLM then knows when and how to call them.
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ToolNode is a prebuilt LangGraph node that:
#   1. reads the tool_calls field from the last AIMessage
#   2. calls the matching Python function with the provided args
#   3. wraps the return value in a ToolMessage and puts it in agent_messages
tools_node = ToolNode(ALL_TOOLS, messages_key="agent_messages")

# ── Prompts ───────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """Classify the user's question as "simple" or "complex".

simple: greetings, factual lookups, conversational, one-sentence answers,
        math calculations, text manipulation (these can be answered with tools fast)
complex: requires multi-step reasoning, analysis, comparisons, deep explanations,
         problem solving, research

Reply with ONLY one word: simple or complex"""

ANALYZE_PROMPT = """You are a deep research analyst. Think through this question step by step.
Explore different angles, consider nuances, identify key concepts, and build a thorough understanding.
Your analysis will be used to craft a high-quality final response.

Be thorough but structured. Use clear sections in your analysis."""

AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant in a chat application.
Answer clearly, concisely, and in a friendly tone.
Format your response with markdown when it helps readability.

You have access to tools — use them whenever they help you give a better answer:
  - calculator     : for any math or number crunching
  - word_count     : to analyse text length
  - get_current_datetime : to find out today's date/time
  - reverse_text   : to flip text
  - text_transform : to change text case or style

{context}"""

# ── Node functions ─────────────────────────────────────────────────────────────

async def classify_node(state: AgentState) -> dict:
    """Classify question as simple or complex."""
    result = await llm.ainvoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=state["question"])
    ])
    intent = result.content.strip().lower()
    return {"intent": intent if intent in ("simple", "complex") else "simple"}


async def analyze_node(state: AgentState) -> dict:
    """Deep analysis for complex questions (reused from project 05)."""
    history_context = ""
    if state.get("history"):
        recent = state["history"][-6:]
        history_context = "\n\nConversation context:\n" + "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content[:300]}"
            for m in recent
        )

    result = await llm.ainvoke([
        SystemMessage(content=ANALYZE_PROMPT + history_context),
        HumanMessage(content=f"Question to analyze: {state['question']}")
    ])
    return {"analysis": result.content}


async def setup_agent_node(state: AgentState) -> dict:
    """
    Initialise agent_messages before the ReAct loop starts.

    This node builds the opening message list:
      - A system message with instructions (and the analysis if available)
      - The conversation history so the agent remembers prior turns
      - The current user question

    These messages are the starting point for the agent ↔ tools loop.
    """
    context = ""
    if state.get("analysis"):
        context = (
            "\n\nYou have already done a deep analysis of this question:\n"
            f"{state['analysis']}\n\n"
            "Use this analysis to inform your answer."
        )

    system = SystemMessage(content=AGENT_SYSTEM_PROMPT.format(context=context))

    # Include conversation history so the agent knows the prior context
    history = state.get("history", [])

    return {
        "agent_messages": [
            system,
            *history,
            HumanMessage(content=state["question"]),
        ]
    }


async def agent_node(state: AgentState) -> dict:
    """
    One step of the ReAct loop — call the LLM with the current messages.

    The LLM will either:
      A) Call a tool  → returns an AIMessage with tool_calls filled in
      B) Give an answer → returns an AIMessage with content filled in

    The result is appended to agent_messages (via the add_messages reducer).
    The graph's should_continue router then decides what to do next.
    """
    result = await llm_with_tools.ainvoke(state["agent_messages"])
    # Returning {"agent_messages": [result]} APPENDS result to the list
    # because of the add_messages reducer on AgentState.agent_messages
    return {"agent_messages": [result]}


def should_continue(state: AgentState) -> str:
    """
    Router: look at the last message from the agent and decide what runs next.

    If the LLM asked to call a tool → go to "tools" node
    If the LLM gave a final answer  → go to "extract" node
    """
    last_message = state["agent_messages"][-1]
    if last_message.tool_calls:
        return "tools"   # loop back through the tool node
    return "extract"     # done — pull out the answer


async def extract_answer_node(state: AgentState) -> dict:
    """
    Pull the final answer text out of agent_messages and store it in 'answer'.
    The last message is always the LLM's final reply (no tool calls).
    """
    last_message = state["agent_messages"][-1]
    return {"answer": last_message.content}
