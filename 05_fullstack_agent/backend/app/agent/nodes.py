from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .state import AgentState

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

CLASSIFY_PROMPT = """Classify the user's question as "simple" or "complex".

simple: greetings, factual lookups, conversational, one-sentence answers
complex: requires multi-step reasoning, analysis, comparisons, explanations of concepts, problem solving

Reply with ONLY one word: simple or complex"""

ANALYZE_PROMPT = """You are a deep research analyst. Think through this question step by step.
Explore different angles, consider nuances, identify key concepts, and build a thorough understanding.
Your analysis will be used to craft a high-quality final response.

Be thorough but structured. Use clear sections in your analysis."""

RESPOND_PROMPT = """You are a helpful AI assistant in a chat application.
Answer clearly, concisely, and in a friendly tone.
Format your response with markdown when it helps readability.

{context}"""


async def classify_node(state: AgentState) -> dict:
    result = await llm.ainvoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=state["question"])
    ])
    intent = result.content.strip().lower()
    return {"intent": intent if intent in ("simple", "complex") else "simple"}


async def analyze_node(state: AgentState) -> dict:
    history_context = ""
    if state["messages"]:
        recent = state["messages"][-6:]  # last 3 exchanges for context
        history_context = "\n\nConversation context:\n" + "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content[:300]}"
            for m in recent
        )

    result = await llm.ainvoke([
        SystemMessage(content=ANALYZE_PROMPT + history_context),
        HumanMessage(content=f"Question to analyze: {state['question']}")
    ])
    return {"analysis": result.content}


async def respond_node(state: AgentState) -> dict:
    context = ""
    if state.get("analysis"):
        context = f"\n\nYou have done deep research on this question. Here is your analysis:\n{state['analysis']}\n\nNow provide a clear, well-structured answer based on this analysis."

    history = state.get("messages", [])

    result = await llm.ainvoke([
        SystemMessage(content=RESPOND_PROMPT.format(context=context)),
        *history,
        HumanMessage(content=state["question"])
    ])
    return {"answer": result.content}
