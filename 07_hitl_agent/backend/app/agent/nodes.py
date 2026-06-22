"""
Graph nodes for the HITL agent.

The key new concept here is interrupt() from langgraph.types.

HOW interrupt() WORKS
─────────────────────
1. The graph is running normally, streaming events to the frontend.
2. clarify_node calls interrupt({...}).
3. LangGraph immediately:
     a. Saves the full graph state to the MemorySaver checkpointer.
     b. Pauses execution — the astream_events loop in chat.py ends cleanly.
4. The backend detects the pause, sends {"type": "interrupt", ...} to the frontend.
5. The frontend shows option buttons to the user.
6. User clicks a button → frontend POSTs to /resume with their choice.
7. Backend calls astream_events(Command(resume=choice), config=same_config).
8. LangGraph restores the saved state, resume() returns the user's choice,
   and clarify_node returns {"user_choice": choice}.
9. The graph continues to respond_node as if nothing happened.

NODE FLOW
─────────
  intro_node    → streams a 1-sentence acknowledgment to the user
  clarify_node  → calls interrupt() — graph pauses here, waits for user choice
  respond_node  → uses user_choice to tailor the final answer
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from .state import AgentState

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

# The options shown to the user at the interrupt point.
# Each key is an option label; each value is the instruction sent to the LLM.
RESPONSE_FORMATS = {
    "Concise summary":          "Answer in 2-3 sentences maximum. Be direct.",
    "Detailed explanation":     "Give a thorough, comprehensive explanation covering all important aspects.",
    "Step-by-step walkthrough": "Break the answer into clear numbered steps.",
    "Examples and analogies":   "Use relatable real-world examples and analogies to explain the concept.",
}

OPTIONS = list(RESPONSE_FORMATS.keys())


async def intro_node(state: AgentState) -> dict:
    """
    Stream a brief 1-sentence acknowledgment before pausing for user input.

    We deliberately do NOT answer the question here — we just set the stage.
    The streaming tokens from this LLM call are forwarded to the frontend
    by chat.py listening for on_chat_model_stream events on the "intro" node.
    """
    result = await llm.ainvoke([
        SystemMessage(content=(
            "You are a helpful assistant. The user asked you a question. "
            "Write exactly ONE short sentence acknowledging their question "
            "and telling them you want to tailor your answer to their needs. "
            "Do NOT answer the question. Do NOT mention any format options."
        )),
        *state["messages"],
        HumanMessage(content=state["question"]),
    ])
    return {"intro_text": result.content}


async def clarify_node(state: AgentState) -> dict:
    """
    Pause the graph and ask the user to choose a response format.

    interrupt() does two things:
      1. Sends the dict you pass to it back to whoever called astream_events
         (accessible via graph.get_state(config).tasks[0].interrupts[0].value)
      2. Pauses the graph — execution stops here until resumed with Command(resume=...)

    The value returned by interrupt() IS whatever was passed to Command(resume=...).
    So `choice` will be the string the user clicked on (e.g. "Concise summary").
    """
    choice = interrupt({
        "question": "How would you like me to answer?",
        "options":  OPTIONS,
    })
    return {"user_choice": choice}


async def respond_node(state: AgentState) -> dict:
    """
    Generate the final answer using the format the user chose.

    By this point state["user_choice"] is set (e.g. "Step-by-step walkthrough")
    and we look up the matching instruction to send to the LLM.
    """
    instruction = RESPONSE_FORMATS.get(
        state["user_choice"],
        "Answer clearly and helpfully.",
    )

    result = await llm.ainvoke([
        SystemMessage(content=(
            f"You are a helpful assistant. {instruction} "
            "Use markdown formatting when it helps readability."
        )),
        *state["messages"],
        HumanMessage(content=state["question"]),
    ])
    return {"answer": result.content}
