"""
Chat API — two endpoints:

  POST /{id}/chat    — starts a new turn, streams intro then pauses at interrupt
  POST /{id}/resume  — resumes the paused graph with the user's chosen option

HOW THE TWO CALLS RELATE
─────────────────────────────────────────────────────────────────────────────
  /chat   creates a new thread_id, stores it in active_threads[conv_id]
          runs the graph until interrupt() fires
          sends {"type": "interrupt", "question": ..., "options": [...]}

  /resume reads active_threads[conv_id] to get the same thread_id
          calls astream_events(Command(resume=choice), config=same_thread)
          LangGraph restores saved state and continues from clarify_node
          streams the final answer tokens from respond_node
          sends {"type": "done"}

thread_id connects the two calls — it's the key the MemorySaver uses
to find the right saved graph state.
"""

import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from ..database import get_db, AsyncSessionLocal
from ..models import Conversation, Message
from ..schemas import ChatRequest, ResumeRequest
from ..agent.graph import agent_graph

router = APIRouter(prefix="/api/conversations", tags=["chat"])

# Maps conversation_id → {"thread_id": str, "intro_text": str, "steps": list, "is_first": bool}
# In production this would live in the database or Redis.
active_threads: dict[int, dict] = {}

STEP_LABELS = {
    "intro":   "Reading your question...",
    "clarify": "Preparing options...",
    "respond": "Writing your answer...",
}


# ── /chat ─────────────────────────────────────────────────────────────────────

@router.post("/{conversation_id}/chat")
async def chat(
    conversation_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    db_messages = result.scalars().all()

    lc_messages = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in db_messages
    ]
    is_first_message = len(db_messages) == 0

    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        agent_steps=[],
    )
    db.add(user_msg)
    await db.commit()

    # Fresh thread_id for every new chat turn
    thread_id = str(uuid.uuid4())
    active_threads[conversation_id] = {
        "thread_id":    thread_id,
        "intro_text":   "",
        "steps":        [],
        "is_first":     is_first_message,
        "question":     request.message,
    }

    return StreamingResponse(
        _stream_intro(conversation_id, request.message, lc_messages, thread_id, is_first_message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_intro(
    conversation_id: int,
    question: str,
    history: list,
    thread_id: str,
    is_first_message: bool,
):
    """
    Run the graph from the beginning.
    Streams intro_node tokens, then detects the interrupt at clarify_node.
    """
    config       = {"configurable": {"thread_id": thread_id}}
    intro_text   = ""
    steps_seen   = []
    current_node = None

    try:
        async for event in agent_graph.astream_events(
            {
                "messages":    history,
                "question":    question,
                "intro_text":  "",
                "user_choice": "",
                "answer":      "",
            },
            config=config,
            version="v2",
        ):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            if node and node != current_node and node in STEP_LABELS:
                current_node = node
                label = STEP_LABELS[node]
                steps_seen.append(label)
                yield f"data: {json.dumps({'type': 'step', 'step': label})}\n\n"

            # Stream tokens only from the intro node
            if kind == "on_chat_model_stream" and node == "intro":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    intro_text += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        # ── After stream ends: was the graph interrupted? ────────────────────
        # get_state returns the current checkpoint. If graph is paused,
        # state.next is non-empty and state.tasks[0].interrupts has the value.
        graph_state = agent_graph.get_state(config)

        if graph_state.next:
            # Graph is paused at clarify_node — forward the interrupt payload
            interrupt_data = graph_state.tasks[0].interrupts[0].value

            # Store intro_text so the resume handler can save it to DB
            active_threads[conversation_id]["intro_text"] = intro_text
            active_threads[conversation_id]["steps"]      = steps_seen

            # Save the intro as a partial assistant message in the DB
            async with AsyncSessionLocal() as session:
                intro_msg = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=intro_text,
                    agent_steps=steps_seen + ["⏸ Waiting for your choice..."],
                )
                session.add(intro_msg)
                if is_first_message:
                    conv = await session.get(Conversation, conversation_id)
                    if conv and conv.title == "New Conversation":
                        conv.title = question[:60] + ("..." if len(question) > 60 else "")
                await session.commit()

            yield f"data: {json.dumps({'type': 'interrupt', 'question': interrupt_data['question'], 'options': interrupt_data['options']})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ── /resume ───────────────────────────────────────────────────────────────────

@router.post("/{conversation_id}/resume")
async def resume(
    conversation_id: int,
    request: ResumeRequest,
    db: AsyncSession = Depends(get_db),
):
    thread_info = active_threads.get(conversation_id)
    if not thread_info:
        raise HTTPException(status_code=400, detail="No active interrupt for this conversation")

    return StreamingResponse(
        _stream_resume(conversation_id, request.choice, thread_info),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_resume(conversation_id: int, choice: str, thread_info: dict):
    """
    Resume the paused graph by passing the user's choice via Command(resume=...).

    LangGraph restores the saved state, clarify_node returns {"user_choice": choice},
    and the graph continues to respond_node which streams the final answer.
    """
    config        = {"configurable": {"thread_id": thread_info["thread_id"]}}
    full_response = ""
    steps_seen    = []
    current_node  = None

    try:
        async for event in agent_graph.astream_events(
            Command(resume=choice),   # ← this is how you resume an interrupted graph
            config=config,
            version="v2",
        ):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            if node and node != current_node and node in STEP_LABELS:
                current_node = node
                label = STEP_LABELS[node]
                steps_seen.append(label)
                yield f"data: {json.dumps({'type': 'step', 'step': label})}\n\n"

            # Stream tokens from respond_node
            if kind == "on_chat_model_stream" and node == "respond":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        # Persist the final answer
        async with AsyncSessionLocal() as session:
            answer_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                agent_steps=[f"✓ {choice}"] + steps_seen,
            )
            session.add(answer_msg)
            await session.commit()

        # Clean up — this thread is done
        active_threads.pop(conversation_id, None)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
