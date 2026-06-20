"""
Streaming chat endpoint.

Changes vs project 05:
  - STEP_LABELS updated to match the new graph nodes
  - Streams tokens from the "agent" node (was "respond")
  - Emits a new "tool" event type when a tool is called — the frontend
    shows these as green badges so you can see which tools the agent used.
"""

import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_core.messages import HumanMessage, AIMessage

from ..database import get_db, AsyncSessionLocal
from ..models import Conversation, Message
from ..schemas import ChatRequest
from ..agent.graph import agent_graph

router = APIRouter(prefix="/api/conversations", tags=["chat"])

# Human-readable labels for each graph node — shown in the UI as step badges
STEP_LABELS = {
    "classify":    "Classifying...",
    "analyze":     "Deep research...",
    "setup_agent": "Preparing agent...",
    "agent":       "Thinking...",
    "tools":       "Running tools...",
    "extract":     "Composing answer...",
}


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

    return StreamingResponse(
        _stream_agent(conversation_id, request.message, lc_messages, is_first_message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_agent(
    conversation_id: int,
    question: str,
    history: list,
    is_first_message: bool,
):
    full_response = ""
    steps_seen: list[str] = []   # node step labels (shown as grey badges)
    tools_used: list[str] = []   # tool names (shown as green badges)
    current_node = None

    try:
        async for event in agent_graph.astream_events(
            {
                "history":        history,
                "question":       question,
                "intent":         "",
                "analysis":       "",
                "agent_messages": [],
                "answer":         "",
            },
            version="v2",
        ):
            kind     = event["event"]
            metadata = event.get("metadata", {})
            node     = metadata.get("langgraph_node", "")

            # ── Step badge: emit when we enter a new graph node ──────────────
            if node and node != current_node and node in STEP_LABELS:
                current_node = node
                label = STEP_LABELS[node]
                steps_seen.append(label)
                yield f"data: {json.dumps({'type': 'step', 'step': label})}\n\n"

            # ── Tool badge: emit when a tool is about to be called ───────────
            # on_tool_start fires right before the tool function runs.
            # event["name"] is the tool function name (e.g. "calculator").
            if kind == "on_tool_start" and node == "tools":
                tool_name = event.get("name", "tool")
                tools_used.append(tool_name)
                yield f"data: {json.dumps({'type': 'tool', 'tool': tool_name})}\n\n"

            # ── Token streaming: only forward content from the agent node ────
            # When the LLM is streaming a TOOL CALL, chunk.content is "".
            # When it's streaming a FINAL ANSWER, chunk.content has text.
            # So filtering on chunk.content naturally excludes tool-call chunks.
            if kind == "on_chat_model_stream" and node == "agent":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        # ── Persist assistant message ────────────────────────────────────────
        async with AsyncSessionLocal() as session:
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                # Store steps + tool calls so they can be shown in history
                agent_steps=steps_seen + [f"🔧 {t}" for t in tools_used],
            )
            session.add(assistant_msg)

            if is_first_message:
                conv = await session.get(Conversation, conversation_id)
                if conv and conv.title == "New Conversation":
                    conv.title = question[:60] + ("..." if len(question) > 60 else "")

            await session.commit()

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
