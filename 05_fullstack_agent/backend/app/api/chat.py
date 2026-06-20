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

STEP_LABELS = {
    "classify": "Analyzing question...",
    "analyze": "Researching deeply...",
    "respond": "Writing response...",
}


@router.post("/{conversation_id}/chat")
async def chat(
    conversation_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    # Load conversation history
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

    # Persist user message before streaming
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
    steps_seen: list[str] = []
    current_node = None

    try:
        async for event in agent_graph.astream_events(
            {
                "question": question,
                "messages": history,
                "intent": "",
                "analysis": "",
                "answer": "",
            },
            version="v2",
        ):
            kind = event["event"]
            metadata = event.get("metadata", {})
            node = metadata.get("langgraph_node", "")

            # Emit a step event when we enter a new known node
            if node and node != current_node and node in STEP_LABELS:
                current_node = node
                label = STEP_LABELS[node]
                steps_seen.append(label)
                yield f"data: {json.dumps({'type': 'step', 'step': label})}\n\n"

            # Stream response tokens only from the respond node
            if kind == "on_chat_model_stream" and node == "respond":
                chunk = event["data"]["chunk"].content
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        # Persist the assistant message
        async with AsyncSessionLocal() as session:
            assistant_msg = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=full_response,
                agent_steps=steps_seen,
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
