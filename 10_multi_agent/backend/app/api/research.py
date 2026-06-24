"""
Research API endpoint.

POST /api/research
  Body : { "topic": "quantum computing" }
  Returns: Server-Sent Events stream

SSE event types:
  { "type": "agent_start", "agent": "planner|researcher|writer" }
      ↑ fires when a node begins executing

  { "type": "token", "agent": "...", "content": "..." }
      ↑ fires for every streaming token from the LLM
      ↑ lets the frontend show live text as the agent "thinks"

  { "type": "agent_done", "agent": "...", "output": {...} }
      ↑ fires when a node finishes
      ↑ "output" contains the node's state update (e.g. {"plan": [...]} for planner)

  { "type": "done" }
      ↑ pipeline complete

  { "type": "error", "message": "..." }
      ↑ something went wrong
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.graph import research_graph

router = APIRouter()

# Nodes we care about — used to filter astream_events output
AGENT_NODES = {"planner", "researcher", "writer"}


class ResearchRequest(BaseModel):
    topic: str


def sse(payload: dict) -> str:
    """Format a dict as one SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


async def _stream(topic: str):
    """
    Run the graph and yield SSE events.

    astream_events(version="v2") emits a rich event stream including:
      - on_chain_start / on_chain_end  → node lifecycle
      - on_chat_model_stream           → individual LLM tokens
      - on_tool_start / on_tool_end    → if agents had tools

    We filter to the three agent nodes and translate graph events
    into a frontend-friendly format.
    """
    initial_state = {
        "topic": topic,
        "plan": [],
        "research": "",
        "article": "",
    }

    try:
        async for event in research_graph.astream_events(initial_state, version="v2"):
            evt_type = event["event"]
            node_name = event.get("name", "")
            metadata_node = event.get("metadata", {}).get("langgraph_node", "")

            # Node started
            if evt_type == "on_chain_start" and node_name in AGENT_NODES:
                yield sse({"type": "agent_start", "agent": node_name})

            # Streaming token from LLM inside a node
            elif evt_type == "on_chat_model_stream" and metadata_node in AGENT_NODES:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield sse({"type": "token", "agent": metadata_node, "content": chunk.content})

            # Node finished — emit its output so frontend can display structured data
            elif evt_type == "on_chain_end" and node_name in AGENT_NODES:
                output = event["data"].get("output", {})
                yield sse({"type": "agent_done", "agent": node_name, "output": output})

        yield sse({"type": "done"})

    except Exception as exc:
        yield sse({"type": "error", "message": str(exc)})


@router.post("/research")
async def research(req: ResearchRequest):
    return StreamingResponse(
        _stream(req.topic),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx response buffering
            "Connection": "keep-alive",
        },
    )
