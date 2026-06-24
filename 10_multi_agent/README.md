# 10 — Multi-Agent Research Pipeline

A fullstack application where **three specialized AI agents** collaborate
to research and write about any topic you give them.

## The Agents

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  🗂️  PLANNER                                                     │
│  Breaks the topic into 4 focused research questions             │
│  Output: ["What is X?", "How does X work?", ...]               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔍  RESEARCHER                                                  │
│  Answers each research question with detailed findings          │
│  Output: structured markdown with one section per question      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  ✍️  WRITER                                                      │
│  Synthesizes the research into a polished, engaging article     │
│  Output: final article in markdown                              │
└─────────────────────────────────────────────────────────────────┘
```

Each agent is a **separate LangGraph node** with its own system prompt
that defines its role, responsibilities, and output format.
They share a single `ResearchState` dict — each reads what it needs
and writes its output back.

## Why Multi-Agent?

| Single Agent | Multi-Agent |
|---|---|
| One system prompt wears all hats | Each agent is expert in one role |
| Long, complex prompt is hard to tune | Short, focused prompts are easier to improve |
| All-or-nothing if one step is wrong | Individual agents can be replaced or upgraded |
| Hard to observe intermediate steps | UI shows each agent's output live |

## Architecture

```
frontend  (React + Vite — port 3000)
    │  POST /api/research  { topic }
    │  ← SSE stream of events
    │
backend   (FastAPI — port 8000)
    │  astream_events()
    │
LangGraph (StateGraph)
    ├─ planner_node    → ChatOpenAI (gpt-4.1-nano)
    ├─ researcher_node → ChatOpenAI (gpt-4.1-nano)
    └─ writer_node     → ChatOpenAI (gpt-4.1-nano)
```

## SSE Event Protocol

Every agent broadcasts its progress via Server-Sent Events:

| Event | Payload | When |
|---|---|---|
| `agent_start` | `{ agent }` | Node begins |
| `token` | `{ agent, content }` | Each streamed token |
| `agent_done` | `{ agent, output }` | Node finishes; output = state update |
| `done` | — | Pipeline complete |
| `error` | `{ message }` | Something went wrong |

## Key LangGraph Concepts

**State** (`state.py`) — a `TypedDict` shared by all agents:
```python
class ResearchState(TypedDict):
    topic: str       # input
    plan: list[str]  # planner output
    research: str    # researcher output
    article: str     # writer output
```

**Nodes** (`nodes.py`) — each agent is an `async def` that reads from
and writes to the state:
```python
async def planner_node(state: ResearchState) -> dict:
    response = await llm.ainvoke([...])
    return {"plan": json.loads(response.content)}
```

**Graph** (`graph.py`) — wires the nodes in a pipeline:
```python
graph.add_edge(START,        "planner")
graph.add_edge("planner",    "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer",     END)
```

**Streaming** — `astream_events(version="v2")` emits token-level events
from every LLM call inside any node, tagged with `metadata["langgraph_node"]`.

## Setup & Run

### 1. Add your API key

```bash
cp backend/.env.example backend/.env
# edit backend/.env — set OPENAI_API_KEY
```

### 2. Start with Docker Compose

```bash
cd 10_multi_agent
docker compose --env-file backend/.env up --build
```

### 3. Open the app

```
http://localhost:3000
```

### Run backend locally (without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Production Extensions

In a real system you would add:

- **Web search tools** for the Researcher (Tavily, Exa, or DuckDuckGo) so it fetches live information
- **Supervisor agent** to dynamically route between agents based on the topic
- **Parallel research** — fan out to multiple Researcher agents simultaneously
- **Critique agent** — reviews the article and sends it back for revision
- **Persistent storage** — save articles to a database
