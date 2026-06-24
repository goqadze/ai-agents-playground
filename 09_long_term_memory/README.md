# 09 — Long-Term Memory

Demonstrates the difference between short-term and long-term memory in LangGraph.

## Short-term vs Long-term

| | Short-term (08) | Long-term (09) |
|---|---|---|
| Storage | RAM (`MemorySaver`) | PostgreSQL |
| Survives restart? | No | Yes |
| Scope | One `thread_id` | Any thread, any session |
| What's stored | All messages | Only facts the agent decides to save |
| Managed by | LangGraph automatically | `@tool` functions the agent calls |

## How it works

Two Postgres tables:

```
langgraph_checkpoints     ← PostgresSaver manages this
                            stores full conversation history per thread_id
                            same as MemorySaver but on disk

long_term_memories        ← our @tool functions manage this
    key      VARCHAR       e.g. "name", "job", "city"
    value    TEXT          e.g. "Alex", "data engineer", "Tbilisi"
    saved_at TIMESTAMPTZ
```

The agent has 4 tools: `save_memory`, `recall_all_memories`, `recall_memory`, `forget_memory`.

The system prompt tells it to call `recall_all_memories()` at the start of every
conversation and `save_memory()` whenever the user shares something important.

## Setup

### 1. Start Postgres

```bash
cd 09_long_term_memory
docker compose up -d
```

### 2. Add your API key

```bash
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY
```

### 3. Run

```bash
uv run python main.py
```

## What the demo shows

| Scenario | What you see |
|---|---|
| Session 1 (thread-A) | Introduce yourself — agent saves name, job, city, language to Postgres |
| Session 2 (thread-B) | Completely new thread — agent recalls all saved facts and greets you by name |
| Inspect | Raw SQL read of the `long_term_memories` table — proves it's really in the DB |
| Interactive | Chat freely — try restarting the script to see memories persist |
