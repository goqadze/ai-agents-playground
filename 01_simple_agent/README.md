# LangGraph Simple Agent

## What is LangGraph?

LangGraph builds AI agents as **graphs** — nodes connected by edges, sharing a common state.

| Concept | Plain English |
|---------|--------------|
| **State** | A shared memory bag (dict) passed between every node |
| **Node** | A function that does one job — call LLM, run a tool, check a condition |
| **Edge** | A wire connecting two nodes. Fixed ("always go here") or conditional ("go here *if* …") |
| **Graph** | The whole thing compiled into a runnable app |

---

## Setup

```bash
# 1. Install (already done)
uv add langgraph langchain-anthropic

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run
uv run python simple_agent.py
```

---

## Flow

```
User message
     │
     ▼
┌─────────────┐
│   chatbot   │  ← LLM decides: answer directly, or call a tool?
└──────┬──────┘
       ├── no tool needed ──► END
       ▼
┌─────────────┐
│    tools    │  ← runs the tool (weather / calculator)
└──────┬──────┘
       └──────────────────► back to chatbot (LLM sees result, writes final reply)
```

Try these prompts:
- `What is the weather in Tokyo?`
- `What is 123 * 456?`
- `Weather in London and what is 10 + 20?`

---

## Next steps

1. **Memory** — `MemorySaver` checkpointer for cross-session history
2. **Multi-agent** — supervisor node that delegates to specialist sub-agents
3. **Streaming** — `graph.stream()` to print tokens as they arrive
4. **Tracing** — set `LANGCHAIN_TRACING_V2=true` for visual run traces in LangSmith
