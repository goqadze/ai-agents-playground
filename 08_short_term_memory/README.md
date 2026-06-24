# 08 — Short-Term Memory

Demonstrates how LangGraph's `MemorySaver` gives an agent memory within a conversation session.

## The concept

```
No memory (default):
  Turn 1: "My name is Alice"  → agent replies
  Turn 2: "What is my name?"  → agent: "I don't know your name"

With MemorySaver + thread_id:
  Turn 1: "My name is Alice"  → agent replies, saves state
  Turn 2: "What is my name?"  → agent loads saved state → "Your name is Alice"
```

Two things enable it:

| Thing | Role |
|---|---|
| `MemorySaver` | Saves a snapshot of all messages after each turn (in RAM) |
| `thread_id` | Key used to look up the right snapshot before each turn |

## Run

```bash
cd 08_short_term_memory
uv run python main.py
```

## What the demo shows

| Scenario | What you see |
|---|---|
| A | Same thread across 4 turns — agent recalls name and hobby |
| B | Two different threads — thread-2 has no idea what was said in thread-1 |
| C | Peek inside MemorySaver — message count grows by 2 each turn |
| D | Interactive chat — type `new` to start a fresh thread |
