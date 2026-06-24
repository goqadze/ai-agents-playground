# Agent ↔ LLM — The Invisible Exchange

When you send a message like *"My name is Alex"*, you see one reply.
Behind the scenes there is a full multi-step conversation between the
**agent framework** and the **LLM** that you never see.
This document makes that invisible exchange visible.

---

## The Cast

```
YOU          — type a message in the terminal
AGENT        — LangGraph's create_react_agent loop (your Python process)
LLM          — OpenAI gpt-4.1-nano (a remote API call)
TOOLS        — your @tool Python functions (save_memory, recall_all_memories, ...)
POSTGRES     — the database where memories are stored
```

---

## Example: "Hi! My name is Alex and I live in Tbilisi."

Here is the complete invisible exchange for that single user message.

```
YOU ──────────────────────────────────────────────────────────────────────────
  "Hi! My name is Alex and I live in Tbilisi."

AGENT (step 1) ───────────────────────────────────────────────────────────────
  Builds a messages list and sends it to the LLM API:

  [
    SystemMessage: "You are a helpful assistant with long-term memory.
                    At the START of every conversation, call recall_all_memories()…",

    HumanMessage:  "Hi! My name is Alex and I live in Tbilisi."
  ]

LLM (response 1) ─────────────────────────────────────────────────────────────
  Does NOT reply with text yet.
  Decides it must check existing memories first.
  Returns a TOOL CALL (not text):

  AIMessage(
    content    = "",                        ← empty — no text yet
    tool_calls = [
      { id: "call_001", name: "recall_all_memories", args: {} }
    ]
  )

AGENT (step 2) ───────────────────────────────────────────────────────────────
  Sees tool_calls is not empty.
  Executes the Python function:

  recall_all_memories()
      │
      └─▶ SELECT key, value FROM long_term_memories ORDER BY saved_at
          ← returns: "No long-term memories stored yet — this is a fresh start."

  Wraps the result in a ToolMessage and appends it to the messages list.
  Messages list is now:

  [
    SystemMessage:  "…"
    HumanMessage:   "Hi! My name is Alex and I live in Tbilisi."
    AIMessage:      tool_calls=[recall_all_memories]        ← LLM's decision
    ToolMessage:    "No long-term memories stored yet…"     ← tool's result
  ]

  Sends the updated list back to the LLM.

LLM (response 2) ─────────────────────────────────────────────────────────────
  Reads all 4 messages — including the tool result.
  Decides it now knows it's a fresh start AND the user just shared their name/city.
  Returns TWO tool calls in one shot:

  AIMessage(
    content    = "",
    tool_calls = [
      { id: "call_002", name: "save_memory", args: { key: "name",  value: "Alex"    } },
      { id: "call_003", name: "save_memory", args: { key: "city",  value: "Tbilisi" } }
    ]
  )

AGENT (step 3) ───────────────────────────────────────────────────────────────
  Executes both tool calls:

  save_memory("name", "Alex")
      │
      └─▶ INSERT INTO long_term_memories (key, value) VALUES ('name', 'Alex')
          ← POSTGRES stores the row

  save_memory("city", "Tbilisi")
      │
      └─▶ INSERT INTO long_term_memories (key, value) VALUES ('city', 'Tbilisi')
          ← POSTGRES stores the row

  Appends two ToolMessages to the list:

  [
    SystemMessage:  "…"
    HumanMessage:   "Hi! My name is Alex and I live in Tbilisi."
    AIMessage:      tool_calls=[recall_all_memories]
    ToolMessage:    "No long-term memories stored yet…"
    AIMessage:      tool_calls=[save_memory(name), save_memory(city)]
    ToolMessage:    "✓ Saved: name = Alex"
    ToolMessage:    "✓ Saved: city = Tbilisi"
  ]

  Sends the full list back to the LLM again.

LLM (response 3) ─────────────────────────────────────────────────────────────
  Reads all 7 messages.
  Sees both saves were successful.
  No more tools needed — generates a final text reply:

  AIMessage(
    content    = "Hello, Alex! It's great to meet you. How can I assist you today?",
    tool_calls = []     ← empty — this is the final answer
  )

AGENT (step 4) ───────────────────────────────────────────────────────────────
  tool_calls is empty → loop ends.
  Returns the last AIMessage to you.

YOU ──────────────────────────────────────────────────────────────────────────
  See: "Hello, Alex! It's great to meet you. How can I assist you today?"
```

---

## The Full Loop — Diagram

```
                        ┌─────────────────────────────────┐
                        │         AGENT LOOP               │
                        │   (create_react_agent internals) │
YOU                     │                                  │              LLM
─────                   │                                  │             ─────
                        │                                  │
"Hi, I'm Alex"          │                                  │
      │                 │                                  │
      ▼                 │                                  │
      ┌─────────────────┤                                  │
      │ build messages  │                                  │
      │ [System, Human] ├──────── API call ───────────────▶│
      └─────────────────┤                                  │ "I should check
                        │                                  │  memories first"
                        │◀─────── tool_call ───────────────┤
                        │         recall_all_memories()    │
                        │                                  │
      ┌─────────────────┤                                  │
      │ run Python fn   │                                  │
      │ → query Postgres│                                  │
      │ → "no memories" │                                  │
      └────────┬────────┤                                  │
               │        │                                  │
               │ append ToolMessage                        │
               ▼        │                                  │
      ┌─────────────────┤                                  │
      │ send updated    ├──────── API call ───────────────▶│
      │ messages list   │                                  │ "New user, I should
      └─────────────────┤                                  │  save name + city"
                        │◀─────── tool_calls ──────────────┤
                        │  save_memory(name=Alex)          │
                        │  save_memory(city=Tbilisi)       │
                        │                                  │
      ┌─────────────────┤                                  │
      │ run both tools  │                                  │
      │ → INSERT rows   │                                  │
      │ → "✓ Saved…"    │                                  │
      └────────┬────────┤                                  │
               │        │                                  │
               │ append ToolMessages                       │
               ▼        │                                  │
      ┌─────────────────┤                                  │
      │ send updated    ├──────── API call ───────────────▶│
      │ messages list   │                                  │ "All done,
      └─────────────────┤                                  │  reply to user"
                        │◀─────── text reply ──────────────┤
                        │  "Hello, Alex!..."               │
                        │  tool_calls = []  ← STOP         │
                        │                                  │
      ┌─────────────────┤                                  │
      │ loop ends       │                                  │
      └────────┬────────┘                                  │
               │
               ▼
"Hello, Alex! It's great to meet you."
```

---

## The Messages List Grows With Every Step

This is the exact list that gets sent to the LLM on the final (3rd) API call.
Every message is part of the conversation context.

```
Index   Type            Content
──────  ──────────────  ───────────────────────────────────────────────────────
  0     SystemMessage   "You are a helpful assistant with long-term memory…"
  1     HumanMessage    "Hi! My name is Alex and I live in Tbilisi."
  2     AIMessage       content="" tool_calls=[recall_all_memories()]
  3     ToolMessage     "No long-term memories stored yet — this is a fresh start."
  4     AIMessage       content="" tool_calls=[save_memory(name), save_memory(city)]
  5     ToolMessage     "✓ Saved to long-term memory: name = Alex"
  6     ToolMessage     "✓ Saved to long-term memory: city = Tbilisi"
  7     AIMessage       content="Hello, Alex! It's great to meet you."  ← FINAL
```

The LLM never "remembers" anything on its own.
It is **stateless** — it only knows what is in the messages list right now.
The agent is responsible for building and maintaining that list.

---

## Second Session — "Hello, do you know who I am?" (thread-B)

This is a brand new thread — the messages list starts completely empty.
But Postgres still has the rows from Session 1.

```
YOU ──────────────────────────────────────────────────────────────────────────
  "Hello, do you know who I am?"   (thread-B — zero history)

AGENT ────────────────────────────────────────────────────────────────────────
  Loads checkpoint for thread-B from PostgresSaver.
  No messages found — fresh start.

  Sends to LLM:
  [
    SystemMessage: "…call recall_all_memories() at the START…"
    HumanMessage:  "Hello, do you know who I am?"
  ]

LLM ──────────────────────────────────────────────────────────────────────────
  Sees the system prompt instruction: "call recall_all_memories() at start".
  Returns tool call:

  AIMessage( tool_calls=[{ name: "recall_all_memories", args: {} }] )

AGENT ────────────────────────────────────────────────────────────────────────
  Runs recall_all_memories()
      │
      └─▶ SELECT key, value FROM long_term_memories ORDER BY saved_at
          ← returns:
             "Long-term memories:
                name: Alex
                city: Tbilisi
                job:  data engineer
                …"

  Appends ToolMessage and sends full list back to LLM:
  [
    SystemMessage
    HumanMessage:  "Hello, do you know who I am?"
    AIMessage:     tool_calls=[recall_all_memories]
    ToolMessage:   "Long-term memories: name: Alex, city: Tbilisi, job: …"
  ]

LLM ──────────────────────────────────────────────────────────────────────────
  Reads the memories. Knows exactly who the user is.
  No more tools needed.

  AIMessage(
    content    = "Hello, Alex! I remember you're from Tbilisi, a data engineer
                  who loves coffee and Python. How can I assist you today?",
    tool_calls = []
  )

YOU ──────────────────────────────────────────────────────────────────────────
  See: "Hello, Alex! I remember you're from Tbilisi…"
```

The agent made **3 API calls** to OpenAI just to answer that one message.
Each one you never saw.

---

## Why the LLM Returns Tool Calls Instead of Text

The LLM does not call tools directly — it cannot.
It is just a text model. What it can do is return a special JSON structure
instead of a regular text reply.

```
Normal text reply:
  { "role": "assistant", "content": "Hello, Alex!" }

Tool call reply:
  {
    "role":       "assistant",
    "content":    null,
    "tool_calls": [
      {
        "id":       "call_001",
        "type":     "function",
        "function": {
          "name":      "save_memory",
          "arguments": "{\"key\": \"name\", \"value\": \"Alex\"}"
        }
      }
    ]
  }
```

The **agent** (your Python process) is the one that actually:
1. Reads that JSON
2. Finds the matching Python function by name
3. Calls it with the given arguments
4. Sends the return value back in the next API call

The LLM never touches Postgres. It only sees text.

---

## What the LLM Knows About Each Tool

When you write:
```python
@tool
def save_memory(key: str, value: str) -> str:
    """Save an important fact about the user to long-term memory..."""
```

`bind_tools()` converts that into a JSON schema that gets sent to OpenAI
in every API request:

```json
{
  "type": "function",
  "function": {
    "name": "save_memory",
    "description": "Save an important fact about the user to long-term memory...",
    "parameters": {
      "type": "object",
      "properties": {
        "key":   { "type": "string" },
        "value": { "type": "string" }
      },
      "required": ["key", "value"]
    }
  }
}
```

The LLM reads the `"description"` field to decide **when** to call the tool.
It reads `"parameters"` to know **what arguments** to pass.
This is why docstrings matter — they are the LLM's instruction manual.

---

## Summary Table

```
What you type           What actually happens (invisible)
──────────────────────  ─────────────────────────────────────────────────────
"Hi, my name is Alex"   1. LLM call 1  → tool_call: recall_all_memories
                        2. Python runs recall_all_memories() → queries Postgres
                        3. LLM call 2  → tool_calls: save_memory(name, city)
                        4. Python runs save_memory() twice  → INSERTs to Postgres
                        5. LLM call 3  → final text reply
                        ──────────────────────────────────────
                        3 API calls, 3 Postgres queries, 8 messages exchanged

"Hello, do you know     1. LLM call 1  → tool_call: recall_all_memories
 who I am?"             2. Python runs recall_all_memories() → queries Postgres
                        3. LLM call 2  → final text reply
                        ──────────────────────────────────────
                        2 API calls, 1 Postgres query, 4 messages exchanged
```

One user message can trigger many LLM calls and database queries.
All of it happens in milliseconds, invisibly, before you see the reply.
