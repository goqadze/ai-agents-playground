# End-to-End Flow — HITL Chat (07_hitl_agent)

This document traces exactly what happens from the moment the user types a message
to the moment the final answer appears on screen — including the human-in-the-loop
pause in the middle.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FULL CONVERSATION FLOW                            │
│                                                                             │
│   BROWSER                  FASTAPI                   LANGGRAPH              │
│   ──────────               ───────                   ─────────              │
│                                                                             │
│   User types               POST /chat                                       │
│   & sends     ──────────▶  saves user msg to DB                             │
│                            StreamingResponse  ──────▶  graph starts         │
│                                                         intro_node (LLM)    │
│   "Great question..."  ◀────────────────────────────── streams tokens       │
│                                                         clarify_node        │
│                                                         interrupt() called  │
│   Option buttons       ◀────────────────────────────── graph PAUSES         │
│   appear                                                state saved to      │
│                                                         MemorySaver         │
│                                                                             │
│   User clicks           POST /resume                                        │
│   "Step-by-step"  ──────────────────▶  Command(resume=choice)               │
│                                                         graph RESUMES       │
│                                                         respond_node (LLM)  │
│   Final answer     ◀────────────────────────────────── streams tokens       │
│   appears                                               graph END           │
│                            saves answer to DB                               │
│                            done event         ──────▶                       │
│   Messages reload  ◀──────────────────                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — User Sends a Message

### 1.1 User Input

**File:** `frontend/src/components/ChatWindow.tsx`

The user types in the textarea and presses Enter.

```
<textarea onKeyDown={handleKeyDown}>
        │
        │ Enter pressed (not Shift+Enter)
        ▼
handleSubmit()
        │
        ▼
onSend(text)   ← prop passed down from App.tsx
```

### 1.2 App Prepares State

**File:** `frontend/src/App.tsx` → `handleSend()`

```
handleSend("What is recursion?")
        │
        ├─ 1. Add user message to UI immediately (optimistic — no waiting)
        │      setMessages([...prev, { role: "user", content: "What is recursion?" }])
        │
        ├─ 2. Set streaming state to empty
        │      setStreaming({ content: "", steps: [], currentStep: null, interrupt: null })
        │
        ├─ 3. setIsLoading(true)
        │
        └─ 4. call streamChat(activeId, text)  ──▶  Phase 2
```

---

## Phase 2 — HTTP Request to Backend

**File:** `frontend/src/api/client.ts` → `streamChat()`

```typescript
async function* streamChat(conversationId, message)
```

This is an **async generator** — it opens the HTTP connection and yields
one parsed event object per line as bytes arrive from the server.

```
streamChat(conversationId, "What is recursion?")
        │
        ├─ fetch() POST /api/conversations/{id}/chat
        │    body:    { message: "What is recursion?" }
        │    headers: Content-Type: application/json
        │
        ├─ waits for response headers (connection stays open)
        │
        └─ opens res.body.getReader()
                │
                │  while(true) loop:
                │    reader.read()  ← blocks until next bytes arrive
                │    TextDecoder   ← bytes → text
                │    split on \n   ← find complete SSE lines
                │    each line:    data: {"type":"token","content":"Great"}
                │    strip "data: " → parse JSON → yield object
                ▼
           yields events one by one to App.tsx
```

> **What is SSE (Server-Sent Events)?**
> The server keeps the HTTP connection open and sends lines like:
> `data: {"type":"token","content":"Hello"}\n\n`
> The client reads them as they arrive — no polling, no WebSockets needed.

---

## Phase 3 — FastAPI Receives the Request

**File:** `backend/app/api/chat.py` → `chat()` endpoint

```
POST /api/conversations/{id}/chat
        │
        ▼
async def chat(conversation_id, request, db):
        │
        ├─ 1. Load conversation history from PostgreSQL
        │      SELECT messages WHERE conversation_id = {id} ORDER BY created_at
        │      Convert to LangChain message objects:
        │         role "user"      → HumanMessage(content="...")
        │         role "assistant" → AIMessage(content="...")
        │
        ├─ 2. Save the incoming user message to DB immediately
        │      (before anything else — ensures it's persisted even if graph crashes)
        │
        ├─ 3. Generate a fresh thread_id  ← uuid4()
        │      Store in active_threads[conversation_id]
        │      This ID links the /chat and /resume calls together
        │
        └─ 4. return StreamingResponse(
                  _stream_intro(...),         ← generator, not called yet
                  media_type="text/event-stream"
              )
              │
              FastAPI sends response headers to browser immediately
              then starts consuming _stream_intro() chunk by chunk
```

> **Why save before streaming?**
> The user message is persisted before the LLM even starts.
> If the server crashes mid-stream, the user message is still in the DB.

---

## Phase 4 — Graph Starts Running

**File:** `backend/app/api/chat.py` → `_stream_intro()`

This generator is where the LangGraph graph actually starts.

```
_stream_intro(conversation_id, question, history, thread_id, ...)
        │
        └─ agent_graph.astream_events(
               {
                   "messages":    [...history],        ← DB conversation history
                   "question":    "What is recursion?",
                   "intro_text":  "",
                   "user_choice": "",
                   "answer":      "",
               },
               config={"configurable": {"thread_id": thread_id}},
               version="v2",
           )
           │
           │  LangGraph begins executing nodes
           │  firing events as each step runs
           ▼
       async for event in ...   ← one event per graph action
```

---

## Phase 5 — LangGraph Executes the Graph

**Files:** `backend/app/agent/graph.py`, `nodes.py`

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GRAPH EXECUTION                              │
│                                                                     │
│   START                                                             │
│     │                                                               │
│     ▼                                                               │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  intro_node                                              │     │
│   │                                                          │     │
│   │  llm.ainvoke([                                           │     │
│   │    SystemMessage("Write ONE sentence acknowledging..."), │     │
│   │    ...history,                                           │     │
│   │    HumanMessage("What is recursion?")                    │     │
│   │  ])                                                      │     │
│   │                                                          │     │
│   │  → fires on_chat_model_stream events as tokens arrive    │     │
│   │  → returns {"intro_text": "Great question! Let me..."}   │     │
│   └──────────────────────────────────────────────────────────┘     │
│     │                                                               │
│     ▼                                                               │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  clarify_node                                            │     │
│   │                                                          │     │
│   │  choice = interrupt({                                    │     │
│   │    "question": "How would you like me to answer?",       │     │
│   │    "options":  ["Concise summary",                       │     │
│   │                 "Detailed explanation",                   │     │
│   │                 "Step-by-step walkthrough",               │     │
│   │                 "Examples and analogies"]                │     │
│   │  })                           ▲                          │     │
│   │                               │                          │     │
│   │  interrupt() does:            │ GRAPH PAUSES HERE        │     │
│   │    1. saves full state ───────┘                          │     │
│   │       to MemorySaver                                     │     │
│   │    2. raises NodeInterrupt internally                    │     │
│   │    3. astream_events loop ends cleanly                   │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│     ⏸  EXECUTION PAUSED — waiting for Command(resume=...)          │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  respond_node          (runs AFTER resume)               │     │
│   │                                                          │     │
│   │  instruction = RESPONSE_FORMATS[state["user_choice"]]    │     │
│   │                                                          │     │
│   │  llm.ainvoke([                                           │     │
│   │    SystemMessage(f"...{instruction}..."),                │     │
│   │    ...history,                                           │     │
│   │    HumanMessage("What is recursion?")                    │     │
│   │  ])                                                      │     │
│   │                                                          │     │
│   │  → fires on_chat_model_stream events as tokens arrive    │     │
│   │  → returns {"answer": "Step 1: A function calls..."}     │     │
│   └──────────────────────────────────────────────────────────┘     │
│     │                                                               │
│    END                                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 6 — Intro Tokens Stream to the Browser

Back in `_stream_intro()`, the `async for` loop processes graph events:

```
Graph event received                    SSE line sent to browser
────────────────────────────────────────────────────────────────────
intro node starts              →   data: {"type":"step","step":"Reading your question..."}
LLM streams "Great"            →   data: {"type":"token","content":"Great"}
LLM streams " question"        →   data: {"type":"token","content":" question"}
LLM streams "! Let me..."      →   data: {"type":"token","content":"! Let me..."}
clarify node starts            →   data: {"type":"step","step":"Preparing options..."}
interrupt() called             →   [astream_events loop ends]
```

After the loop ends, `_stream_intro()` checks the graph state:

```python
graph_state = agent_graph.get_state(config)

if graph_state.next:      # ← non-empty = graph was interrupted
    interrupt_data = graph_state.tasks[0].interrupts[0].value
    # Save intro text as partial assistant message to DB
    # Then send the interrupt event:
    yield 'data: {"type":"interrupt","question":"How would you like...","options":[...]}'
```

---

## Phase 7 — Frontend Receives the Interrupt

**File:** `frontend/src/App.tsx` → `consumeStream()`

```
for await (const event of streamChat(...)) {

    event.type === "step"
        → setStreaming({ currentStep: "Reading your question..." })
        → grey spinner badge appears

    event.type === "token"
        → setStreaming({ content: prev.content + "Great question!..." })
        → text appears word by word

    event.type === "interrupt"    ◀─── THIS IS THE KEY EVENT
        → setStreaming({
              interrupt: {
                question: "How would you like me to answer?",
                options:  ["Concise summary", "Detailed explanation", ...]
              }
          })
        → setIsLoading(false)     ← re-enable interaction
        → return                  ← STOP consuming the stream
}
```

The textarea is disabled and option buttons appear below the streamed text:

```
┌─────────────────────────────────────────────────────────┐
│  Assistant bubble                                       │
│                                                         │
│  ● Reading your question...  ● Preparing options...     │  ← step badges
│                                                         │
│  Great question! Let me tailor my answer to             │  ← streamed intro
│  exactly what you need.                                 │
│                                                         │
│  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄      │
│  How would you like me to answer?                       │  ← interrupt question
│                                                         │
│  ┌─────────────────────────┐                           │
│  │  Concise summary        │                           │  ← option buttons
│  └─────────────────────────┘                           │
│  ┌─────────────────────────┐                           │
│  │  Detailed explanation   │                           │
│  └─────────────────────────┘                           │
│  ┌─────────────────────────┐                           │
│  │  Step-by-step walkthrough│                          │
│  └─────────────────────────┘                           │
│  ┌─────────────────────────┐                           │
│  │  Examples and analogies │                           │
│  └─────────────────────────┘                           │
└─────────────────────────────────────────────────────────┘

[ Pick an option above to continue...      ]  ← input disabled
```

---

## Phase 8 — User Picks an Option

**File:** `frontend/src/App.tsx` → `handleResume()`

```
User clicks "Step-by-step walkthrough"
        │
        ▼
handleResume("Step-by-step walkthrough")
        │
        ├─ 1. Update streaming UI — replace interrupt buttons with a choice badge
        │      setStreaming({
        │          interrupt: null,
        │          steps: [...prev.steps, "✓ Step-by-step walkthrough"]
        │      })
        │
        ├─ 2. setIsLoading(true)
        │
        └─ 3. call resumeChat(activeId, "Step-by-step walkthrough")  ──▶  Phase 9
```

---

## Phase 9 — Resume Request to Backend

**File:** `frontend/src/api/client.ts` → `resumeChat()`

```
resumeChat(conversationId, "Step-by-step walkthrough")
        │
        └─ fetch() POST /api/conversations/{id}/resume
               body: { choice: "Step-by-step walkthrough" }
               │
               same SSE reading loop as streamChat
               yields events one by one
```

---

## Phase 10 — Backend Resumes the Graph

**File:** `backend/app/api/chat.py` → `resume()` + `_stream_resume()`

```
POST /api/conversations/{id}/resume
        │
        ▼
async def resume(conversation_id, request, db):
        │
        ├─ look up active_threads[conversation_id]
        │  → {"thread_id": "abc-123", ...}   ← same thread_id from /chat
        │
        └─ return StreamingResponse(_stream_resume(...))


_stream_resume(conversation_id, "Step-by-step walkthrough", thread_info)
        │
        └─ agent_graph.astream_events(
               Command(resume="Step-by-step walkthrough"),  ← the key
               config={"configurable": {"thread_id": "abc-123"}},
           )
```

`Command(resume=...)` is how you pass a value back into a paused graph.
LangGraph looks up thread `"abc-123"` in MemorySaver, restores the exact
state it saved at the interrupt point, and `interrupt()` returns
`"Step-by-step walkthrough"` as if it was a normal function call.

```
clarify_node continues:
    choice = "Step-by-step walkthrough"   ← returned by interrupt()
    return {"user_choice": "Step-by-step walkthrough"}

graph moves to respond_node:
    instruction = "Break the answer into clear numbered steps."
    llm.ainvoke([SystemMessage("...Break into steps..."), ..., HumanMessage("What is recursion?")])
    → streams the final answer token by token
```

---

## Phase 11 — Final Answer Streams to Browser

```
Graph event received                    SSE line sent to browser
────────────────────────────────────────────────────────────────────
respond node starts            →   data: {"type":"step","step":"Writing your answer..."}
LLM streams "Step 1:"          →   data: {"type":"token","content":"Step 1:"}
LLM streams " A function"      →   data: {"type":"token","content":" A function"}
...more tokens...
graph reaches END              →   [astream_events loop ends]
                               →   [assistant message saved to DB]
                               →   data: {"type":"done"}
```

---

## Phase 12 — Frontend Finalises

**File:** `frontend/src/App.tsx` → `consumeStream()`

```
event.type === "step"
    → "Writing your answer..." badge appears

event.type === "token"
    → text builds up in the streaming bubble word by word

event.type === "done"
    → setStreaming(null)           ← remove streaming bubble
    → getConversation(activeId)   ← reload messages from DB
    → setMessages(data.messages)  ← replace with persisted messages
    → update sidebar title        ← shows first 60 chars of question
```

The conversation now shows three persisted messages:
1. **User:** "What is recursion?"
2. **Assistant (intro):** "Great question! Let me tailor my answer..." + step badges
3. **Assistant (answer):** Full step-by-step explanation + "✓ Step-by-step walkthrough" badge

---

## The Role of thread_id and MemorySaver

```
/chat creates:   thread_id = "abc-123"
                 active_threads[conv_id] = { thread_id: "abc-123", ... }

MemorySaver saves at interrupt():
  key:   "abc-123"
  value: { messages: [...], question: "...", intro_text: "...", user_choice: "", answer: "" }

/resume reads:   thread_id = active_threads[conv_id]["thread_id"]  →  "abc-123"
                 Command(resume=choice) + config thread_id "abc-123"

MemorySaver restores:
  finds "abc-123" → restores full state → graph continues
```

Without the **checkpointer** (`graph.compile(checkpointer=memory)`), `interrupt()` would have
nowhere to save state and `Command(resume=...)` would fail. The checkpointer is what
makes Human-in-the-Loop possible.

---

## Complete File Map

```
07_hitl_agent/
│
├── frontend/src/
│   ├── components/
│   │   ├── ChatWindow.tsx     Phase 1  — user input + renders option buttons
│   │   ├── MessageBubble.tsx           — renders persisted messages with badges
│   │   └── Sidebar.tsx                — conversation list
│   ├── App.tsx                Phase 2, 7, 8  — handleSend, consumeStream, handleResume
│   └── api/
│       └── client.ts          Phase 2, 9  — streamChat(), resumeChat(), readSSE()
│
└── backend/app/
    ├── api/
    │   └── chat.py            Phase 3, 4, 10  — /chat endpoint, /resume endpoint
    └── agent/
        ├── graph.py           Phase 5  — graph wiring + MemorySaver checkpointer
        ├── nodes.py           Phase 5  — intro_node, clarify_node, respond_node
        └── state.py                    — AgentState TypedDict
```

---

## SSE Event Reference

| Event type | Sent by | Frontend action |
|---|---|---|
| `step` | `_stream_intro` or `_stream_resume` | Show grey badge |
| `token` | `_stream_intro` or `_stream_resume` | Append text to bubble |
| `interrupt` | `_stream_intro` (after loop ends) | Show option buttons, disable input |
| `done` | `_stream_resume` (after loop ends) | Reload messages from DB |
| `error` | Either, on exception | Log error, clear streaming state |
