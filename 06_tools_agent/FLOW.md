# Request Flow — Frontend to Backend

How a user message travels from the browser all the way through LangGraph and back.

---

## Big Picture

```
Browser                          FastAPI Backend                    LangGraph
──────────────────────────────────────────────────────────────────────────────
User types message
        │
        │  POST /api/conversations/{id}/chat
        │ ─────────────────────────────────────────────────────────────────▶
        │                                chat() receives request
        │                                        │
        │                                        │ save user msg to DB
        │                                        │
        │                                        │ return StreamingResponse
        │ ◀─────────────────────────────────────
        │  HTTP headers arrive (connection open)
        │
        │                                _stream_agent() generator starts
        │                                        │
        │                                        │ agent_graph.astream_events()
        │                                        │ ─────────────────────────▶
        │                                        │              classify node
        │                                        │              route node
        │                                        │              agent node
        │                                        │              tools node (if needed)
        │                                        │              extract node
        │  data: {"type":"step",...}             │ ◀─────────────────────────
        │ ◀─────────────────────────────────────
        │  data: {"type":"token","content":"Hi"} │
        │ ◀─────────────────────────────────────
        │  data: {"type":"done"}                 │
        │ ◀─────────────────────────────────────
        │
setStreaming() → React re-renders
```

---

## Step 1 — User Sends a Message

**File:** `frontend/src/components/ChatWindow.tsx`

The user types in the textarea and presses Enter (or clicks the send button).

```
┌─────────────────────────────────────────────┐
│  ChatWindow component                       │
│                                             │
│  <textarea onKeyDown={handleKeyDown} />     │
│                                             │
│  handleKeyDown:                             │
│    if Enter pressed (not Shift+Enter)       │
│      → call handleSubmit()                  │
│                                             │
│  handleSubmit:                              │
│    → trim input                             │
│    → call onSend(text)        (prop)        │
└─────────────────────────────────────────────┘
```

`onSend` is passed down from `App.tsx` as a prop — it points to `handleSend`.

---

## Step 2 — App Prepares and Calls the API

**File:** `frontend/src/App.tsx` — `handleSend()` function

```
handleSend(text)
    │
    ├─ 1. Add user message to UI immediately (optimistic update)
    │      setMessages(prev => [...prev, optimisticUserMsg])
    │      The message shows up instantly — no waiting for the server.
    │
    ├─ 2. Reset streaming state
    │      setStreaming({ content: "", steps: [], tools: [], currentStep: null })
    │
    └─ 3. Call streamChat(activeId, text)   ← starts the API call
```

---

## Step 3 — HTTP Request is Sent

**File:** `frontend/src/api/client.ts` — `streamChat()` function

```typescript
async function* streamChat(conversationId, message)
```

This is an **async generator** — it sends the HTTP request and then yields
one event at a time as bytes arrive from the server.

```
streamChat()
    │
    ├─ fetch() POST /api/conversations/{id}/chat
    │    body: { message: "What is 2 + 2?" }
    │    headers: Content-Type: application/json
    │
    ├─ waits for the response headers (connection opens)
    │
    └─ opens res.body.getReader()
         ↓
         reads raw bytes in a while(true) loop
         converts bytes → text with TextDecoder
         splits text on \n to find complete SSE lines
         each line looks like:
             data: {"type": "token", "content": "The"}
             ↓
         strips "data: " prefix → parses JSON → yields the object
```

> **What is SSE?**
> Server-Sent Events is a protocol where the server keeps the HTTP connection
> open and sends lines in the format `data: <payload>\n\n`.
> The browser reads them one by one as they arrive.

---

## Step 4 — FastAPI Receives the Request

**File:** `backend/app/api/chat.py` — `chat()` endpoint

```
POST /api/conversations/{id}/chat
        │
        ▼
async def chat(conversation_id, request, db):
        │
        ├─ 1. Load conversation history from PostgreSQL
        │      SELECT messages WHERE conversation_id = {id}
        │      Convert DB rows → LangChain message objects
        │         role "user"      → HumanMessage(content=...)
        │         role "assistant" → AIMessage(content=...)
        │
        ├─ 2. Save the new user message to the DB right now
        │      (so it's persisted before anything else happens)
        │
        └─ 3. return StreamingResponse(
                  _stream_agent(...),       ← generator, not called yet
                  media_type="text/event-stream"
              )
```

> **Key point:** `StreamingResponse` does NOT call `_stream_agent` immediately.
> It stores a reference to the generator and starts consuming it chunk by chunk
> as FastAPI writes the HTTP response body to the browser.

---

## Step 5 — The Stream Generator Runs

**File:** `backend/app/api/chat.py` — `_stream_agent()` function

This is where the LangGraph graph is actually started.

```
_stream_agent(conversation_id, question, history, is_first_message)
        │
        └─ agent_graph.astream_events(
               {
                   "history":        [...],   ← conversation history
                   "question":       "...",   ← current user message
                   "intent":         "",      ← will be filled by classify node
                   "analysis":       "",      ← will be filled by analyze node
                   "agent_messages": [],      ← will grow during ReAct loop
                   "answer":         "",      ← will be filled by extract node
               },
               version="v2"
           )
```

`astream_events` runs the graph and fires events as each node executes.
The generator listens for three event types and yields SSE lines:

```
Event received from graph            SSE line sent to browser
─────────────────────────────────────────────────────────────
new graph node starts          →     data: {"type":"step","step":"Classifying..."}
tool function is about to run  →     data: {"type":"tool","tool":"calculator"}
LLM streams a token            →     data: {"type":"token","content":"The answer"}
graph finishes                 →     data: {"type":"done"}
exception thrown               →     data: {"type":"error","message":"..."}
```

---

## Step 6 — LangGraph Runs the Graph

**Files:** `backend/app/agent/graph.py`, `nodes.py`, `tools.py`

```
Initial state enters the graph
        │
        ▼
┌─────────────┐
│  classify   │  LLM reads the question and replies "simple" or "complex"
└──────┬──────┘
       │
       │ route_by_intent()
       │
       ├─── "simple" ──────────────────────────────┐
       │                                           │
       └─── "complex" ──▶ ┌──────────┐            │
                          │ analyze  │            │
                          └────┬─────┘            │
                               │                  │
                               ▼                  ▼
                          ┌──────────────────────────┐
                          │       setup_agent        │
                          │  builds agent_messages:  │
                          │  [SystemMessage,          │
                          │   ...history,             │
                          │   HumanMessage(question)] │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────┐
                   ┌────▶ │         agent           │ ◀─────┐
                   │      │  llm_with_tools.invoke  │       │
                   │      └────────────┬────────────┘       │
                   │                   │                     │
                   │        should_continue()                │
                   │                   │                     │
                   │      ┌────────────┴────────────┐        │
                   │      │                         │        │
                   │  "extract"                  "tools"     │
                   │      │                         │        │
                   │      ▼                         ▼        │
                   │  ┌────────┐           ┌─────────────┐   │
                   │  │extract │           │    tools    │ ──┘
                   │  │answer  │           │  ToolNode   │
                   │  └───┬────┘           │runs the tool│
                   │      │                └─────────────┘
                   │      ▼
                   │     END
                   │
                   └── answer stored in state["answer"]
```

**The ReAct loop** (agent ↔ tools) repeats until the LLM stops requesting tools:

```
Iteration 1:
  agent: "I need to call calculator with '2 + 2'"
  → AIMessage with tool_calls = [{ name: "calculator", args: { expression: "2+2" } }]
  → should_continue returns "tools"

tools node:
  → calls calculator("2 + 2") → returns "4"
  → wraps result in ToolMessage and appends to agent_messages

Iteration 2:
  agent: reads ToolMessage("4"), now knows the answer
  → AIMessage with content = "2 + 2 equals 4."
  → should_continue returns "extract"

extract node:
  → state["answer"] = "2 + 2 equals 4."
  → END
```

---

## Step 7 — Events Flow Back to the Browser

As the graph runs, `_stream_agent` translates each event into an SSE line
and yields it. FastAPI writes each line to the HTTP response body immediately.

```
Graph event                         Browser receives
──────────────────────────────────────────────────────────────────────────
classify node starts          →   data: {"type":"step","step":"Classifying..."}
setup_agent node starts       →   data: {"type":"step","step":"Preparing agent..."}
agent node starts             →   data: {"type":"step","step":"Thinking..."}
tools node starts             →   data: {"type":"step","step":"Running tools..."}
calculator tool is called     →   data: {"type":"tool","tool":"calculator"}
agent node starts again       →   data: {"type":"step","step":"Thinking..."}
LLM streams "2"               →   data: {"type":"token","content":"2"}
LLM streams " +"              →   data: {"type":"token","content":" +"}
LLM streams " 2"              →   data: {"type":"token","content":" 2"}
LLM streams " equals 4."      →   data: {"type":"token","content":" equals 4."}
graph finishes                →   data: {"type":"done"}
```

---

## Step 8 — Frontend Handles Each Event

**File:** `frontend/src/App.tsx` — `for await` loop inside `handleSend()`

```
for await (const event of streamChat(activeId, text)) {

    "step"   → setStreaming({ currentStep: "Classifying..." })
               React re-renders → grey spinner badge appears in the chat bubble

    "tool"   → setStreaming({ tools: [...prev.tools, "calculator"] })
               React re-renders → green "🔧 calculator" badge appears

    "token"  → setStreaming({ content: prev.content + event.content })
               React re-renders → text appears character by character in the bubble

    "done"   → setStreaming(null)          ← removes the streaming bubble
               getConversation(activeId)  ← fetches final saved message from DB
               setMessages(data.messages) ← replaces streaming bubble with real message

    "error"  → setStreaming(null)
               console.error(event.message)
}
```

---

## Step 9 — After Streaming: Persist and Reload

After the graph finishes and `"done"` is sent, two things happen:

**Backend** (inside `_stream_agent` after the `async for` loop):

```
Save assistant message to PostgreSQL:
  content    = full_response  (all tokens joined)
  agent_steps = ["Classifying...", "Thinking...", "🔧 calculator", ...]

If this was the first message in the conversation:
  Update conversation title = first 60 chars of the question
```

**Frontend** (on receiving `"done"`):

```
getConversation(activeId)
  → GET /api/conversations/{id}
  → returns all messages including the just-saved assistant message
  → setMessages(data.messages) renders them with proper step/tool badges
  → updates the sidebar title if this was a new conversation
```

---

## Complete File Map

```
06_tools_agent/
│
├── frontend/src/
│   ├── components/
│   │   └── ChatWindow.tsx    Step 1  — user input, calls onSend()
│   ├── App.tsx               Step 2  — handleSend(), for await loop
│   └── api/
│       └── client.ts         Step 3  — streamChat() async generator, fetch + SSE reader
│
└── backend/app/
    ├── api/
    │   └── chat.py           Step 4 & 5 — chat() endpoint, _stream_agent() generator
    └── agent/
        ├── graph.py          Step 6  — graph wiring (nodes + edges)
        ├── nodes.py          Step 6  — classify, analyze, setup_agent, agent, extract
        └── tools.py          Step 6  — @tool functions (calculator, word_count, etc.)
```
