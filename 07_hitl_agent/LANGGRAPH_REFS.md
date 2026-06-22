# LangGraph Object Reference

Cheat sheets for the two objects you work with most when building streaming agents:

1. **`event`** — yielded by `astream_events()`
2. **`graph_state`** — returned by `get_state()`

---

## 1. The `event` Object — from `astream_events()`

Every iteration of `async for event in graph.astream_events(...)` gives you one of these.

### Full shape

```python
event = {
    "event":      str,          # what happened — see event types below
    "name":       str,          # name of the thing that fired (node, tool, model, chain)
    "run_id":     str,          # unique ID for this specific run
    "tags":       list[str],    # tags attached to the component
    "metadata":   dict,         # langgraph internals — most important one
    "data":       dict,         # the actual payload — changes per event type
}
```

### The `metadata` dict — most useful sub-object

```python
event["metadata"] = {
    "langgraph_node":        str,   # name of the graph node currently running
                                    # e.g. "intro", "clarify", "respond", "tools"

    "langgraph_step":        int,   # step number in the graph execution (0-indexed)

    "langgraph_triggers":    list,  # what triggered this node to run
                                    # e.g. ["start:intro"] or ["intro"]

    "langgraph_checkpoint_id": str, # ID of the checkpoint saved before this step

    "ls_model_name":         str,   # LLM model name (when inside an LLM call)
                                    # e.g. "gpt-4.1-nano"

    "ls_provider":           str,   # LLM provider — e.g. "openai"

    "ls_temperature":        float, # temperature used for this LLM call

    "ls_max_tokens":         int,   # max_tokens if set
}
```

**How we use it in chat.py:**
```python
node = event.get("metadata", {}).get("langgraph_node", "")
# → "intro" | "clarify" | "respond" | "agent" | "tools" | ""
```

---

### Event Types — `event["event"]`

Each `event["event"]` string follows the pattern `on_{thing}_{lifecycle}`.

#### Chain events (graph nodes)

```python
"on_chain_start"    # a node just started running
"on_chain_stream"   # a node produced an intermediate output
"on_chain_end"      # a node finished running
```

`event["data"]` for chain events:
```python
# on_chain_start
{ "input": { ...the state dict passed to this node... } }

# on_chain_stream
{ "chunk": { ...partial output from the node... } }

# on_chain_end
{ "output": { ...the dict the node returned... } }
```

#### LLM / Chat model events

```python
"on_chat_model_start"   # LLM call started
"on_chat_model_stream"  # one token chunk arrived  ← most used for streaming
"on_chat_model_end"     # LLM call finished
```

`event["data"]` for chat model events:
```python
# on_chat_model_start
{
    "input": {
        "messages": [[SystemMessage(...), HumanMessage(...)]]
    }
}

# on_chat_model_stream  ← THIS IS THE ONE WE USE
{
    "chunk": AIMessageChunk(
        content="Hello",        # the token text — empty string during tool calls
        tool_call_chunks=[...], # filled when LLM is streaming a tool call
        id="run-abc123",
        response_metadata={...}
    )
}

# on_chat_model_end
{
    "output": AIMessage(
        content="Hello, how can I help?",
        tool_calls=[...],
        response_metadata={
            "finish_reason": "stop",  # "stop" | "tool_calls" | "length"
            "model_name":    "gpt-4.1-nano",
            "usage": {
                "prompt_tokens":     42,
                "completion_tokens": 18,
                "total_tokens":      60
            }
        }
    )
}
```

**How we use it in chat.py:**
```python
if kind == "on_chat_model_stream" and node == "respond":
    chunk = event["data"]["chunk"]
    if chunk.content:           # empty during tool calls, has text for final answer
        yield chunk.content
```

#### Tool events

```python
"on_tool_start"   # tool function is about to be called
"on_tool_end"     # tool function returned a result
```

`event["data"]` for tool events:
```python
# on_tool_start
{
    "input": {
        "expression": "2 + 2"   # the args the LLM passed to the tool
    }
}

# on_tool_end
{
    "output": ToolMessage(
        content="4",            # what the tool returned (as string)
        name="calculator",
        tool_call_id="call_abc"
    )
}
```

**How we use it in chat.py (06_tools_agent):**
```python
if kind == "on_tool_start" and node == "tools":
    tool_name = event.get("name", "tool")   # "calculator", "word_count", etc.
```

#### LLM (non-chat) events

```python
"on_llm_start"    # raw LLM (not chat model) started
"on_llm_stream"   # raw LLM token
"on_llm_end"      # raw LLM finished
```

*(Rarely used — only for non-chat LLMs like `OpenAI()` instead of `ChatOpenAI()`)*

---

### Quick Event Filter Reference

```python
async for event in graph.astream_events(input, config=config, version="v2"):
    kind = event["event"]
    node = event.get("metadata", {}).get("langgraph_node", "")
    name = event.get("name", "")

    # ── Stream final answer tokens ─────────────────────────────────────────
    if kind == "on_chat_model_stream" and node == "respond":
        chunk = event["data"]["chunk"]
        if chunk.content:
            print(chunk.content, end="")

    # ── Detect tool calls ─────────────────────────────────────────────────
    if kind == "on_tool_start":
        print(f"Calling tool: {name}")
        print(f"With args:    {event['data']['input']}")

    # ── See tool results ──────────────────────────────────────────────────
    if kind == "on_tool_end":
        print(f"Tool result:  {event['data']['output'].content}")

    # ── Know which node is running ────────────────────────────────────────
    if kind == "on_chain_start" and node:
        print(f"Node started: {node}")

    # ── See what a node returned ──────────────────────────────────────────
    if kind == "on_chain_end" and node:
        print(f"Node output:  {event['data']['output']}")

    # ── Token usage per LLM call ──────────────────────────────────────────
    if kind == "on_chat_model_end":
        usage = event["data"]["output"].response_metadata.get("usage", {})
        print(f"Tokens used: {usage.get('total_tokens')}")
```

---

### Full Event Example — `on_chat_model_stream`

This is what the object literally looks like when you `print(event)`:

```python
{
    "event":   "on_chat_model_stream",
    "name":    "ChatOpenAI",
    "run_id":  "f3a2b1c4-...",
    "tags":    ["seq:step:2"],
    "metadata": {
        "langgraph_step":          2,
        "langgraph_node":          "respond",
        "langgraph_triggers":      ["clarify"],
        "langgraph_checkpoint_id": "1ef8a...",
        "ls_model_name":           "gpt-4.1-nano",
        "ls_provider":             "openai",
        "ls_temperature":          0.7,
    },
    "data": {
        "chunk": AIMessageChunk(
            content=" recursion",
            id="run-f3a2b1c4",
            tool_call_chunks=[]
        )
    }
}
```

---

## 2. The `graph_state` Object — from `get_state()`

```python
graph_state = agent_graph.get_state(config)
# async version:
graph_state = await agent_graph.aget_state(config)
```

### Full shape — `StateSnapshot`

```python
graph_state = StateSnapshot(
    values   = dict,             # current state dict — same shape as AgentState
    next     = tuple[str, ...],  # node names that will run next
                                 # EMPTY tuple () = graph is finished
                                 # non-empty      = graph is paused/interrupted
    tasks    = tuple[PregelTask, ...],  # pending tasks (one per next node)
    config   = dict,             # the config used (includes thread_id)
    metadata = dict,             # step count, source, writes info
    created_at   = str,          # ISO timestamp of this checkpoint
    parent_config = dict | None, # config of the previous checkpoint
)
```

---

### `graph_state.values` — the current state dict

```python
graph_state.values
# → the full AgentState dict at the time of the snapshot

# Example:
{
    "messages":    [HumanMessage("What is recursion?")],
    "question":    "What is recursion?",
    "intro_text":  "Great question! Let me tailor my answer...",
    "user_choice": "",      # empty — interrupt fired before this was set
    "answer":      "",
}
```

---

### `graph_state.next` — what runs next

```python
graph_state.next
# → ()                  graph is DONE — nothing left to run
# → ("respond",)        graph is PAUSED — "respond" node will run when resumed
# → ("agent", "tools")  multiple nodes waiting (parallel branches)

# How we use it to detect an interrupt:
if graph_state.next:
    # graph is paused — was interrupted
    ...
else:
    # graph finished normally
    ...
```

---

### `graph_state.tasks` — pending task objects

One `PregelTask` per node in `graph_state.next`.

```python
graph_state.tasks
# → tuple of PregelTask objects

task = graph_state.tasks[0]   # first pending task
```

#### `PregelTask` shape

```python
task = PregelTask(
    id          = str,                  # unique task ID
    name        = str,                  # node name — e.g. "clarify"
    path        = tuple,                # internal routing path
    error       = Exception | None,     # set if the task failed
    interrupts  = tuple[Interrupt, ...],# interrupt objects (see below)
    state       = dict | None,          # task-level state (subgraphs)
    result      = dict | None,          # set after task completes
)
```

**Common usage:**
```python
task = graph_state.tasks[0]

task.name          # → "clarify"   which node is pending
task.error         # → None        or an Exception if it crashed
task.interrupts    # → (Interrupt(...),)   if interrupt() was called
```

---

### `task.interrupts` — the interrupt objects

```python
task.interrupts
# → tuple of Interrupt objects

interrupt_obj = task.interrupts[0]   # first (usually only) interrupt
```

#### `Interrupt` shape

```python
interrupt_obj = Interrupt(
    value     = any,     # whatever you passed to interrupt({...})
    resumable = bool,    # True — can be resumed with Command(resume=...)
    ns        = list,    # internal namespace path
    when      = str,     # "during" — when in the node lifecycle it fired
)
```

**How we use it in chat.py:**
```python
interrupt_data = graph_state.tasks[0].interrupts[0].value
# → {
#       "question": "How would you like me to answer?",
#       "options":  ["Concise summary", "Detailed explanation", ...]
#   }
```

---

### `graph_state.metadata` — execution info

```python
graph_state.metadata
# → {
#       "step":    2,          # how many steps have run so far (0-indexed)
#       "source":  "loop",     # "input" | "loop" | "update"
#       "writes":  {           # what the last node wrote to state
#           "intro_text": "Great question! Let me..."
#       },
#       "parents": {}
#   }
```

---

### Detecting interrupt vs finished — the pattern

```python
graph_state = agent_graph.get_state(config)

if graph_state.next:
    # ── INTERRUPTED ───────────────────────────────────────────────────────
    pending_node   = graph_state.next[0]              # "clarify"
    task           = graph_state.tasks[0]
    interrupt_obj  = task.interrupts[0]
    interrupt_data = interrupt_obj.value              # your dict from interrupt({...})

    print(f"Paused at node: {pending_node}")
    print(f"Interrupt data: {interrupt_data}")

else:
    # ── FINISHED ──────────────────────────────────────────────────────────
    final_state = graph_state.values
    print(f"Answer: {final_state['answer']}")
```

---

### Resuming — `Command(resume=...)`

```python
from langgraph.types import Command

# Pass the user's choice back to the interrupted node
async for event in graph.astream_events(
    Command(resume="Step-by-step walkthrough"),
    config={"configurable": {"thread_id": "abc-123"}},
    version="v2",
):
    ...
```

`Command` shape:
```python
Command(
    resume = any,      # the value returned by interrupt() in the node
                       # can be a string, dict, list — whatever makes sense
    goto   = str | list[str] | None,   # override which node runs next (advanced)
    update = dict | None,              # update state values before resuming (advanced)
)
```

---

## Quick Reference Card

```
astream_events yields:
┌─────────────────────────────┬──────────────────────────────────────────────┐
│ event["event"]              │ event["data"] key you need                   │
├─────────────────────────────┼──────────────────────────────────────────────┤
│ on_chat_model_stream        │ ["chunk"].content  → token text              │
│ on_chat_model_end           │ ["output"].response_metadata["usage"]        │
│ on_tool_start               │ ["input"]          → tool arguments          │
│ on_tool_end                 │ ["output"].content → tool return value       │
│ on_chain_start              │ ["input"]          → node input state        │
│ on_chain_end                │ ["output"]         → node return dict        │
└─────────────────────────────┴──────────────────────────────────────────────┘

event["metadata"] keys you need:
  langgraph_node   → which node is running right now
  langgraph_step   → how many steps have run
  ls_model_name    → which LLM model is being called

get_state() returns StateSnapshot:
┌──────────────────┬─────────────────────────────────────────────────────────┐
│ property         │ meaning                                                 │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ .values          │ full state dict (your AgentState)                       │
│ .next            │ () = done  |  ("node",) = paused/interrupted            │
│ .tasks[0].name   │ name of the pending node                                │
│ .tasks[0]        │                                                         │
│  .interrupts[0]  │                                                         │
│  .value          │ the dict you passed to interrupt({...})                 │
│ .metadata["step"]│ how many nodes have run so far                          │
│ .created_at      │ ISO timestamp of this checkpoint                        │
└──────────────────┴─────────────────────────────────────────────────────────┘
```
