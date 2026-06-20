# 03 — LangChain ReAct Agent

An AI agent that can **reason** and **use tools** in a loop — instead of just answering once.

---

## The big idea: chain vs agent

| | Chain (project 02) | Agent (this project) |
|---|---|---|
| How it runs | Straight line, one shot | Loop until done |
| Can use tools? | No | Yes |
| Decides what to do next? | No | Yes |
| Good for | Fixed tasks | Open-ended questions |

---

## How the ReAct loop works

```
User question
      │
      ▼
  ┌─────────────────────────┐
  │  LLM: "Do I need a     │
  │  tool to answer this?" │
  └────────────┬────────────┘
               │
       ┌───────┴────────┐
    YES (tool call)   NO (final answer)
       │                │
       ▼                ▼
  ┌─────────┐        DONE ✅
  │  Tool   │
  │  runs   │
  └────┬────┘
       │ result
       └──────▶ back to LLM (loop again)
```

**RE**ason → **Act** (call tool) → observe result → **RE**ason again → …  
This is called the **ReAct** pattern.

---

## Tools in this project

| Tool | What it does |
|---|---|
| `add(a, b)` | Adds two numbers |
| `multiply(a, b)` | Multiplies two numbers |
| `word_count(text)` | Counts words in a string |
| `reverse_text(text)` | Reverses a string |

The LLM reads the **name** and **docstring** of each tool to decide when to use it.  
You never tell it "use `add` for addition" — it figures that out itself.

---

## Setup

### 1. Add your OpenAI API key

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-real-key-here...
```

### 2. Install dependencies

```bash
cd 03_langchain_agents
uv sync
```

---

## Run

```bash
uv run python main.py
```

You'll see each step of the agent loop printed out:

```
Question: Add 15 and 27, then multiply the result by 3.
  🔧 LLM wants to call tool: add
     with args: {'a': 15, 'b': 27}
  📦 Tool result (add): 42.0
  🔧 LLM wants to call tool: multiply
     with args: {'a': 42.0, 'b': 3}
  📦 Tool result (multiply): 126.0

  ✅ Final answer: The result is 126.
```

---

## Key concepts

| Concept | What it means |
|---|---|
| **`@tool` decorator** | Turns a Python function into a LangChain tool. The docstring tells the LLM when to use it. |
| **`create_react_agent()`** | Builds the ReAct loop graph (from LangGraph). Takes `model` + `tools`. |
| **`agent.stream()`** | Runs the agent and yields each step so you can observe the loop. |
| **LangGraph** | LangChain's library for building agents as state machines / graphs. |
| **`stream_mode="updates"`** | Returns one event per graph-node update (agent step or tool step). |

---

## Project layout

```
03_langchain_agents/
├── .env            ← your API key
├── tools.py        ← tool definitions (@tool decorated functions)
├── main.py         ← agent setup and demo runner
├── pyproject.toml  ← dependencies
└── README.md       ← this file
```
