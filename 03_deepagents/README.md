# 03 — Deep Agents + LangChain + OpenAI

An AI agent built with [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) — a framework on top of LangChain + LangGraph that adds planning, file system management, subagent delegation, and more.

---

## How this compares to the previous projects

| Project | What it uses | What the agent can do |
|---|---|---|
| 02 — chain | LangChain prompt → model | Answer once, no tools |
| 03 (old) — ReAct | LangChain + LangGraph | Loop and call tools |
| **03 — Deep Agents** | **Deep Agents + LangChain** | **Plan → tools → subagents → memory** |

---

## What Deep Agents adds

```
Plain agent loop:
  question → LLM → tool? → result → LLM → answer

Deep Agent pipeline:
  question
    │
    ▼
  1. PLAN     — breaks the task into steps
    │
    ▼
  2. EXECUTE  — calls tools for each step
    │
    ▼
  3. STORE    — saves intermediate results to a virtual file system
    │
    ▼
  4. DELEGATE — spawns subagents for complex sub-tasks (if needed)
    │
    ▼
  5. SYNTHESIZE — compiles everything into a final answer
```

---

## Tools in this project

| Tool | What it does |
|---|---|
| `add(a, b)` | Adds two numbers |
| `multiply(a, b)` | Multiplies two numbers |
| `word_count(text)` | Counts words in a string |
| `get_weather(city)` | Returns fake weather data (placeholder for a real API) |

---

## Setup

### 1. Add your API key

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-real-key-here...
```

### 2. Install dependencies

```bash
cd 03_deepagents
uv sync
```

---

## Run

```bash
uv run python main.py
```

---

## Key concepts

| Concept | What it means |
|---|---|
| **`create_deep_agent()`** | Main entry point. Takes `model`, `tools`, and `system_prompt`. |
| **`"openai:gpt-4.1-nano"`** | Deep Agents model format: `"provider:model-name"`. |
| **`agent.invoke()`** | Runs the full pipeline and returns the final state dict. |
| **`result["messages"][-1]`** | The last message is always the agent's final answer. |
| **Task planning** | Agent automatically breaks complex tasks into steps before acting. |
| **Virtual file system** | Agent can write/read files to store context across long tasks. |
| **Subagents** | Agent can spawn specialist agents to handle sub-tasks in parallel. |

---

## Project layout

```
03_deepagents/
├── .env            ← your API key
├── tools.py        ← plain Python functions (no decorators needed)
├── main.py         ← agent setup and demo
├── pyproject.toml  ← dependencies
└── README.md       ← this file
```
