# AI Development Reference

A quick-reference guide for **LangChain**, **LangGraph**, and **Deep Agents**.  
Built from the projects in this repo — every example comes from code you've already run.

---

## Table of Contents

1. [LangChain Core](#1-langchain-core)
2. [LangGraph](#2-langgraph)
3. [Deep Agents](#3-deep-agents)
4. [Comparison Cheat Sheet](#4-comparison-cheat-sheet)

---

## 1. LangChain Core

LangChain is a framework that makes it easy to build apps powered by LLMs.
Its building blocks are **components** you chain together with the `|` operator.

### Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│                    LangChain pipeline                      │
│                                                            │
│  ┌──────────────────┐                                      │
│  │ ChatPromptTemplate│  ← "You are a chef. Answer: {q}"   │
│  └────────┬─────────┘                                      │
│           │  .format_messages()                            │
│           ▼                                                │
│  ┌────────────────┐                                        │
│  │  ChatOpenAI    │  ← calls OpenAI API, returns response  │
│  └────────┬───────┘                                        │
│           │  AIMessage(content="Pasta is ...")             │
│           ▼                                                │
│  ┌─────────────────┐                                       │
│  │ StrOutputParser │  ← strips wrapper, returns plain str  │
│  └─────────────────┘                                       │
└────────────────────────────────────────────────────────────┘
```

---

### 1.1 `ChatOpenAI` — the LLM

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4.1-nano",   # which OpenAI model to use
    temperature=0,           # 0 = deterministic, 1 = creative
    api_key="sk-..."         # your OpenAI key
)

# Call it directly (returns AIMessage object)
response = model.invoke("What is the capital of France?")
print(response.content)   # "Paris"
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | str | Model name, e.g. `"gpt-4.1-nano"`, `"gpt-4o"` |
| `temperature` | float 0–2 | 0 = focused/deterministic, 1+ = creative/random |
| `api_key` | str | Your OpenAI API key |
| `max_tokens` | int | Max length of the response |

---

### 1.2 `ChatPromptTemplate` — reusable prompts

A template with `{placeholders}` you fill in at runtime.

```python
from langchain_core.prompts import ChatPromptTemplate

# Define the template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}. Be concise."),
    ("human",  "{question}"),
])

# Fill in the placeholders → produces a list of messages
messages = prompt.format_messages(
    role="chef",
    question="How do I boil an egg?"
)
# → [SystemMessage("You are a helpful chef..."), HumanMessage("How do I boil an egg?")]
```

**Other template formats:**

```python
# Single string template
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")

# Invoke directly (skips manual format_messages)
result = prompt.invoke({"text": "Hello world"})
```

---

### 1.3 `StrOutputParser` — extract text

The model returns an `AIMessage` object. `StrOutputParser` extracts just the string.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Can be used standalone
ai_message = model.invoke("Say hello")
text = parser.invoke(ai_message)   # "Hello!"

# Or chained (most common)
chain = model | parser
text = chain.invoke("Say hello")   # "Hello!"
```

---

### 1.4 Chains with `|`

The `|` operator connects components. Data flows left → right.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model  = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
parser = StrOutputParser()

# Build the chain
chain = prompt | model | parser

# Run it
result = chain.invoke({"topic": "Python programming"})
print(result)
```

```
Data flow:
  {"topic": "Python"}
         │
         ▼
  ChatPromptTemplate  →  [SystemMessage, HumanMessage("Tell me a joke about Python")]
         │
         ▼
  ChatOpenAI          →  AIMessage(content="Why do Python devs wear glasses?...")
         │
         ▼
  StrOutputParser     →  "Why do Python devs wear glasses?..."
```

---

### 1.5 `@tool` decorator — give the LLM tools

Tools are Python functions the LLM can decide to call. The `@tool` decorator registers them.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's 22°C and sunny in {city}"
```

**How the LLM decides which tool to call:**

```
┌─────────────────────────────────────────────────────────┐
│  LLM sees:                                              │
│                                                         │
│  Tool: add                                              │
│  Description: Add two integers and return the result.   │
│  Parameters: a (int), b (int)                           │
│                                                         │
│  Tool: get_weather                                      │
│  Description: Return the current weather for a city.    │
│  Parameters: city (str)                                 │
│                                                         │
│  Question: "What's 5 + 3?"                             │
│  → LLM decides: call `add` with a=5, b=3               │
└─────────────────────────────────────────────────────────┘
```

The **docstring** is what the LLM reads. Write it clearly — it's not for humans, it's for the AI.

---

### 1.6 `.invoke()` vs `.stream()`

| Method | Returns | Use when |
|---|---|---|
| `.invoke(input)` | Final result only | You just want the answer |
| `.stream(input)` | Iterator of chunks | You want to print output as it arrives |
| `.batch([inputs])` | List of results | You have many inputs to process at once |

```python
# invoke — waits for the full response
answer = chain.invoke({"topic": "cats"})

# stream — prints word by word
for chunk in chain.stream({"topic": "cats"}):
    print(chunk, end="", flush=True)
```

---

### 1.7 Memory — `ConversationBufferMemory`

Lets the model remember earlier messages in a conversation.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

# Save turns
memory.save_context(
    {"input": "My name is Alice"},
    {"output": "Nice to meet you, Alice!"}
)

# Load history
history = memory.load_memory_variables({})
# → {"history": [HumanMessage("My name is Alice"), AIMessage("Nice to meet you, Alice!")]}
```

---

## 2. LangGraph

LangGraph models your agent as a **graph** — nodes are steps, edges decide what runs next.
This gives you precise control over multi-step agent logic.

### Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│                    LangGraph state machine                 │
│                                                            │
│    START                                                   │
│      │                                                     │
│      ▼                                                     │
│  ┌───────┐   tool call?   ┌──────────┐                    │
│  │  LLM  │ ─────────────▶ │   Tool   │                    │
│  │ node  │ ◀───────────── │   node   │                    │
│  └───┬───┘   tool result  └──────────┘                    │
│      │                                                     │
│      │ no tool call (final answer)                        │
│      ▼                                                     │
│     END                                                    │
└────────────────────────────────────────────────────────────┘
```

---

### 2.1 `create_react_agent()` — prebuilt ReAct graph

The easiest way to build an agent. Builds the full ReAct loop for you.

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
tools = [add, multiply, get_weather]   # list of @tool functions

agent = create_react_agent(model=model, tools=tools)
```

**Running it:**

```python
# invoke — get only the final answer
result = agent.invoke({"messages": [("user", "What is 6 * 7?")]})
final_answer = result["messages"][-1].content

# stream — see every step of the loop
for step in agent.stream(
    {"messages": [("user", "What is 6 * 7?")]},
    stream_mode="updates"   # one event per node
):
    node_name = list(step.keys())[0]    # "agent" or "tools"
    messages  = step[node_name]["messages"]
    for msg in messages:
        print(type(msg).__name__, msg.content)
```

**The ReAct loop in detail:**

```
User: "What is 6 * 7, then add 10?"

Step 1 — agent node:
  LLM thinks: "I need to multiply first"
  LLM outputs: tool_call { name: "multiply", args: {a: 6, b: 7} }

Step 2 — tools node:
  Framework calls multiply(6, 7) → 42
  Returns ToolMessage(content="42")

Step 3 — agent node:
  LLM thinks: "Now I have 42. Add 10."
  LLM outputs: tool_call { name: "add", args: {a: 42, b: 10} }

Step 4 — tools node:
  Framework calls add(42, 10) → 52
  Returns ToolMessage(content="52")

Step 5 — agent node:
  LLM thinks: "No more tools needed."
  LLM outputs: AIMessage(content="The answer is 52")

Loop ends → return final answer
```

---

### 2.2 `StateGraph` — build a custom graph

When you need more control than `create_react_agent` provides.

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState

# MessagesState is a pre-built state dict with a "messages" list
# that automatically appends new messages (doesn't overwrite)

def my_llm_node(state: MessagesState):
    """Call the LLM and return new messages."""
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def my_tool_node(state: MessagesState):
    """Run the tool the LLM asked for."""
    last_message = state["messages"][-1]
    # ... call the tool ...
    return {"messages": [tool_result]}

def should_call_tool(state: MessagesState) -> str:
    """Router: decide which node runs next."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"   # go to tool node
    return END           # stop

# Build the graph
graph = StateGraph(MessagesState)

graph.add_node("llm",   my_llm_node)
graph.add_node("tools", my_tool_node)

graph.set_entry_point("llm")   # always start here

graph.add_conditional_edges(
    "llm",              # from this node
    should_call_tool,   # call this function to decide
    {                   # map return value → next node
        "tools": "tools",
        END: END,
    }
)
graph.add_edge("tools", "llm")  # after tools, always go back to LLM

app = graph.compile()
result = app.invoke({"messages": [("user", "Hello")]})
```

**Graph anatomy:**

```
Nodes         = steps that do work (call LLM, call tool, call API, etc.)
Edges         = fixed paths  (A always goes to B)
Conditional   = dynamic paths (function decides where to go next)
State         = shared dict passed between all nodes
EntryPoint    = first node to run
END           = terminal node (stop the graph)
```

---

### 2.3 `MessagesState` — built-in state

```python
from langgraph.graph.message import MessagesState

# MessagesState is equivalent to:
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # The Annotated[..., operator.add] means:
    # when a node returns {"messages": [new_msg]},
    # it APPENDS to the list instead of replacing it
```

---

### 2.4 Memory across conversations — `MemorySaver`

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent  = create_react_agent(model=model, tools=tools, checkpointer=memory)

# thread_id groups messages into a "conversation"
config = {"configurable": {"thread_id": "user-123"}}

# First message
agent.invoke({"messages": [("user", "My name is Alice")]}, config=config)

# Second message — the agent remembers "Alice"
result = agent.invoke({"messages": [("user", "What's my name?")]}, config=config)
```

---

### 2.5 `stream_mode` options

| Mode | What you get | Use when |
|---|---|---|
| `"values"` | Full state after each step | You want the whole picture each time |
| `"updates"` | Only what changed in each step | You want to see each node's output |
| `"messages"` | Token-by-token streaming | You want to print as the LLM types |

```python
# See each step
for step in agent.stream(input, stream_mode="updates"):
    print(step)

# Stream tokens live
for token in agent.stream(input, stream_mode="messages"):
    print(token[0].content, end="")
```

---

## 3. Deep Agents

Deep Agents is built on top of LangChain + LangGraph.
It adds a full **planning pipeline** so the agent thinks before acting.

### Architecture overview

```
┌────────────────────────────────────────────────────────────┐
│                   Deep Agent pipeline                      │
│                                                            │
│   User question                                            │
│        │                                                   │
│        ▼                                                   │
│   1. PLAN ──── breaks task into steps                      │
│        │                                                   │
│        ▼                                                   │
│   2. EXECUTE ─ calls tools for each step                   │
│        │                                                   │
│        ▼                                                   │
│   3. STORE ─── saves results to virtual file system        │
│        │                                                   │
│        ▼                                                   │
│   4. DELEGATE ─ spawns subagents for hard sub-tasks        │
│        │         (optional)                                │
│        ▼                                                   │
│   5. SYNTHESIZE ─ compiles everything into final answer    │
└────────────────────────────────────────────────────────────┘
```

---

### 3.1 `create_deep_agent()` — main entry point

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-4.1-nano",   # provider:model-name format
    tools=ALL_TOOLS,                # list of plain Python functions
    system_prompt="You are a helpful assistant. Plan before acting.",
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | str | `"provider:model-name"` — see table below |
| `tools` | list | Plain Python functions (no `@tool` decorator needed) |
| `system_prompt` | str | Optional instructions to shape agent behavior |

**Model string format:**

```
"openai:gpt-4.1-nano"        ← OpenAI GPT-4.1 nano
"openai:gpt-4o"              ← OpenAI GPT-4o
"anthropic:claude-sonnet-4"  ← Anthropic Claude Sonnet 4
"google:gemini-2.0-flash"    ← Google Gemini 2.0 Flash
```

---

### 3.2 Defining tools

**LangChain tools** need the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""       ← LLM reads this to decide when to use the tool
    return a + b
```

**Deep Agents tools** are plain functions — no decorator:

```python
def add(a: int, b: int) -> int:
    """Add two integers."""       ← same docstring convention applies
    return a + b

def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's 22°C and sunny in {city}"

ALL_TOOLS = [add, get_weather]
```

---

### 3.3 Running the agent

```python
# invoke — runs the full pipeline, returns final state
result = agent.invoke({
    "messages": [{"role": "user", "content": "Add 50 and 75, then multiply by 4"}]
})

# The final answer is always the last message
final_answer = result["messages"][-1].content
print(final_answer)
```

```
Input format:
  {"messages": [{"role": "user", "content": "..."}]}

Output format (result is a dict):
  result["messages"]        → all messages exchanged
  result["messages"][-1]   → last message = final answer
  result["messages"][-1].content  → the text string
```

---

### 3.4 Deep Agents vs plain LangGraph agent

```
┌──────────────────────────────────────────────────────────────┐
│              Capability comparison                           │
│                                                              │
│  Feature                  │ LangGraph   │ Deep Agents        │
│  ─────────────────────────│─────────────│────────────────    │
│  Tool calling             │     ✓       │       ✓            │
│  ReAct loop               │     ✓       │       ✓            │
│  Task planning            │     ✗       │       ✓            │
│  Virtual file system      │     ✗       │       ✓            │
│  Subagent delegation      │     ✗       │       ✓            │
│  Human-in-the-loop        │     ✓ *     │       ✓            │
│  Long-term memory         │     ✓ *     │       ✓            │
│  Custom graph control     │     ✓       │       ✗            │
│                                                              │
│  * requires manual setup in LangGraph                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Comparison Cheat Sheet

### When to use what

```
Task                                  │ Use
──────────────────────────────────────│───────────────────────────
Single prompt → answer                │ LangChain chain (prompt | model | parser)
Agent that calls tools in a loop      │ LangGraph create_react_agent()
Agent with precise custom logic       │ LangGraph StateGraph
Complex planning + long tasks         │ Deep Agents create_deep_agent()
```

---

### API quick reference

**LangChain chain:**

```python
chain = prompt | model | parser
result = chain.invoke({"key": "value"})       # → str
result = chain.stream({"key": "value"})       # → iterator of str chunks
```

**LangGraph ReAct agent:**

```python
agent = create_react_agent(model=model, tools=tools)
result = agent.invoke({"messages": [("user", "question")]})
answer = result["messages"][-1].content       # → str

for step in agent.stream(input, stream_mode="updates"):
    ...   # → dict per node
```

**Deep Agents:**

```python
agent = create_deep_agent(model="openai:gpt-4.1-nano", tools=tools)
result = agent.invoke({"messages": [{"role": "user", "content": "question"}]})
answer = result["messages"][-1].content       # → str
```

---

### Import cheat sheet

```python
# LangChain core
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory

# LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Deep Agents
from deepagents import create_deep_agent

# Utilities
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

### Message types

| Type | Created by | Contains |
|---|---|---|
| `HumanMessage` | User input | The user's question |
| `AIMessage` | The LLM | Text response OR tool call request |
| `ToolMessage` | The framework | Result of a tool that was called |
| `SystemMessage` | You (system prompt) | Instructions shaping the LLM's behavior |

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# AIMessage with a tool call:
msg.tool_calls   # → [{"name": "add", "args": {"a": 5, "b": 3}}]
msg.content      # → "" (empty when it's a tool call, not text)

# AIMessage with final answer:
msg.tool_calls   # → []
msg.content      # → "The answer is 42"
```

---

*Projects in this repo: `01_simple_agent` → `02_openai_intro` → `03_langchain_agents` → `03_deepagents` → `04_reference`*
