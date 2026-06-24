"""
08 — Short-Term Memory with LangGraph

WHAT IS SHORT-TERM MEMORY?
───────────────────────────
Short-term memory means the agent remembers what was said earlier
IN THE SAME CONVERSATION SESSION — but forgets everything when
the session ends (or when you start a new thread).

This is different from:
  • No memory      — agent forgets after every single message
  • Long-term memory — agent remembers across multiple sessions (stored in a DB)

HOW IT WORKS IN LANGGRAPH
──────────────────────────
LangGraph implements short-term memory using two things:

  1. MemorySaver — an in-memory checkpointer.
     After each message exchange it saves a snapshot of the full
     conversation state (all messages so far).

  2. thread_id — a string that identifies a conversation.
     Every time you send a message you pass the same thread_id.
     LangGraph finds the saved snapshot for that thread and
     pre-loads all previous messages before calling the LLM.

     Same thread_id  →  agent has context  →  remembers you
     New  thread_id  →  blank slate         →  forgets everything

  ┌──────────────────────────────────────────────────────────────┐
  │  MemorySaver (lives in RAM)                                  │
  │                                                              │
  │  "thread-alice" → [turn1, turn2, turn3, ...]                 │
  │  "thread-bob"   → [turn1, turn2, ...]                        │
  │  "thread-xyz"   → [turn1]                                    │
  └──────────────────────────────────────────────────────────────┘

THIS DEMO
─────────
Three scenarios shown back-to-back:

  Scenario A — single thread, multiple turns
    The agent remembers your name, hobby, and uses them later.

  Scenario B — two separate threads
    Shows that thread-1 and thread-2 are completely isolated.

  Scenario C — interactive chat
    You talk to the agent yourself. Type 'new' to start a fresh thread.
"""

import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=api_key)

# MemorySaver is the checkpointer — it saves conversation state after each turn.
# Everything is stored in RAM, so it resets when the process restarts.
memory = MemorySaver()

# create_react_agent with checkpointer= enables memory.
# Without checkpointer= the agent has zero memory — every message is standalone.
agent = create_react_agent(
    model=model,
    tools=[],               # no tools needed for this demo
    checkpointer=memory,    # ← this single argument enables short-term memory
)


# ── Helper ────────────────────────────────────────────────────────────────────

def chat(message: str, thread_id: str) -> str:
    """
    Send one message to the agent on a specific thread and return the reply.

    config["configurable"]["thread_id"] is how LangGraph knows which
    saved conversation snapshot to load before calling the LLM.
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content


def divider(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def exchange(label: str, message: str, thread_id: str):
    """Print one question/answer exchange."""
    reply = chat(message, thread_id)
    print(f"\n  [{label}]")
    print(f"  You  : {message}")
    print(f"  Agent: {reply}")


# ── Scenario A — single thread, multiple turns ────────────────────────────────

def scenario_a():
    divider("SCENARIO A — Same thread: agent remembers context")

    print("""
  We send 4 messages on the SAME thread_id.
  Each message adds to the memory. The agent carries context
  from turn 1 all the way through turn 4.
  """)

    thread = "demo-thread-alice"

    exchange("Turn 1 — introduce name",    "Hi, my name is Alice.",              thread)
    exchange("Turn 2 — add a detail",      "I love hiking in the mountains.",    thread)
    exchange("Turn 3 — recall name",       "What is my name?",                   thread)
    exchange("Turn 4 — recall both facts", "What do you know about me so far?",  thread)

    print("""
  ✓ The agent recalled "Alice" and "hiking" even though
    they were mentioned in earlier turns — not in the last message.
  """)


# ── Scenario B — two isolated threads ─────────────────────────────────────────

def scenario_b():
    divider("SCENARIO B — Different threads: completely isolated")

    print("""
  We introduce "Bob" on thread-1, then ask about him on thread-2.
  thread-2 has never seen thread-1's messages — it starts blank.
  """)

    exchange("Thread 1 — introduce Bob", "My name is Bob and I work as a chef.", "thread-1")
    exchange("Thread 1 — recall",        "What is my job?",                      "thread-1")
    exchange("Thread 2 — fresh start",   "What is my name and job?",             "thread-2")

    print("""
  ✓ Thread-1 knows Bob is a chef.
  ✗ Thread-2 has no idea — different thread_id = different memory.
  """)


# ── Scenario C — how memory grows turn by turn ────────────────────────────────

def scenario_c():
    divider("SCENARIO C — Peek inside: how many messages are stored?")

    print("""
  After each turn we inspect the saved state to show exactly
  how the messages list grows inside MemorySaver.
  """)

    thread = "demo-inspect"
    config = {"configurable": {"thread_id": thread}}

    turns = [
        "My favourite colour is blue.",
        "I have a dog named Max.",
        "What pet do I have and what is its name?",
    ]

    for i, message in enumerate(turns, 1):
        reply = chat(message, thread)

        # get_state() returns the current snapshot for this thread
        state = agent.get_state(config)

        msg_count = len(state.values.get("messages", []))
        print(f"\n  Turn {i}:")
        print(f"    You  : {message}")
        print(f"    Agent: {reply}")
        print(f"    ── Messages stored in MemorySaver: {msg_count} ──")
        #
        # Each turn adds 2 messages: HumanMessage + AIMessage
        # Turn 1 → 2 messages
        # Turn 2 → 4 messages
        # Turn 3 → 6 messages

    print("""
  Each turn adds 2 messages (Human + AI) to the stored state.
  The LLM sees ALL of them on the next turn — that's how it "remembers".
  """)


# ── Scenario D — interactive ──────────────────────────────────────────────────

def scenario_d():
    divider("SCENARIO D — Interactive chat")

    print("""
  Chat with the agent yourself.
  Commands:
    new   → start a fresh thread (agent forgets everything)
    quit  → exit
  """)

    thread_id = str(uuid.uuid4())
    print(f"  Started thread: {thread_id[:8]}...\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            print(f"\n  ── New thread started: {thread_id[:8]}... (memory cleared) ──\n")
            continue

        reply = chat(user_input, thread_id)
        print(f"  Agent: {reply}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          08 — Short-Term Memory with LangGraph               ║
╚══════════════════════════════════════════════════════════════╝

  Key concept:
    MemorySaver  +  thread_id  =  short-term memory

  MemorySaver saves the conversation state in RAM after each turn.
  thread_id tells LangGraph which saved state to load.
  Same thread → remembers. New thread → forgets.
""")

    scenario_a()
    scenario_b()
    scenario_c()
    scenario_d()
