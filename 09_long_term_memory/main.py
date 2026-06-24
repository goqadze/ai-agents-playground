"""
09 — Long-Term Memory with LangGraph + PostgreSQL

SHORT-TERM vs LONG-TERM MEMORY
────────────────────────────────────────────────────────────────────────
Short-term (project 08 — MemorySaver):
  • Stored in RAM
  • Lost when the process restarts
  • Scoped to one thread_id — other threads can't see it
  • Good for: remembering what was said earlier in THIS conversation

Long-term (this project — PostgreSQL):
  • Stored in a real database — survives restarts forever
  • Accessible from ANY session and ANY thread_id
  • The agent explicitly decides what is worth remembering
  • Good for: user profile, preferences, facts that matter across sessions

ARCHITECTURE
────────────────────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────────┐
  │  Two separate Postgres tables                                   │
  │                                                                 │
  │  langgraph_checkpoints   ← managed by PostgresSaver             │
  │  (conversation history)    stores messages per thread_id        │
  │                            survives restart, but scoped to      │
  │                            one thread                           │
  │                                                                 │
  │  long_term_memories      ← managed by our @tool functions       │
  │  (facts about the user)    key/value store                      │
  │                            one row per fact                     │
  │                            accessible from ANY thread           │
  └─────────────────────────────────────────────────────────────────┘

HOW THE AGENT USES MEMORY
────────────────────────────────────────────────────────────────────────
The system prompt instructs the agent to:
  1. Call recall_all_memories() at the start of each conversation
     so it can greet the user with what it already knows
  2. Call save_memory() whenever the user shares something worth keeping
  3. Call forget_memory() if the user asks to be forgotten

The agent DECIDES what to remember — we don't store every message,
just the facts the LLM judges to be important.

DEMO FLOW
────────────────────────────────────────────────────────────────────────
  Session 1  (thread-A) — introduce yourself, agent saves facts to DB
  Session 2  (thread-B) — completely new thread, agent recalls all facts
  Inspection             — show the raw rows in the memories table
  Interactive            — talk to the agent yourself
"""

import os
import sys
import time
import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://memuser:mempass@localhost:5432/memorydb"
)

# ── Wait for Postgres to be ready ─────────────────────────────────────────────

def wait_for_postgres(retries: int = 15, delay: float = 2.0):
    """Retry connecting until Postgres is ready (useful right after docker compose up)."""
    for attempt in range(1, retries + 1):
        try:
            psycopg.connect(DATABASE_URL).close()
            print("  ✓ Connected to PostgreSQL\n")
            return
        except psycopg.OperationalError:
            print(f"  Waiting for PostgreSQL... ({attempt}/{retries})")
            time.sleep(delay)
    print("  ✗ Could not connect to PostgreSQL. Is Docker running?")
    print("    Run:  docker compose up -d")
    sys.exit(1)

# ── Memory table setup ────────────────────────────────────────────────────────

def init_memory_table():
    """Create the long_term_memories table if it doesn't exist."""
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                key       VARCHAR(200) PRIMARY KEY,
                value     TEXT        NOT NULL,
                saved_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()

# ── Memory tools ──────────────────────────────────────────────────────────────
#
# These are plain @tool functions.
# The LLM reads the docstrings to decide when to call each one.

@tool
def save_memory(key: str, value: str) -> str:
    """
    Save an important fact about the user to long-term memory.
    Use this whenever the user shares personal information:
    their name, job, preferences, goals, or anything worth remembering
    in future conversations.
    Examples:
      save_memory("name", "Alice")
      save_memory("job", "software engineer")
      save_memory("favourite_food", "sushi")
      save_memory("city", "Tbilisi")
    """
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("""
            INSERT INTO long_term_memories (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, saved_at = NOW()
        """, (key, value))
        conn.commit()
    return f"✓ Saved to long-term memory: {key} = {value}"


@tool
def recall_all_memories() -> str:
    """
    Retrieve all facts stored in long-term memory.
    Always call this at the beginning of a conversation to greet
    the user with what you already know about them.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT key, value FROM long_term_memories ORDER BY saved_at"
        ).fetchall()
    if not rows:
        return "No long-term memories stored yet — this is a fresh start."
    lines = [f"  {key}: {value}" for key, value in rows]
    return "Long-term memories:\n" + "\n".join(lines)


@tool
def recall_memory(key: str) -> str:
    """
    Look up one specific fact from long-term memory by its key.
    Use this when you need a particular piece of information.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT value FROM long_term_memories WHERE key = %s", (key,)
        ).fetchone()
    return row[0] if row else f"No memory found for key '{key}'."


@tool
def forget_memory(key: str) -> str:
    """
    Delete a specific fact from long-term memory.
    Use this if the user asks to be forgotten or wants to correct a saved fact.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        result = conn.execute(
            "DELETE FROM long_term_memories WHERE key = %s RETURNING key", (key,)
        ).fetchone()
        conn.commit()
    return f"✓ Forgot '{key}'." if result else f"No memory found for key '{key}'."


MEMORY_TOOLS = [save_memory, recall_all_memories, recall_memory, forget_memory]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.

You have four memory tools:
  save_memory(key, value)  — save a fact about the user
  recall_all_memories()    — get everything stored
  recall_memory(key)       — look up one specific fact
  forget_memory(key)       — delete a fact

Rules:
1. At the START of every conversation, call recall_all_memories() silently.
   Use the result to personalise your greeting.
2. Whenever the user shares personal info (name, job, city, preferences, goals),
   call save_memory() immediately to store it.
3. If the user says "forget" or "don't remember", call forget_memory().
4. Be natural — don't say "I called recall_all_memories", just use the info.
"""

# ── Helper ────────────────────────────────────────────────────────────────────

def chat(agent, message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content


def divider(title: str):
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print('─' * 62)


def exchange(agent, label: str, message: str, thread_id: str):
    reply = chat(agent, message, thread_id)
    print(f"\n  [{label}]  thread: {thread_id}")
    print(f"  You  : {message}")
    print(f"  Agent: {reply}")

# ── Scenarios ─────────────────────────────────────────────────────────────────

def scenario_a(agent):
    divider("SESSION 1 — Introduce yourself  (thread-A)")
    print("""
  The agent starts fresh (or recalls whatever is already stored).
  We share facts — the agent saves them to Postgres.
  """)
    exchange(agent, "Turn 1", "Hi! My name is Alex and I live in Tbilisi.", "thread-A")
    exchange(agent, "Turn 2", "I work as a data engineer and I love coffee.",  "thread-A")
    exchange(agent, "Turn 3", "My favourite programming language is Python.",  "thread-A")


def scenario_b(agent):
    divider("SESSION 2 — Completely new thread  (thread-B)")
    print("""
  Different thread_id — no conversation history at all.
  But the agent calls recall_all_memories() and knows everything
  that was saved in Session 1.
  """)
    exchange(agent, "Fresh thread", "Hello, do you know who I am?", "thread-B")
    exchange(agent, "Follow-up",    "What do you know about my work?", "thread-B")


def scenario_c():
    divider("INSPECT — Raw rows in the database")
    print("""
  Bypass the agent entirely and read directly from Postgres.
  This proves the data is REALLY in the database — not in RAM.
  """)
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT key, value, saved_at FROM long_term_memories ORDER BY saved_at"
        ).fetchall()

    if not rows:
        print("  (no memories stored yet)")
    else:
        print(f"\n  {'KEY':<25} {'VALUE':<30} {'SAVED AT'}")
        print(f"  {'─'*25} {'─'*30} {'─'*25}")
        for key, value, saved_at in rows:
            print(f"  {key:<25} {value:<30} {saved_at}")
    print()


def scenario_d(agent):
    divider("INTERACTIVE — Chat with the agent")
    print("""
  Chat freely. The agent will remember what you share across sessions.
  Commands:
    new    → start a new thread_id (history resets, long-term memory stays)
    forget → type "forget: <key>" to delete a specific memory
    quit   → exit
  """)

    import uuid
    thread_id = f"interactive-{uuid.uuid4().hex[:6]}"
    print(f"  Thread: {thread_id}\n")

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
            thread_id = f"interactive-{uuid.uuid4().hex[:6]}"
            print(f"\n  ── New thread: {thread_id} (conversation history reset) ──\n")
            continue

        reply = chat(agent, user_input, thread_id)
        print(f"  Agent: {reply}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║          09 — Long-Term Memory with LangGraph + Postgres       ║
╚════════════════════════════════════════════════════════════════╝

  Short-term (08): MemorySaver — RAM — lost on restart
  Long-term  (09): PostgreSQL  — disk — survives forever

  Two Postgres tables:
    langgraph_checkpoints  ← conversation history (managed by PostgresSaver)
    long_term_memories     ← user facts (managed by @tool functions)
""")

    wait_for_postgres()
    init_memory_table()

    model = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=OPENAI_API_KEY)

    # PostgresSaver replaces MemorySaver.
    # It creates its own tables (langgraph_checkpoints, etc.) on setup().
    # The connection must stay open for the lifetime of the agent.
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()   # creates langgraph checkpoint tables if not exist

        agent = create_react_agent(
            model=model,
            tools=MEMORY_TOOLS,
            checkpointer=checkpointer,
            prompt=SYSTEM_PROMPT,
        )

        scenario_a(agent)
        scenario_b(agent)
        scenario_c()
        scenario_d(agent)

    print("\n  Done. Restart the script — Session 2 will still remember Alex.\n")


if __name__ == "__main__":
    main()
