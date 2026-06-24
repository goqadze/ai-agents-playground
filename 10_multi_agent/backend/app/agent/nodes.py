"""
The three specialized agents — each is a LangGraph node function.

PLANNER   — breaks the topic into focused research questions
RESEARCHER — answers each question with detailed findings
WRITER    — synthesizes the findings into a polished article

Each agent has its own system prompt that defines its role and output format.
This is the core of the multi-agent pattern: specialization through prompting.

In a production system the Researcher would also have web-search tools
(Tavily, DuckDuckGo, Exa) to fetch real-time information.  Here we rely
on the LLM's training knowledge so the project has zero external dependencies.
"""

import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .state import ResearchState

# streaming=True makes astream_events capture individual tokens from this LLM,
# which lets the frontend show live text as each agent "thinks".
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.4,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ── Planner ───────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a research planner. Your job is to break down any topic
into exactly 4 focused, specific research questions.

Return ONLY a valid JSON array of 4 strings — no explanation, no markdown fences.
Example output:
["What is X and when was it invented?", "How does X work technically?", "What are the main uses of X today?", "What challenges and limitations does X have?"]
"""


async def planner_node(state: ResearchState) -> dict:
    """
    Planner Agent — node 1 of 3.

    Input : state["topic"]
    Output: state["plan"]  — list of 4 research questions

    The LLM is told to return strict JSON so we can parse the questions
    and display them as a checklist in the frontend.
    """
    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])

    try:
        plan = json.loads(response.content)
        if not isinstance(plan, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract any lines that look like questions
        plan = [
            line.strip()
            for line in response.content.splitlines()
            if "?" in line and len(line.strip()) > 10
        ]
        if not plan:
            plan = [
                f"What is {state['topic']} and where did it come from?",
                f"How does {state['topic']} work?",
                f"What are the most important uses of {state['topic']}?",
                f"What are the limitations and challenges of {state['topic']}?",
            ]

    return {"plan": plan}


# ── Researcher ────────────────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a research analyst. You will receive a topic and
a numbered list of research questions. Answer each question in its own section.

Format your response exactly like this:

## 1. [Repeat the question here]
[A detailed, factual paragraph of 4-6 sentences with concrete information,
specific examples, numbers, and names where relevant.]

## 2. [Repeat the question here]
[Same depth and style.]

...and so on for every question.

Be specific and informative. Avoid vague generalities.
"""


async def researcher_node(state: ResearchState) -> dict:
    """
    Researcher Agent — node 2 of 3.

    Input : state["topic"], state["plan"]
    Output: state["research"]  — findings in structured markdown

    Receives all research questions at once and answers them in a single
    LLM call, which keeps streaming smooth and coherent.
    """
    questions_text = "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(state["plan"])
    )

    response = await llm.ainvoke([
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"Research questions:\n{questions_text}"
        )),
    ])

    return {"research": response.content}


# ── Writer ────────────────────────────────────────────────────────────────────

WRITER_SYSTEM = """You are a professional writer and journalist. Your job is to
take raw research findings and turn them into a polished, engaging article.

Guidelines:
- Write in a clear, confident, informative tone — like a quality magazine article
- Use ## headings for each major section
- Bold (**key term**) important concepts on first use
- Write flowing prose paragraphs — avoid bullet-point dumps
- Open with a compelling one-paragraph introduction that hooks the reader
- Close with a meaningful conclusion paragraph
- Target length: 500–700 words

Output only the article — no preamble like "Here is the article:".
"""


async def writer_node(state: ResearchState) -> dict:
    """
    Writer Agent — node 3 of 3.

    Input : state["topic"], state["research"]
    Output: state["article"]  — final article in markdown

    The writer never sees the raw questions — only the topic and the
    research findings.  This mirrors how a real editorial workflow works:
    the writer synthesizes, not just copies.
    """
    response = await llm.ainvoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"Research findings:\n\n{state['research']}"
        )),
    ])

    return {"article": response.content}
