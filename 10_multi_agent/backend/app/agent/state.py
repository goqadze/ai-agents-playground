"""
State shared across all agents in the pipeline.

Each agent reads from this state and writes its output back.
The state flows through the graph:  Planner → Researcher → Writer
"""

from typing import TypedDict


class ResearchState(TypedDict):
    topic: str          # user's original question / topic
    plan: list[str]     # Planner output  — research questions to answer
    research: str       # Researcher output — findings in markdown
    article: str        # Writer output    — final polished article in markdown
