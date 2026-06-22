"""
Agent state for the HITL (Human-in-the-Loop) agent.

New fields vs project 06:
  - intro_text  : the brief acknowledgment the agent streams before pausing
  - user_choice : the option the user picked at the interrupt point
                  (empty string until the user responds)
"""

from typing import TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages:    list[BaseMessage]  # conversation history loaded from DB
    question:    str                # current user question
    intro_text:  str                # set by intro_node
    user_choice: str                # set after interrupt() resolves
    answer:      str                # set by respond_node
