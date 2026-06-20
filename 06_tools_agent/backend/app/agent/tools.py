"""
LangChain tools for the agent.

Each function decorated with @tool becomes a tool the LLM can call.
The LLM reads the function name + docstring to decide WHEN to call it,
and reads the parameter names + types to know WHAT to pass.

Rule: write the docstring for the LLM, not for humans.
Clear, specific docstrings = the LLM picks the right tool at the right time.
"""

import math
from datetime import datetime
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any arithmetic: addition, subtraction, multiplication,
    division, powers, square roots, percentages, etc.
    Examples: '2 + 2', '15 * 7', 'sqrt(144)', '2 ** 10', '(3 + 4) * 5'
    """
    try:
        # Allow only safe math functions — never eval arbitrary user code
        allowed = {
            "sqrt": math.sqrt,
            "pow":  math.pow,
            "abs":  abs,
            "round": round,
            "pi":   math.pi,
            "e":    math.e,
            "log":  math.log,
            "log10": math.log10,
            "sin":  math.sin,
            "cos":  math.cos,
            "tan":  math.tan,
            "ceil": math.ceil,
            "floor": math.floor,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"{result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def word_count(text: str) -> str:
    """
    Count the number of words, characters, and sentences in a text.
    Use this when asked to count or analyze the length of any piece of text.
    """
    words     = len(text.split())
    chars     = len(text)
    chars_nsp = len(text.replace(" ", ""))
    sentences = len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])
    return (
        f"Words: {words} | "
        f"Characters (with spaces): {chars} | "
        f"Characters (no spaces): {chars_nsp} | "
        f"Sentences: {sentences}"
    )


@tool
def get_current_datetime() -> str:
    """
    Return the current date and time.
    Use this whenever the user asks what time or date it is,
    or when the answer depends on today's date.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %H:%M:%S")


@tool
def reverse_text(text: str) -> str:
    """
    Reverse the characters in a text string.
    Use this when asked to reverse, flip, or mirror text.
    """
    return text[::-1]


@tool
def text_transform(text: str, mode: str) -> str:
    """
    Transform text case or style.
    mode options:
      - 'upper'      → ALL CAPS
      - 'lower'      → all lowercase
      - 'title'      → Title Case
      - 'snake'      → snake_case
      - 'count_vowels' → count the vowel letters
    Use this when asked to change the case or style of text.
    """
    mode = mode.strip().lower()
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    if mode == "title":
        return text.title()
    if mode == "snake":
        return text.lower().replace(" ", "_")
    if mode == "count_vowels":
        count = sum(1 for c in text.lower() if c in "aeiou")
        return f"{count} vowels in '{text}'"
    return f"Unknown mode '{mode}'. Choose: upper, lower, title, snake, count_vowels"


# Collected list exported to the rest of the app
ALL_TOOLS = [calculator, word_count, get_current_datetime, reverse_text, text_transform]
