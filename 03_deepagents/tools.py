"""
Custom tools for our Deep Agent.

A tool is just a plain Python function.
Deep Agents (like LangChain) reads the function NAME and DOCSTRING to decide:
  - what the tool does
  - when to use it

You don't need any decorator here — just pass the function list to
create_deep_agent(tools=[...]) and the framework handles the rest.
"""


def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


def word_count(text: str) -> int:
    """Count the number of words in a given text string."""
    return len(text.split())


def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns a weather description."""
    # Fake data — in a real app you'd call a weather API here
    fake_weather = {
        "london":    "Cloudy, 14°C",
        "new york":  "Sunny, 22°C",
        "tokyo":     "Rainy, 18°C",
        "tbilisi":   "Sunny, 25°C",
    }
    return fake_weather.get(city.lower(), f"Weather data not available for {city}.")


# All tools passed to the agent
ALL_TOOLS = [add, multiply, word_count, get_weather]
