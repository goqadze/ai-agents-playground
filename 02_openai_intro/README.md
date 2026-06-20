# 02 — LangChain + OpenAI Intro

A minimal LangChain app that calls OpenAI so you can learn how the core building blocks fit together.

---

## What the app does

1. Loads your OpenAI API key from `.env`
2. Creates a **model**, a **prompt template**, and a **chain**
3. Explains two topics automatically, then lets you type your own

---

## Core concepts

| Concept | What it is |
|---|---|
| **`ChatOpenAI`** | LangChain's wrapper around OpenAI's chat model. You configure the model name and temperature here. |
| **`ChatPromptTemplate`** | A reusable prompt with `{placeholders}`. You fill them in at runtime with `.invoke()`. |
| **`StrOutputParser`** | Extracts the plain text string from the model's response object so you don't have to. |
| **Chain (`\|`)** | The pipe operator connects steps: `prompt \| model \| parser`. Data flows left to right. |
| **`.invoke({"key": "value"})`** | Runs the full chain with the given inputs and returns the final output. |

---

## Setup

### 1. Add your API key

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-real-key-here...
```

Get a key at <https://platform.openai.com/api-keys>.

### 2. Install dependencies

```bash
cd 02_openai_intro
uv sync
```

---

## Run

```bash
uv run python main.py
```

---

## Project layout

```
02_openai_intro/
├── .env            ← your secret API key (never commit this)
├── main.py         ← all the code, heavily commented
├── pyproject.toml  ← dependencies (langchain, langchain-openai, python-dotenv)
└── README.md       ← this file
```
