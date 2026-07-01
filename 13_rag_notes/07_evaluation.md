# RAG Evaluation

Without evaluation you're guessing. You might change the chunking strategy, reranker, or prompt and have no idea if it actually helped.

---

## What to Measure

```
RAG pipeline has two parts to evaluate:

1. Retrieval quality
   Did we get the right chunks?

2. Generation quality
   Given the chunks, did the LLM answer correctly?
```

### The 4 Core Metrics

| Metric | Question | Measures |
|--------|----------|---------|
| **Context Precision** | Are the retrieved chunks actually relevant? | Retrieval quality |
| **Context Recall** | Did we retrieve all the relevant content? | Retrieval completeness |
| **Faithfulness** | Does the answer stick to the retrieved context? | Hallucination |
| **Answer Relevancy** | Does the answer actually address the question? | Generation quality |

---

## RAGAS — Automated RAG Evaluation

RAGAS evaluates your pipeline without needing human-labeled data. It uses an LLM to score each metric.

```python
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Your test data
data = {
    "question": ["What is the return policy?", "How long does shipping take?"],
    "answer": ["Returns are accepted within 30 days.", "Shipping takes 3-5 business days."],
    "contexts": [
        ["Returns are accepted within 30 days of purchase date."],
        ["Standard shipping: 3-5 business days. Express: 1-2 days."],
    ],
    "ground_truth": ["30 days", "3-5 business days"],
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# {'faithfulness': 0.97, 'answer_relevancy': 0.91, 'context_precision': 0.88, 'context_recall': 0.85}
```

---

## What Each Score Means

### Faithfulness
Does the answer contain only information from the retrieved context?

```
Context:  "Returns are accepted within 30 days."
Answer:   "You can return items within 30 days."      → 1.0 (faithful)
Answer:   "You can return items within 60 days."      → 0.0 (hallucinated)
Answer:   "Returns are 30 days. No restocking fee."   → 0.5 (partial hallucination)
```

**Target: > 0.85**

### Context Precision
Of the retrieved chunks, how many were actually useful?

```
Retrieved 5 chunks, 3 were relevant to the query → precision = 3/5 = 0.60
Retrieved 5 chunks, 5 were relevant               → precision = 5/5 = 1.00
```

Low precision = too much noise sent to LLM. Fix: better retrieval, reranking, threshold.

### Context Recall
Of all relevant content in the KB, how much was retrieved?

```
KB has 4 relevant chunks, you retrieved 3 of them → recall = 3/4 = 0.75
```

Low recall = missing important context. Fix: higher K, multi-query, better chunking.

### Answer Relevancy
Does the answer address the question asked?

```
Question: "What is the return window?"
Answer:   "Our return policy was updated in 2024."  → low relevancy (doesn't answer)
Answer:   "You can return items within 30 days."    → high relevancy
```

---

## Building a Test Set

You need question-answer pairs to evaluate against. Three ways:

### 1. Manual (Gold standard)
Write 20–50 questions you know the answers to from your docs. Time-consuming but accurate.

### 2. RAGAS Synthetic Generation
```python
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o"),
    critic_llm=ChatOpenAI(model="gpt-4o"),
    embeddings=OpenAIEmbeddings(),
)

testset = generator.generate_with_langchain_docs(documents, test_size=20)
```

### 3. Production logs
Take real user questions + correct answers from your support team. Best signal for production systems.

---

## LangSmith (Tracing + Evaluation)

LangSmith traces every step of your pipeline — see exactly what was retrieved and why.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "rag-evaluation"

# All LangChain calls are now traced automatically
result = rag_chain.invoke("What is the return policy?")
# → visible in smith.langchain.com with full trace
```

In LangSmith you can see:
- Which chunks were retrieved (and their scores)
- Exact prompt sent to LLM
- LLM output
- Latency at each step

---

## Practical Evaluation Loop

```
1. Pick 20 representative questions
2. Run pipeline → collect (question, retrieved_chunks, answer)
3. Score with RAGAS
4. Identify the weakest metric
5. Make one change (chunking / retrieval / prompt)
6. Re-run → compare scores
7. Repeat
```

Don't change multiple things at once — you won't know what helped.

---

## Target Scores (Rough Benchmarks)

| Metric | Acceptable | Good | Excellent |
|--------|-----------|------|-----------|
| Faithfulness | > 0.75 | > 0.85 | > 0.95 |
| Answer Relevancy | > 0.70 | > 0.80 | > 0.90 |
| Context Precision | > 0.60 | > 0.75 | > 0.85 |
| Context Recall | > 0.65 | > 0.75 | > 0.90 |
