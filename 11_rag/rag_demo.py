"""
Simplest possible RAG demo.

Flow:
  1. Index a few sample documents into ChromaDB (in-memory, no file on disk)
  2. Accept a user question from the terminal
  3. Retrieve the top-3 most relevant chunks
  4. Send them to OpenAI as context
  5. Print the grounded answer

Run:
  python rag_demo.py
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Sample knowledge base — replace with your own documents
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "doc1",
        "text": (
            "RAG stands for Retrieval-Augmented Generation. It is a technique that combines "
            "information retrieval with text generation. Instead of relying solely on a language "
            "model's parametric knowledge, RAG retrieves relevant documents from an external "
            "knowledge base and uses them as context when generating an answer."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Vector embeddings convert text into high-dimensional numeric vectors. Similar texts "
            "produce vectors that are close together in vector space. This property allows us to "
            "search for semantically similar documents using cosine similarity, even when the "
            "exact words don't match."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "ChromaDB is an open-source vector database designed for AI applications. It stores "
            "documents alongside their embeddings and supports fast similarity search. ChromaDB "
            "can run entirely in memory (no file system) or persist to disk, making it ideal for "
            "prototyping RAG pipelines locally."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "Chunking is the process of splitting large documents into smaller pieces before "
            "indexing. This matters because embedding models have token limits and because "
            "smaller chunks lead to more precise retrieval. A common strategy is to use "
            "overlapping chunks (chunk_overlap) so that context is never cut off at a boundary."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "Hallucinations in LLMs are confident-sounding but factually incorrect outputs. "
            "RAG reduces hallucinations by grounding the model's answer in retrieved documents "
            "from a trusted knowledge base. The model is instructed to answer only from the "
            "provided context, not from its training data."
        ),
    },
    {
        "id": "doc6",
        "text": (
            "OpenAI offers a family of GPT models including GPT-4o, GPT-4 Turbo, and GPT-3.5 Turbo. "
            "These models can be accessed via the OpenAI API using the 'openai' Python SDK. "
            "GPT-4o supports a 128k token context window and is capable of text, vision, and audio. "
            "OpenAI also provides text-embedding-3-small and text-embedding-3-large for embeddings."
        ),
    },
]


def build_vector_store() -> chromadb.Collection:
    """Create an in-memory ChromaDB collection and index all documents."""
    # EphemeralClient = lives only in RAM, no files written
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="rag_demo",
        # ChromaDB's default embedding function uses sentence-transformers
        # (downloaded automatically on first run, ~80 MB)
    )

    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
    )

    print(f"✅ Indexed {len(DOCUMENTS)} documents into ChromaDB\n")
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[str]:
    """Return the top_k most relevant document chunks for the query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    # results["documents"] is a list-of-lists (one list per query)
    return results["documents"][0]


def generate(context_chunks: list[str], question: str) -> str:
    """Send the retrieved chunks + question to OpenAI and return the answer."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    context = "\n\n---\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using ONLY the context "
                    "provided below. If the answer is not in the context, say 'I don't have enough "
                    "information in the knowledge base to answer that.'\n\n"
                    f"CONTEXT:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


def rag(collection: chromadb.Collection, question: str) -> str:
    """Full RAG pipeline: retrieve → augment → generate."""
    print(f"🔍 Retrieving relevant chunks for: '{question}'")
    chunks = retrieve(collection, question)

    print(f"\n📄 Top {len(chunks)} retrieved chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk[:100]}…")

    print("\n🤖 Generating answer with OpenAI…\n")
    answer = generate(chunks, question)
    return answer


def main():
    collection = build_vector_store()

    sample_questions = [
        "What is RAG and how does it work?",
        "How does RAG reduce hallucinations?",
        "What is ChromaDB used for?",
    ]

    print("=" * 60)
    print("RAG DEMO — Type a question or press Enter for samples")
    print("=" * 60)

    while True:
        print("\nSample questions:")
        for i, q in enumerate(sample_questions, 1):
            print(f"  {i}. {q}")
        print("  0. Quit")

        user_input = input("\nYour question (or 1/2/3/0): ").strip()

        if user_input == "0" or user_input.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input in ("1", "2", "3"):
            question = sample_questions[int(user_input) - 1]
        elif user_input:
            question = user_input
        else:
            continue

        print(f"\n{'─' * 60}")
        answer = rag(collection, question)
        print(f"💬 Answer:\n{answer}")
        print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
