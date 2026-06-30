/**
 * Chat panel — connects to the LangGraph RAG agent via @langchain/react useStream.
 *
 * useStream() handles:
 *  - thread creation (a new conversation thread on first message)
 *  - streaming tokens from the LangGraph server as they arrive
 *  - message state accumulation across turns
 *
 * The RAG agent (graph.py) runs:
 *   retrieve → embed query → pgvector similarity search → top-3 chunks
 *   generate → build context prompt → GPT-4.1-nano → streamed response
 */

import { useStream } from "@langchain/react";
import { useEffect, useRef, useState } from "react";

const LANGGRAPH_URL = import.meta.env.VITE_LANGGRAPH_URL ?? "http://localhost:8123";

// Shape of the LangGraph RAG state (mirrors rag_agent/graph.py RAGState)
interface RAGState {
  messages: Array<{
    id: string;
    type: string;
    content: string | Array<{ type: string; text?: string }>;
  }>;
  context: string[];
}

function messageText(content: RAGState["messages"][number]["content"]): string {
  if (typeof content === "string") return content;
  return content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("");
}

export default function Chat() {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // useStream connects to the LangGraph server at LANGGRAPH_URL.
  // assistantId "rag" matches LANGSERVE_GRAPHS key in the server Dockerfile.
  const stream = useStream<RAGState>({
    apiUrl: LANGGRAPH_URL,
    assistantId: "rag",
  });

  // Scroll to latest message whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [stream.messages]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || stream.isLoading) return;
    // submit() dispatches a new run on the bound thread.
    // The input shape must match the graph's state keys.
    stream.submit({ messages: [{ role: "human", content: input.trim() }] });
    setInput("");
  }

  const visibleMessages = stream.messages.filter(
    (m) => m.type === "human" || m.type === "ai"
  );

  return (
    <main className="chat-panel">
      <h2 className="panel-title">Chat with your documents</h2>

      {stream.threadId && (
        <p className="thread-id">Thread: {stream.threadId}</p>
      )}

      <div className="messages">
        {visibleMessages.length === 0 && (
          <div className="empty-state">
            Add documents in the panel on the left, then ask a question here.
          </div>
        )}

        {visibleMessages.map((msg) => (
          <div
            key={msg.id}
            className={`message ${msg.type === "human" ? "message-human" : "message-ai"}`}
          >
            <span className="message-role">
              {msg.type === "human" ? "You" : "Assistant"}
            </span>
            <p className="message-content">{messageText(msg.content)}</p>
          </div>
        ))}

        {stream.isLoading && (
          <div className="message message-ai">
            <span className="message-role">Assistant</span>
            <p className="message-content thinking">Thinking…</p>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {stream.error && (
        <p className="status-msg err">
          Error: {stream.error instanceof Error ? stream.error.message : String(stream.error)}
        </p>
      )}

      <form onSubmit={handleSubmit} className="input-form">
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your documents…"
          disabled={stream.isLoading}
        />
        <button
          className="btn-primary"
          type="submit"
          disabled={stream.isLoading || !input.trim()}
        >
          Send
        </button>
      </form>
    </main>
  );
}
