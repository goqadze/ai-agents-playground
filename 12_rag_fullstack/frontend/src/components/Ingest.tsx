/**
 * Ingest panel — lets the user add documents to the pgvector knowledge base.
 *
 * Two paths:
 *  1. Paste raw text + give it a source name → POST /ingest/text
 *  2. Upload a .txt file              → POST /ingest/file
 *
 * The LangGraph RAG graph reads from the same collection, so anything
 * ingested here is immediately available when chatting.
 */

import { useState } from "react";

const INGEST_URL = import.meta.env.VITE_INGEST_URL ?? "http://localhost:8000";

export default function Ingest() {
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [docCount, setDocCount] = useState<number | null>(null);
  const [status, setStatus] = useState<{ msg: string; ok: boolean } | null>(null);
  const [loading, setLoading] = useState(false);

  async function refreshCount() {
    try {
      const res = await fetch(`${INGEST_URL}/documents`);
      const data = await res.json();
      setDocCount(data.count);
    } catch {
      // ignore
    }
  }

  async function handleTextIngest() {
    if (!text.trim()) return;
    setLoading(true);
    setStatus(null);
    try {
      const res = await fetch(`${INGEST_URL}/ingest/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, source: source.trim() || "manual" }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStatus({ msg: `Indexed ${data.chunks_indexed} chunks from "${data.source}"`, ok: true });
      setText("");
      setSource("");
      refreshCount();
    } catch (e: unknown) {
      setStatus({ msg: `Error: ${e instanceof Error ? e.message : String(e)}`, ok: false });
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setStatus(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${INGEST_URL}/ingest/file`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStatus({ msg: `Indexed ${data.chunks_indexed} chunks from "${data.source}"`, ok: true });
      e.target.value = "";
      refreshCount();
    } catch (e: unknown) {
      setStatus({ msg: `Error: ${e instanceof Error ? e.message : String(e)}`, ok: false });
    } finally {
      setLoading(false);
    }
  }

  return (
    <aside className="ingest-panel">
      <h2 className="panel-title">Knowledge Base</h2>

      {docCount !== null && (
        <p className="doc-count">{docCount} chunk{docCount !== 1 ? "s" : ""} indexed</p>
      )}
      <button className="btn-ghost" onClick={refreshCount}>
        Refresh count
      </button>

      <hr className="divider" />

      <section>
        <h3>Upload .txt file</h3>
        <input
          type="file"
          accept=".txt"
          onChange={handleFileUpload}
          disabled={loading}
          className="file-input"
        />
      </section>

      <hr className="divider" />

      <section>
        <h3>Paste text</h3>
        <input
          className="text-input"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="Source name (e.g. company-faq)"
          disabled={loading}
        />
        <textarea
          className="textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste any text here — it will be chunked, embedded, and stored in pgvector."
          rows={8}
          disabled={loading}
        />
        <button
          className="btn-primary"
          onClick={handleTextIngest}
          disabled={loading || !text.trim()}
        >
          {loading ? "Indexing…" : "Add to knowledge base"}
        </button>
      </section>

      {status && (
        <p className={`status-msg ${status.ok ? "ok" : "err"}`}>{status.msg}</p>
      )}

      <hr className="divider" />

      <details className="flow-diagram">
        <summary>How RAG works</summary>
        <pre>{`
Ingest phase (this panel):
  Text → Chunking → Embedding → pgvector

Query phase (chat):
  Question → Embed → pgvector search
  → Top-k chunks → LLM prompt → Answer
        `.trim()}</pre>
      </details>
    </aside>
  );
}
