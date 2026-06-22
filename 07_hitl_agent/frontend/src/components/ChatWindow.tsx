import { useEffect, useRef, useState } from 'react'
import type { Message, StreamingMessage } from '../types'
import MessageBubble from './MessageBubble'

interface Props {
  messages:  Message[]
  streaming: StreamingMessage | null
  onSend:    (text: string) => void
  onResume:  (choice: string) => void  // called when user picks an option
  isLoading: boolean
}

function renderMarkdown(text: string): string {
  return text
    .replace(/```([\w]*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
}

export default function ChatWindow({ messages, streaming, onSend, onResume, isLoading }: Props) {
  const [input, setInput]   = useState('')
  const bottomRef           = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming?.content, streaming?.interrupt])

  const handleSubmit = () => {
    const text = input.trim()
    if (!text || isLoading) return
    setInput('')
    onSend(text)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // Input is disabled while loading OR while waiting for the user to pick an option
  const inputDisabled = isLoading || streaming?.interrupt != null

  return (
    <div className="chat-window">
      <div className="messages-area">
        {messages.length === 0 && !streaming && (
          <div className="empty-chat">
            <div className="empty-icon">🤝</div>
            <h2>Start a conversation</h2>
            <p>Ask anything — the agent will pause halfway and ask how you'd like it answered.</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Live streaming bubble */}
        {streaming && (
          <div className="message-row assistant-row">
            <div className="message-bubble assistant-bubble">

              {/* Step badges */}
              {(streaming.steps.length > 0 || streaming.currentStep) && (
                <div className="agent-steps">
                  {streaming.steps.map((s, i) => (
                    <span key={i} className="step-badge done">{s}</span>
                  ))}
                  {streaming.currentStep && (
                    <span className="step-badge active">
                      <span className="spinner" /> {streaming.currentStep}
                    </span>
                  )}
                </div>
              )}

              {/* Streamed text so far */}
              {streaming.content ? (
                <div
                  className="message-text prose"
                  dangerouslySetInnerHTML={{ __html: `<p>${renderMarkdown(streaming.content)}</p>` }}
                />
              ) : !streaming.interrupt ? (
                <span className="typing-indicator"><span /><span /><span /></span>
              ) : null}

              {/* ── Interrupt: option picker ─────────────────────────────── */}
              {streaming.interrupt && (
                <div className="interrupt-block">
                  <p className="interrupt-question">{streaming.interrupt.question}</p>
                  <div className="interrupt-options">
                    {streaming.interrupt.options.map((opt) => (
                      <button
                        key={opt}
                        className="option-btn"
                        onClick={() => onResume(opt)}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </div>
              )}

            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-area">
        <div className={`input-wrapper ${inputDisabled ? 'input-disabled' : ''}`}>
          <textarea
            className="chat-input"
            placeholder={
              streaming?.interrupt
                ? 'Pick an option above to continue...'
                : 'Ask anything... (Shift+Enter for newline)'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={inputDisabled}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={inputDisabled || !input.trim()}
          >
            {isLoading && !streaming?.interrupt ? '...' : '↑'}
          </button>
        </div>
      </div>
    </div>
  )
}
