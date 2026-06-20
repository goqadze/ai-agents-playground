import { useEffect, useRef, useState } from 'react'
import type { Message, StreamingMessage } from '../types'
import MessageBubble from './MessageBubble'

interface Props {
  messages: Message[]
  streaming: StreamingMessage | null
  onSend: (text: string) => void
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

export default function ChatWindow({ messages, streaming, onSend, isLoading }: Props) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming?.content])

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

  return (
    <div className="chat-window">
      <div className="messages-area">
        {messages.length === 0 && !streaming && (
          <div className="empty-chat">
            <div className="empty-icon">💬</div>
            <h2>Start a conversation</h2>
            <p>Ask anything — simple or complex. The AI will route your question through the best pipeline.</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {streaming && (
          <div className="message-row assistant-row">
            <div className="message-bubble assistant-bubble">
              {streaming.currentStep && (
                <div className="agent-steps">
                  {streaming.steps.map((s, i) => (
                    <span key={i} className="step-badge done">{s}</span>
                  ))}
                  <span className="step-badge active">
                    <span className="spinner" /> {streaming.currentStep}
                  </span>
                </div>
              )}
              {streaming.steps.length > 0 && !streaming.currentStep && (
                <div className="agent-steps">
                  {streaming.steps.map((s, i) => (
                    <span key={i} className="step-badge done">{s}</span>
                  ))}
                </div>
              )}
              {streaming.content ? (
                <div
                  className="message-text prose"
                  dangerouslySetInnerHTML={{ __html: `<p>${renderMarkdown(streaming.content)}</p>` }}
                />
              ) : (
                <span className="typing-indicator">
                  <span /><span /><span />
                </span>
              )}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-input"
            placeholder="Ask anything... (Shift+Enter for newline)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? '...' : '↑'}
          </button>
        </div>
      </div>
    </div>
  )
}
