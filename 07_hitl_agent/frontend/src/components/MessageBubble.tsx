import type { Message } from '../types'

interface Props {
  message: Message
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

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        {!isUser && message.agent_steps.length > 0 && (
          <div className="agent-steps">
            {message.agent_steps.map((step, i) => (
              <span key={i} className="step-badge">{step}</span>
            ))}
          </div>
        )}
        {isUser ? (
          <p className="message-text">{message.content}</p>
        ) : (
          <div
            className="message-text prose"
            dangerouslySetInnerHTML={{ __html: `<p>${renderMarkdown(message.content)}</p>` }}
          />
        )}
      </div>
    </div>
  )
}
