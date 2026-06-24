import type { AgentState, AgentMeta } from '../types'

interface Props {
  meta: AgentMeta
  state: AgentState
}

const STATUS_LABELS: Record<string, string> = {
  idle:     'Waiting',
  thinking: 'Working…',
  done:     'Done',
  error:    'Error',
}

export default function AgentCard({ meta, state }: Props) {
  const { status, streamContent, plan } = state

  return (
    <div className={`agent-card agent-card--${status}`}>
      {/* Header */}
      <div className="agent-card__header">
        <span className="agent-card__emoji">{meta.emoji}</span>
        <div className="agent-card__title">
          <span className="agent-card__label">{meta.label}</span>
          <span className="agent-card__desc">{meta.description}</span>
        </div>
        <span className={`agent-card__badge agent-card__badge--${status}`}>
          {STATUS_LABELS[status]}
          {status === 'thinking' && <span className="pulse-dot" />}
        </span>
      </div>

      {/* Body */}
      {(status === 'thinking' || status === 'done') && (
        <div className="agent-card__body">

          {/* Planner done: show question list */}
          {meta.name === 'planner' && status === 'done' && plan && plan.length > 0 ? (
            <ol className="plan-list">
              {plan.map((q, i) => (
                <li key={i} className="plan-item">
                  <span className="plan-check">✓</span> {q}
                </li>
              ))}
            </ol>
          ) : (
            /* All agents while thinking, or researcher/writer when done */
            <div className="agent-card__stream">
              <pre className="stream-text">
                {streamContent || <span className="thinking-dots">●●●</span>}
              </pre>
            </div>
          )}

        </div>
      )}
    </div>
  )
}
