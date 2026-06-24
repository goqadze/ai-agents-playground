import { useState } from 'react'

interface Props {
  onSubmit: (topic: string) => void
  disabled: boolean
}

const EXAMPLES = [
  'Quantum computing',
  'The history of the internet',
  'How black holes work',
  'The future of renewable energy',
  'Machine learning fundamentals',
]

export default function TopicForm({ onSubmit, disabled }: Props) {
  const [topic, setTopic] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = topic.trim()
    if (trimmed) onSubmit(trimmed)
  }

  return (
    <div className="topic-form-wrapper">
      <form className="topic-form" onSubmit={handleSubmit}>
        <input
          type="text"
          className="topic-input"
          placeholder="Enter any topic — e.g. 'How GPS works'"
          value={topic}
          onChange={e => setTopic(e.target.value)}
          disabled={disabled}
          autoFocus
        />
        <button
          type="submit"
          className="research-btn"
          disabled={disabled || !topic.trim()}
        >
          {disabled ? 'Researching…' : 'Research'}
        </button>
      </form>

      {!disabled && (
        <div className="examples">
          <span className="examples-label">Try:</span>
          {EXAMPLES.map(ex => (
            <button
              key={ex}
              className="example-chip"
              onClick={() => { setTopic(ex); onSubmit(ex) }}
            >
              {ex}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
