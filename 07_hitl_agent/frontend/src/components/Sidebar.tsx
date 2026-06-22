import type { Conversation } from '../types'

interface Props {
  conversations: Conversation[]
  activeId: number | null
  onSelect: (id: number) => void
  onCreate: () => void
  onDelete: (id: number) => void
}

export default function Sidebar({ conversations, activeId, onSelect, onCreate, onDelete }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-title">HITL Chat</span>
        <button className="new-btn" onClick={onCreate} title="New conversation">+</button>
      </div>

      <div className="sidebar-list">
        {conversations.length === 0 && (
          <p className="sidebar-empty">No conversations yet</p>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`sidebar-item ${c.id === activeId ? 'active' : ''}`}
            onClick={() => onSelect(c.id)}
          >
            <span className="sidebar-item-title">{c.title}</span>
            <button
              className="delete-btn"
              onClick={(e) => { e.stopPropagation(); onDelete(c.id) }}
              title="Delete"
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <span>LangGraph Human-in-the-Loop</span>
      </div>
    </aside>
  )
}
