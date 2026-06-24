import { useState, useCallback } from 'react'
import TopicForm from './components/TopicForm'
import AgentCard from './components/AgentCard'
import ArticlePanel from './components/ArticlePanel'
import { startResearch } from './api/client'
import { AGENTS } from './types'
import type { AgentState, AgentName, SSEEvent } from './types'
import './App.css'

// Initial state for all three agent cards
const INITIAL_AGENTS: Record<AgentName, AgentState> = {
  planner:    { status: 'idle', streamContent: '' },
  researcher: { status: 'idle', streamContent: '' },
  writer:     { status: 'idle', streamContent: '' },
}

export default function App() {
  const [isRunning, setIsRunning]   = useState(false)
  const [currentTopic, setTopic]    = useState('')
  const [agents, setAgents]         = useState<Record<AgentName, AgentState>>(INITIAL_AGENTS)
  const [article, setArticle]       = useState('')
  const [error, setError]           = useState<string | null>(null)

  // Helper: update one agent's state immutably
  const updateAgent = useCallback(
    (name: AgentName, patch: Partial<AgentState>) => {
      setAgents(prev => ({
        ...prev,
        [name]: { ...prev[name], ...patch },
      }))
    },
    [],
  )

  const handleSubmit = useCallback((topic: string) => {
    // Reset everything
    setTopic(topic)
    setAgents(INITIAL_AGENTS)
    setArticle('')
    setError(null)
    setIsRunning(true)

    const handleEvent = (event: SSEEvent) => {
      switch (event.type) {
        case 'agent_start':
          updateAgent(event.agent, { status: 'thinking', streamContent: '' })
          break

        case 'token':
          setAgents(prev => ({
            ...prev,
            [event.agent]: {
              ...prev[event.agent],
              streamContent: prev[event.agent].streamContent + event.content,
            },
          }))
          break

        case 'agent_done': {
          const output = event.output as Record<string, unknown>
          const patch: Partial<AgentState> = { status: 'done' }

          if (event.agent === 'planner' && Array.isArray(output.plan)) {
            patch.plan = output.plan as string[]
          }
          if (event.agent === 'researcher' && typeof output.research === 'string') {
            patch.research = output.research
          }
          if (event.agent === 'writer' && typeof output.article === 'string') {
            patch.article = output.article
            setArticle(output.article)
          }

          updateAgent(event.agent, patch)
          break
        }

        case 'done':
          setIsRunning(false)
          break

        case 'error':
          setError(event.message)
          setIsRunning(false)
          break
      }
    }

    startResearch(topic, handleEvent, (msg) => {
      setError(msg)
      setIsRunning(false)
    })
  }, [updateAgent])

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-logo">🤖</div>
          <div>
            <h1 className="header-title">Multi-Agent Research Pipeline</h1>
            <p className="header-sub">
              Three specialized AI agents collaborate to research and write about any topic
            </p>
          </div>
        </div>
      </header>

      <main className="app-main">
        {/* Topic input */}
        <TopicForm onSubmit={handleSubmit} disabled={isRunning} />

        {/* Pipeline visualization — only shown after first submit */}
        {currentTopic && (
          <>
            {/* Arrow pipeline header */}
            <div className="pipeline-header">
              <span className="pipeline-topic">"{currentTopic}"</span>
              <div className="pipeline-flow">
                {AGENTS.map((meta, i) => (
                  <span key={meta.name} className="pipeline-flow__item">
                    <span className={`pipeline-node pipeline-node--${agents[meta.name].status}`}>
                      {meta.emoji} {meta.label}
                    </span>
                    {i < AGENTS.length - 1 && <span className="pipeline-arrow">→</span>}
                  </span>
                ))}
              </div>
            </div>

            {/* Agent cards */}
            <div className="pipeline-cards">
              {AGENTS.map(meta => (
                <AgentCard
                  key={meta.name}
                  meta={meta}
                  state={agents[meta.name]}
                />
              ))}
            </div>

            {/* Error */}
            {error && (
              <div className="error-banner">
                ⚠️ {error}
              </div>
            )}

            {/* Final article */}
            <ArticlePanel topic={currentTopic} article={article} />
          </>
        )}
      </main>
    </div>
  )
}
