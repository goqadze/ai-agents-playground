import { useCallback, useEffect, useState } from 'react'
import type { Conversation, Message, StreamingMessage } from './types'
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  resumeChat,
  streamChat,
} from './api/client'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [streaming, setStreaming] = useState<StreamingMessage | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    listConversations().then(setConversations).catch(console.error)
  }, [])

  const loadConversation = useCallback(async (id: number) => {
    const data = await getConversation(id)
    setMessages(data.messages)
    setActiveId(id)
  }, [])

  const handleSelect = useCallback(
    (id: number) => {
      if (id === activeId) return
      setStreaming(null)
      loadConversation(id)
    },
    [activeId, loadConversation],
  )

  const handleCreate = useCallback(async () => {
    const conv = await createConversation()
    setConversations((prev) => [conv, ...prev])
    setMessages([])
    setStreaming(null)
    setActiveId(conv.id)
  }, [])

  const handleDelete = useCallback(
    async (id: number) => {
      await deleteConversation(id)
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (id === activeId) {
        setActiveId(null)
        setMessages([])
        setStreaming(null)
      }
    },
    [activeId],
  )

  // ── Consume a stream of SSE events and update streaming state ──────────────
  const consumeStream = useCallback(
    async (gen: AsyncGenerator<Record<string, unknown>>) => {
      let completedSteps: string[] = []

      for await (const event of gen) {
        if (event.type === 'step' && event.step) {
          completedSteps = [...completedSteps, event.step as string]
          setStreaming((prev) =>
            prev ? { ...prev, steps: completedSteps, currentStep: event.step as string } : null,
          )
        } else if (event.type === 'token' && event.content) {
          setStreaming((prev) =>
            prev
              ? { ...prev, content: prev.content + (event.content as string), currentStep: null }
              : null,
          )
        } else if (event.type === 'interrupt') {
          // Graph paused — show option buttons, keep streamed content visible
          setStreaming((prev) =>
            prev
              ? {
                  ...prev,
                  currentStep: null,
                  interrupt: {
                    question: event.question as string,
                    options:  event.options as string[],
                  },
                }
              : null,
          )
          setIsLoading(false)
          return  // stop consuming — user must pick an option first
        } else if (event.type === 'done') {
          setStreaming(null)
          if (activeId) {
            const data = await getConversation(activeId)
            setMessages(data.messages)
            setConversations((prev) =>
              prev.map((c) => (c.id === activeId ? { ...c, title: data.title } : c)),
            )
          }
        } else if (event.type === 'error') {
          console.error('Stream error:', event.message)
          setStreaming(null)
        }
      }
    },
    [activeId],
  )

  // ── Send a new message ─────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (!activeId || isLoading) return

      setMessages((prev) => [
        ...prev,
        { id: Date.now(), role: 'user', content: text, agent_steps: [], created_at: new Date().toISOString() },
      ])
      setIsLoading(true)
      setStreaming({ content: '', steps: [], currentStep: null, interrupt: null, done: false })

      try {
        await consumeStream(streamChat(activeId, text) as AsyncGenerator<Record<string, unknown>>)
      } catch (err) {
        console.error(err)
        setStreaming(null)
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, isLoading, consumeStream],
  )

  // ── User picks an option from the interrupt prompt ─────────────────────────
  const handleResume = useCallback(
    async (choice: string) => {
      if (!activeId) return

      // Replace the interrupt UI with a "choice badge" and start streaming again
      setStreaming((prev) =>
        prev ? { ...prev, interrupt: null, steps: [...prev.steps, `✓ ${choice}`], currentStep: null } : null,
      )
      setIsLoading(true)

      try {
        await consumeStream(resumeChat(activeId, choice) as AsyncGenerator<Record<string, unknown>>)
      } catch (err) {
        console.error(err)
        setStreaming(null)
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, consumeStream],
  )

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelect}
        onCreate={handleCreate}
        onDelete={handleDelete}
      />

      <main className="main">
        {activeId ? (
          <ChatWindow
            messages={messages}
            streaming={streaming}
            onSend={handleSend}
            onResume={handleResume}
            isLoading={isLoading}
          />
        ) : (
          <div className="welcome">
            <div className="welcome-icon">🤝</div>
            <h1>HITL Chat</h1>
            <p>
              Ask anything — the agent will acknowledge your question, then pause
              and ask how you'd like it answered before continuing.
            </p>
            <button className="welcome-btn" onClick={handleCreate}>
              Start a new conversation
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
