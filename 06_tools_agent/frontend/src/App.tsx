import { useCallback, useEffect, useState } from 'react'
import type { Conversation, Message, StreamingMessage } from './types'
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
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

  const handleSend = useCallback(
    async (text: string) => {
      if (!activeId || isLoading) return

      const optimisticUserMsg: Message = {
        id: Date.now(),
        role: 'user',
        content: text,
        agent_steps: [],
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, optimisticUserMsg])
      setIsLoading(true)
      setStreaming({ content: '', steps: [], tools: [], currentStep: null, done: false })

      try {
        let completedSteps: string[] = []
        let completedTools: string[] = []

        for await (const event of streamChat(activeId, text)) {
          if (event.type === 'step' && event.step) {
            completedSteps = [...completedSteps]
            setStreaming((prev) =>
              prev ? { ...prev, steps: completedSteps, currentStep: event.step! } : null,
            )
          } else if (event.type === 'tool' && event.tool) {
            // A tool was called — add it to the tool badges
            completedTools = [...completedTools, event.tool]
            setStreaming((prev) =>
              prev ? { ...prev, tools: completedTools } : null,
            )
          } else if (event.type === 'token' && event.content) {
            setStreaming((prev) =>
              prev
                ? {
                    ...prev,
                    content: prev.content + event.content!,
                    steps: completedSteps,
                    currentStep: null,
                  }
                : null,
            )
          } else if (event.type === 'done') {
            setStreaming(null)
            const data = await getConversation(activeId)
            setMessages(data.messages)
            setConversations((prev) =>
              prev.map((c) => (c.id === activeId ? { ...c, title: data.title } : c)),
            )
          } else if (event.type === 'error') {
            console.error('Stream error:', event.message)
            setStreaming(null)
          }
        }
      } catch (err) {
        console.error('Chat error:', err)
        setStreaming(null)
      } finally {
        setIsLoading(false)
      }
    },
    [activeId, isLoading],
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
            isLoading={isLoading}
          />
        ) : (
          <div className="welcome">
            <div className="welcome-icon">🔧</div>
            <h1>ToolChat</h1>
            <p>
              An AI assistant with LangChain tools — ask it to do math, count words,
              get the time, reverse text, and more.
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
