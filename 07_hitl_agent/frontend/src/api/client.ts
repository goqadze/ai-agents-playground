import type { Conversation, ConversationWithMessages } from '../types'

const BASE = '/api'

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch(`${BASE}/conversations`)
  if (!res.ok) throw new Error('Failed to load conversations')
  return res.json()
}

export async function createConversation(): Promise<Conversation> {
  const res = await fetch(`${BASE}/conversations`, { method: 'POST' })
  if (!res.ok) throw new Error('Failed to create conversation')
  return res.json()
}

export async function getConversation(id: number): Promise<ConversationWithMessages> {
  const res = await fetch(`${BASE}/conversations/${id}`)
  if (!res.ok) throw new Error('Failed to load conversation')
  return res.json()
}

export async function deleteConversation(id: number): Promise<void> {
  await fetch(`${BASE}/conversations/${id}`, { method: 'DELETE' })
}

// Starts a new chat turn — streams until interrupt or done
export async function* streamChat(
  conversationId: number,
  message: string,
): AsyncGenerator<{ type: string; content?: string; step?: string; question?: string; options?: string[]; message?: string }> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  if (!res.ok || !res.body) throw new Error('Stream failed')
  yield* readSSE(res.body)
}

// Resumes a paused graph with the user's chosen option
export async function* resumeChat(
  conversationId: number,
  choice: string,
): AsyncGenerator<{ type: string; content?: string; step?: string; message?: string }> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ choice }),
  })
  if (!res.ok || !res.body) throw new Error('Resume failed')
  yield* readSSE(res.body)
}

// Shared SSE reader — used by both streamChat and resumeChat
async function* readSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<Record<string, unknown>> {
  const reader  = body.getReader()
  const decoder = new TextDecoder()
  let buffer    = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6))
        } catch {
          // skip malformed lines
        }
      }
    }
  }
}
