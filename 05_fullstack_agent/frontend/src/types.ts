export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  agent_steps: string[]
  created_at: string
}

export interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[]
}

// Streaming message being built
export interface StreamingMessage {
  content: string
  steps: string[]
  currentStep: string | null
  done: boolean
}
