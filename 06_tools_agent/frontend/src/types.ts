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

// Streaming message being built token-by-token
export interface StreamingMessage {
  content: string
  steps: string[]       // node step labels (grey badges)
  tools: string[]       // tool names the agent called (green badges)
  currentStep: string | null
  done: boolean
}
