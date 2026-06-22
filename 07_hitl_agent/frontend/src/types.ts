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

// Interrupt payload — sent by backend when the graph pauses for user input
export interface InterruptData {
  question: string
  options: string[]
}

export interface StreamingMessage {
  content: string
  steps: string[]
  currentStep: string | null
  // When set, the agent has paused and is waiting for the user to pick an option
  interrupt: InterruptData | null
  done: boolean
}
