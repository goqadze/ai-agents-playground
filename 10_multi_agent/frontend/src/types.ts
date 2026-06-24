// Status of each agent card in the UI
export type AgentStatus = 'idle' | 'thinking' | 'done' | 'error'

// What each agent has produced so far (live during streaming)
export interface AgentState {
  status: AgentStatus
  streamContent: string   // text arriving token by token while thinking
  plan?: string[]         // planner final output
  research?: string       // researcher final output
  article?: string        // writer final output
}

// SSE event shapes received from the backend
export type SSEEvent =
  | { type: 'agent_start'; agent: AgentName }
  | { type: 'token';       agent: AgentName; content: string }
  | { type: 'agent_done';  agent: AgentName; output: Record<string, unknown> }
  | { type: 'done' }
  | { type: 'error'; message: string }

export type AgentName = 'planner' | 'researcher' | 'writer'

// Display metadata for each agent card
export interface AgentMeta {
  name: AgentName
  label: string
  emoji: string
  description: string
  outputLabel: string
}

export const AGENTS: AgentMeta[] = [
  {
    name: 'planner',
    label: 'Planner',
    emoji: '🗂️',
    description: 'Breaks your topic into focused research questions',
    outputLabel: 'Research Plan',
  },
  {
    name: 'researcher',
    label: 'Researcher',
    emoji: '🔍',
    description: 'Answers each question with detailed findings',
    outputLabel: 'Research Findings',
  },
  {
    name: 'writer',
    label: 'Writer',
    emoji: '✍️',
    description: 'Synthesizes findings into a polished article',
    outputLabel: 'Article Draft',
  },
]
