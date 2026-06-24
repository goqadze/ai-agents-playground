import type { SSEEvent } from '../types'

/**
 * POST /api/research and read the SSE stream.
 *
 * Calls onEvent for every parsed SSE event until the stream ends or
 * onError is called.  Returns a cleanup function that aborts the fetch.
 */
export function startResearch(
  topic: string,
  onEvent: (event: SSEEvent) => void,
  onError: (message: string) => void,
): () => void {
  const controller = new AbortController()

  ;(async () => {
    let response: Response
    try {
      response = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic }),
        signal: controller.signal,
      })
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        onError('Failed to connect to backend.')
      }
      return
    }

    if (!response.ok || !response.body) {
      onError(`Backend error: ${response.status}`)
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // SSE lines are separated by \n\n
      const parts = buffer.split('\n\n')
      buffer = parts.pop() ?? ''

      for (const part of parts) {
        const line = part.trim()
        if (!line.startsWith('data:')) continue

        const json = line.slice('data:'.length).trim()
        try {
          const event = JSON.parse(json) as SSEEvent
          onEvent(event)
        } catch {
          // malformed line — skip
        }
      }
    }
  })()

  return () => controller.abort()
}
