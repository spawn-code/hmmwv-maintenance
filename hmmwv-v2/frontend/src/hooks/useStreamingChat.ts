/**
 * useStreamingChat — POST SSE streaming via fetch + ReadableStream.
 *
 * Uses fetch (not EventSource) because the chat endpoint requires
 * a POST body. Parses "data: {...}\n\n" SSE frames and dispatches
 * each event type to the appropriate Zustand store action.
 */
import { useCallback } from 'react'
import { useChatStore }    from '../store/useChatStore'
import { useSessionStore } from '../store/useSessionStore'
import type { ChatRequest, SSEEvent, SourceRef, ImageRef } from '../types'

export function useStreamingChat() {
  const {
    startStream, appendToken, updateAgentStep,
    setPendingSources, setPendingImages, finalizeStream, setError,
  } = useChatStore()
  const { appendMessages, refreshSessionSummary } = useSessionStore()

  const sendMessage = useCallback(async (
    request: ChatRequest,
    userContent: string,
  ) => {
    startStream()

    let fullContent = ''

    try {
      const resp = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      })

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`)
      }
      if (!resp.body) {
        throw new Error('Response has no body')
      }

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      let   buffer  = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Split on SSE double-newline delimiter
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''           // keep incomplete tail

        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') continue

          let event: SSEEvent
          try {
            event = JSON.parse(raw)
          } catch {
            continue
          }

          switch (event.type) {
            case 'token':
              fullContent += event.content ?? ''
              appendToken(event.content ?? '')
              break

            case 'agent_status':
              updateAgentStep(
                event.step  ?? '',
                event.label ?? '',
                event.done  ?? false,
                event.elapsed,
              )
              break

            case 'sources':
              setPendingSources((event.data ?? []) as SourceRef[])
              break

            case 'images':
              setPendingImages((event.data ?? []) as ImageRef[])
              break

            case 'done':
              break

            case 'error':
              setError(event.message ?? 'Unknown error')
              break
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      return
    }

    // Persist messages to session store (optimistic, no API call)
    const { pendingSources, pendingImages } = useChatStore.getState()
    const ts = new Date().toISOString()
    appendMessages(
      request.session_id,
      userContent,
      fullContent,
      pendingSources,
      pendingImages,
      ts,
    )

    // Reload session summary to get server-assigned title
    await refreshSessionSummary(request.session_id)

    finalizeStream()
  }, [
    startStream, appendToken, updateAgentStep,
    setPendingSources, setPendingImages, finalizeStream, setError,
    appendMessages, refreshSessionSummary,
  ])

  return { sendMessage }
}
