import { useState, useRef, useEffect } from 'react'
import { useSessionStore }   from '../../store/useSessionStore'
import { useChatStore }      from '../../store/useChatStore'
import { useStreamingChat }  from '../../hooks/useStreamingChat'
import MessageList           from './MessageList'
import ChatInputBar          from '../input/ChatInputBar'
import type { ChatInputBarHandle } from '../input/ChatInputBar'

export default function ChatPage() {
  const { activeSession, activeSessionId, createSession, selectSession } = useSessionStore()
  const { isStreaming, resetStream } = useChatStore()
  const { sendMessage } = useStreamingChat()

  const [input,          setInput]          = useState('')
  const [deepAnalysis,   setDeepAnalysis]   = useState(false)
  const [vehicleVariant, setVehicleVariant] = useState('')
  const [category,       setCategory]       = useState('')

  const inputRef = useRef<ChatInputBarHandle>(null)

  // Auto-focus input on mount / session change
  useEffect(() => {
    inputRef.current?.focus()
  }, [activeSessionId])

  // Ensure we always have an active session
  const ensureSession = async (): Promise<string | null> => {
    if (activeSessionId) return activeSessionId
    const s = await createSession()
    if (!s) return null
    selectSession(s.id)
    return s.id
  }

  const handleSubmit = async () => {
    const query = input.trim()
    if (!query || isStreaming) return

    const sessionId = await ensureSession()
    if (!sessionId) return

    setInput('')
    resetStream()

    await sendMessage(
      {
        session_id:          sessionId,
        query,
        vehicle_variant:     vehicleVariant,
        maintenance_category: category,
        deep_analysis:       deepAnalysis,
      },
      query,
    )

    inputRef.current?.focus()
  }

  const handleCategoryClick = (text: string) => {
    setInput(text)
    inputRef.current?.focus()
  }

  // Session title for topbar
  const sessionTitle = activeSession?.title || 'NEW DIAGNOSIS'

  return (
    <div className="flex flex-col h-full">
      {/* Topbar */}
      <div
        className="flex items-center justify-between px-6 py-3 border-b shrink-0"
        style={{ borderColor: 'var(--color-border)' }}
      >
        <span
          className="font-mono text-sm tracking-widest uppercase"
          style={{ color: 'var(--color-muted)' }}
        >
          {sessionTitle}
        </span>

        {/* Cloud status pill */}
        <div
          className="flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono"
          style={{
            backgroundColor: 'var(--color-bg-surface)',
            border: '1px solid var(--color-border)',
          }}
        >
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: 'var(--color-olive)' }}
          />
          <span style={{ color: 'var(--color-muted)' }}>CLOUD</span>
        </div>
      </div>

      {/* Messages */}
      <MessageList onCategoryClick={handleCategoryClick} />

      {/* Input */}
      <ChatInputBar
        ref={inputRef}
        value={input}
        onChange={setInput}
        onSubmit={handleSubmit}
        disabled={isStreaming}
        deepAnalysis={deepAnalysis}
        onToggleDeep={() => setDeepAnalysis(d => !d)}
        vehicleVariant={vehicleVariant}
        onVehicleChange={setVehicleVariant}
        category={category}
        onCategoryChange={setCategory}
      />
    </div>
  )
}
