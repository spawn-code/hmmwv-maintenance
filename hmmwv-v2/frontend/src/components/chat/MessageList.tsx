import { useEffect } from 'react'
import { useAutoScroll }   from '../../hooks/useAutoScroll'
import { useSessionStore } from '../../store/useSessionStore'
import { useChatStore }    from '../../store/useChatStore'
import MessageBubble from './MessageBubble'
import WelcomeCard   from './WelcomeCard'
import type { SessionMessage } from '../../types'

interface Props {
  onCategoryClick: (text: string) => void
}

export default function MessageList({ onCategoryClick }: Props) {
  const { activeSession }                          = useSessionStore()
  const { isStreaming, streamingContent, pendingSources, pendingImages } = useChatStore()
  const { ref, resetScroll } = useAutoScroll([streamingContent, isStreaming])

  // Reset scroll when session changes
  useEffect(() => {
    resetScroll()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSession?.id])

  const messages = activeSession?.messages ?? []
  const showWelcome = messages.length === 0 && !isStreaming

  // Build the streaming assistant message
  const streamingMsg: SessionMessage | null = isStreaming
    ? {
        id:        '__streaming__',
        role:      'assistant',
        content:   streamingContent,
        timestamp: new Date().toISOString(),
        sources:   pendingSources,
        images:    pendingImages,
      }
    : null

  return (
    <div
      ref={ref}
      className="flex-1 overflow-y-auto px-6 py-6 space-y-6"
    >
      {showWelcome && <WelcomeCard onCategoryClick={onCategoryClick} />}

      {messages.map((msg, idx) => (
        <MessageBubble
          key={msg.id ?? `msg-${idx}`}
          message={msg}
          isStreaming={false}
        />
      ))}

      {streamingMsg && (
        <MessageBubble
          message={streamingMsg}
          isStreaming={true}
        />
      )}
    </div>
  )
}
