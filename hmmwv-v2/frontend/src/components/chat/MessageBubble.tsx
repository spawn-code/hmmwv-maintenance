import { Wrench } from 'lucide-react'
import MarkdownRenderer from './MarkdownRenderer'
import SourcesPanel from './SourcesPanel'
import ImageGallery from './ImageGallery'
import StreamingDots from './StreamingDots'
import ProgressCard from './ProgressCard'
import type { SessionMessage } from '../../types'

interface Props {
  message:    SessionMessage
  isStreaming?: boolean
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex justify-end animate-fade-in">
        <div
          className="max-w-[75%] rounded px-4 py-3 text-sm"
          style={{
            backgroundColor: 'var(--color-bg-surface2)',
            borderLeft: '2px solid var(--color-amber)',
            color: 'var(--color-bright)',
          }}
        >
          {message.content}
        </div>
      </div>
    )
  }

  // Assistant bubble
  const showDots      = isStreaming && !message.content
  const showProgress  = isStreaming

  return (
    <div className="flex gap-3 animate-fade-in">
      {/* Avatar */}
      <div
        className="flex items-center justify-center w-7 h-7 rounded shrink-0 mt-0.5"
        style={{
          backgroundColor: 'var(--color-bg-surface2)',
          border: '1px solid var(--color-border)',
        }}
      >
        <Wrench size={14} style={{ color: 'var(--color-olive)' }} />
      </div>

      {/* Content card */}
      <div className="flex-1 min-w-0">
        {showProgress && <ProgressCard />}

        <div
          className="rounded px-4 py-3"
          style={{
            backgroundColor: 'var(--color-bg-surface)',
            border: '1px solid var(--color-border)',
          }}
        >
          {showDots ? (
            <StreamingDots />
          ) : (
            <MarkdownRenderer content={message.content} />
          )}

          {/* Blinking cursor while streaming */}
          {isStreaming && message.content && (
            <span className="animate-blink inline-block ml-0.5" style={{ color: 'var(--color-amber)' }}>▊</span>
          )}
        </div>

        {/* Sources & images (only shown once streaming is complete) */}
        {!isStreaming && (
          <>
            <SourcesPanel sources={message.sources ?? []} />
            <ImageGallery  images={message.images   ?? []} />
          </>
        )}
      </div>
    </div>
  )
}
