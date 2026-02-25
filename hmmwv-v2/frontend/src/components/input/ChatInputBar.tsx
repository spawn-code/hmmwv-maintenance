import { useRef, useEffect, forwardRef, useImperativeHandle } from 'react'
import type { KeyboardEvent } from 'react'
import { ArrowRight, Microscope } from 'lucide-react'
import VehicleSelector  from './VehicleSelector'
import CategorySelector from './CategorySelector'

interface Props {
  value:            string
  onChange:         (v: string) => void
  onSubmit:         () => void
  disabled?:        boolean
  deepAnalysis:     boolean
  onToggleDeep:     () => void
  vehicleVariant:   string
  onVehicleChange:  (v: string) => void
  category:         string
  onCategoryChange: (v: string) => void
}

export interface ChatInputBarHandle {
  focus: () => void
  setValue: (v: string) => void
}

const ChatInputBar = forwardRef<ChatInputBarHandle, Props>(function ChatInputBar(
  {
    value, onChange, onSubmit, disabled,
    deepAnalysis, onToggleDeep,
    vehicleVariant, onVehicleChange,
    category, onCategoryChange,
  },
  ref,
) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px'
  }, [value])

  useImperativeHandle(ref, () => ({
    focus:    () => textareaRef.current?.focus(),
    setValue: (v) => onChange(v),
  }))

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit()
    }
  }

  return (
    <div
      className="border-t px-4 py-3"
      style={{ borderColor: 'var(--color-border)', backgroundColor: 'var(--color-bg-base)' }}
    >
      {/* Selector row */}
      <div className="flex items-center gap-2 mb-2">
        <VehicleSelector  value={vehicleVariant} onChange={onVehicleChange}  />
        <CategorySelector value={category}       onChange={onCategoryChange} />

        {/* Deep Analysis toggle */}
        <button
          onClick={onToggleDeep}
          className="flex items-center gap-1.5 text-xs px-2 py-1 rounded transition-colors cursor-pointer"
          style={{
            backgroundColor: deepAnalysis ? 'var(--color-amber)' : 'var(--color-bg-surface2)',
            color:           deepAnalysis ? '#000' : 'var(--color-muted)',
            border: `1px solid ${deepAnalysis ? 'var(--color-amber)' : 'var(--color-border)'}`,
            fontWeight: deepAnalysis ? 600 : 400,
          }}
          title="Enable multi-agent deep analysis (slower but more thorough)"
        >
          <Microscope size={11} />
          🔍 Deep Analysis
        </button>
      </div>

      {/* Input row */}
      <div
        className="flex items-end gap-2 rounded"
        style={{
          backgroundColor: 'var(--color-bg-surface)',
          border: '1px solid var(--color-border)',
          padding: '0.5rem',
        }}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={handleKey}
          disabled={disabled}
          placeholder="Describe the issue with your HMMWV..."
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm outline-none"
          style={{
            color: 'var(--color-text)',
            minHeight: '2rem',
            maxHeight: '160px',
            lineHeight: '1.5',
          }}
        />

        <button
          onClick={onSubmit}
          disabled={disabled || !value.trim()}
          className="flex items-center justify-center w-8 h-8 rounded shrink-0 transition-opacity cursor-pointer disabled:opacity-40"
          style={{
            backgroundColor: 'var(--color-olive)',
            color: '#000',
          }}
          title="Send (Enter)"
        >
          <ArrowRight size={15} strokeWidth={2.5} />
        </button>
      </div>

      <p className="text-xs mt-1 text-center" style={{ color: 'var(--color-muted)' }}>
        Press <kbd className="font-mono">Enter</kbd> to send · <kbd className="font-mono">Shift+Enter</kbd> for new line
      </p>
    </div>
  )
})

export default ChatInputBar
