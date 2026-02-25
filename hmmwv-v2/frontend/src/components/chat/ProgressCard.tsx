import { Check } from 'lucide-react'
import { useChatStore } from '../../store/useChatStore'

const STEP_META: Record<string, { label: string; emoji: string; parallel?: boolean }> = {
  retriever:  { label: 'Searching knowledge base',  emoji: '🔍' },
  procedure:  { label: 'Writing procedure',          emoji: '📋', parallel: true },
  safety:     { label: 'Safety analysis',            emoji: '⚠️', parallel: true },
  parts:      { label: 'Parts identification',       emoji: '🔧', parallel: true },
  simplifier: { label: 'Simplifying language',       emoji: '✏️' },
  editor:     { label: 'Final synthesis',            emoji: '✅' },
}

const STEP_ORDER = ['retriever', 'procedure', 'safety', 'parts', 'simplifier', 'editor']

export default function ProgressCard() {
  const { agentSteps } = useChatStore()

  if (agentSteps.length === 0) return null

  // Show parallel group label
  const parallelSteps = ['procedure', 'safety', 'parts']
  let shownParallelLabel = false

  return (
    <div
      className="rounded p-4 mb-3"
      style={{
        backgroundColor: 'var(--color-bg-surface)',
        border: '1px solid var(--color-border)',
      }}
    >
      <div
        className="font-mono text-xs mb-3 tracking-widest"
        style={{ color: 'var(--color-muted)' }}
      >
        DEEP ANALYSIS IN PROGRESS
      </div>

      <div className="space-y-2">
        {STEP_ORDER.map(stepId => {
          const meta  = STEP_META[stepId]
          const step  = agentSteps.find(s => s.step === stepId)
          if (!step && !meta) return null

          const isParallel  = parallelSteps.includes(stepId)
          const showLabel   = isParallel && !shownParallelLabel
          if (showLabel) shownParallelLabel = true

          const isDone      = step?.done ?? false
          const isRunning   = step && !isDone
          const elapsed     = step?.elapsed

          return (
            <div key={stepId}>
              {showLabel && (
                <div
                  className="text-xs font-mono mb-1 pl-6"
                  style={{ color: 'var(--color-muted)' }}
                >
                  — parallel —
                </div>
              )}
              <div className="flex items-center gap-3">
                {/* Status icon */}
                <div className="w-4 shrink-0 flex items-center justify-center">
                  {isDone ? (
                    <Check size={12} style={{ color: 'var(--color-olive)' }} />
                  ) : isRunning ? (
                    <span
                      className="inline-block w-2 h-2 rounded-full animate-pulse-dot"
                      style={{ backgroundColor: 'var(--color-amber)' }}
                    />
                  ) : (
                    <span
                      className="inline-block w-2 h-2 rounded-full"
                      style={{ backgroundColor: 'var(--color-border)' }}
                    />
                  )}
                </div>

                {/* Label */}
                <span
                  className="text-xs flex-1"
                  style={{
                    color: isDone    ? 'var(--color-text)' :
                           isRunning ? 'var(--color-bright)' :
                           'var(--color-muted)',
                  }}
                >
                  {meta.emoji} {meta.label}
                </span>

                {/* Elapsed */}
                {isDone && elapsed != null && (
                  <span className="text-xs" style={{ color: 'var(--color-muted)' }}>
                    {elapsed.toFixed(1)}s
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
