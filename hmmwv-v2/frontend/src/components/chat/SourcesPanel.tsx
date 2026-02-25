import { useState } from 'react'
import { ChevronDown, ChevronRight, FileText } from 'lucide-react'
import type { SourceRef } from '../../types'

interface Props {
  sources: SourceRef[]
}

export default function SourcesPanel({ sources }: Props) {
  const [open, setOpen] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-2">
      {/* Toggle button */}
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 text-xs cursor-pointer hover:opacity-80 transition-opacity"
        style={{ color: 'var(--color-amber)', background: 'none', border: 'none', padding: 0 }}
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <FileText size={12} />
        <span>{sources.length} source{sources.length !== 1 ? 's' : ''}</span>
      </button>

      {/* Expanded list */}
      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((src, i) => (
            <div
              key={i}
              className="rounded p-3 text-xs"
              style={{
                backgroundColor: 'var(--color-bg-base)',
                borderLeft: '2px solid var(--color-amber)',
                paddingLeft: '0.75rem',
              }}
            >
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="font-mono px-1 py-0.5 rounded text-xs"
                  style={{
                    backgroundColor: 'var(--color-bg-surface2)',
                    color: 'var(--color-amber)',
                  }}
                >
                  {src.metadata?.source_file ?? 'TM'}
                </span>
                {src.metadata?.page_number && (
                  <span style={{ color: 'var(--color-muted)' }}>
                    p.{src.metadata.page_number}
                  </span>
                )}
                {src.metadata?.section_title && (
                  <span
                    className="truncate"
                    style={{ color: 'var(--color-muted)' }}
                  >
                    {src.metadata.section_title}
                  </span>
                )}
              </div>
              <p
                className="line-clamp-3"
                style={{ color: 'var(--color-muted)' }}
              >
                {src.text?.slice(0, 200)}
                {(src.text?.length ?? 0) > 200 ? '…' : ''}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
