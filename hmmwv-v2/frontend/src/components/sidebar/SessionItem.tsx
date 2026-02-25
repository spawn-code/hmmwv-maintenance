import { useState, useRef, useEffect } from 'react'
import { Pencil, Trash2, Check, X } from 'lucide-react'
import { useSessionStore } from '../../store/useSessionStore'
import { formatRelativeTime } from '../../utils/dateGroups'
import type { SessionSummary } from '../../types'

interface Props {
  session:  SessionSummary
  isActive: boolean
  onClick:  () => void
}

export default function SessionItem({ session, isActive, onClick }: Props) {
  const { updateSession, deleteSession } = useSessionStore()
  const [hovered,  setHovered]  = useState(false)
  const [editing,  setEditing]  = useState(false)
  const [newTitle, setNewTitle] = useState(session.title)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  const handleRename = async () => {
    const t = newTitle.trim()
    if (t && t !== session.title) {
      await updateSession(session.id, { title: t })
    }
    setEditing(false)
  }

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    await deleteSession(session.id)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter')  handleRename()
    if (e.key === 'Escape') setEditing(false)
  }

  return (
    <div
      className="relative flex items-center gap-2 px-3 py-2 rounded mx-1 cursor-pointer transition-colors group"
      style={{
        backgroundColor: isActive ? 'var(--color-bg-surface)' : hovered ? 'var(--color-bg-surface2)' : 'transparent',
        borderLeft: isActive ? '2px solid var(--color-amber)' : '2px solid transparent',
      }}
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {editing ? (
        <div className="flex-1 flex items-center gap-1" onClick={e => e.stopPropagation()}>
          <input
            ref={inputRef}
            value={newTitle}
            onChange={e => setNewTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent text-xs outline-none"
            style={{ color: 'var(--color-bright)', borderBottom: '1px solid var(--color-amber)' }}
          />
          <button onClick={handleRename}><Check size={12} style={{ color: 'var(--color-olive)' }} /></button>
          <button onClick={() => setEditing(false)}><X size={12} style={{ color: 'var(--color-muted)' }} /></button>
        </div>
      ) : (
        <>
          <div className="flex-1 min-w-0">
            <div
              className="text-xs truncate"
              style={{ color: isActive ? 'var(--color-bright)' : 'var(--color-text)' }}
            >
              {session.title || 'New session'}
            </div>
            <div className="text-xs" style={{ color: 'var(--color-muted)' }}>
              {formatRelativeTime(session.updated_at)}
            </div>
          </div>

          {(hovered || isActive) && (
            <div className="flex items-center gap-1 shrink-0">
              <button
                onClick={e => { e.stopPropagation(); setEditing(true) }}
                className="p-1 rounded hover:opacity-80"
                title="Rename"
              >
                <Pencil size={11} style={{ color: 'var(--color-muted)' }} />
              </button>
              <button
                onClick={handleDelete}
                className="p-1 rounded hover:opacity-80"
                title="Delete"
              >
                <Trash2 size={11} style={{ color: 'var(--color-danger)' }} />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
