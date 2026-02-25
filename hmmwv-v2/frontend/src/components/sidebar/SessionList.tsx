import { useEffect } from 'react'
import { Plus } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useSessionStore } from '../../store/useSessionStore'
import { groupSessionsByDate } from '../../utils/dateGroups'
import SessionItem from './SessionItem'

export default function SessionList() {
  const {
    sessions, activeSessionId,
    loadSessions, createSession, selectSession,
  } = useSessionStore()
  const navigate = useNavigate()

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  const handleNew = async () => {
    const s = await createSession()
    if (s) {
      selectSession(s.id)
      navigate('/')
    }
  }

  const handleSelect = (id: string) => {
    selectSession(id)
    navigate('/')
  }

  const groups = groupSessionsByDate(sessions)

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header row */}
      <div
        className="flex items-center justify-between px-4 py-2"
        style={{ borderBottom: '1px solid var(--color-border)' }}
      >
        <span
          className="text-xs font-mono tracking-widest"
          style={{ color: 'var(--color-muted)' }}
        >
          SESSIONS
        </span>
        <button
          onClick={handleNew}
          className="flex items-center justify-center w-5 h-5 rounded transition-colors cursor-pointer"
          style={{ color: 'var(--color-amber)' }}
          title="New session"
        >
          <Plus size={14} strokeWidth={2.5} />
        </button>
      </div>

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto py-1 scrollbar-hidden">
        {groups.length === 0 && (
          <div className="px-4 py-6 text-center text-xs" style={{ color: 'var(--color-muted)' }}>
            No sessions yet.
            <br />Click + to start.
          </div>
        )}
        {groups.map(group => (
          <div key={group.label} className="mb-1">
            <div
              className="px-4 py-1 text-xs font-mono tracking-wider"
              style={{ color: 'var(--color-muted)' }}
            >
              {group.label.toUpperCase()}
            </div>
            {group.sessions.map(s => (
              <SessionItem
                key={s.id}
                session={s}
                isActive={s.id === activeSessionId}
                onClick={() => handleSelect(s.id)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
