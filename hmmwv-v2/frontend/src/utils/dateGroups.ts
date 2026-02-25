// Groups sessions by Today / Yesterday / Older based on created_at timestamp

import type { SessionSummary, SessionGroup } from '../types'

function isToday(dateStr: string): boolean {
  const date = new Date(dateStr)
  const now  = new Date()
  return (
    date.getFullYear() === now.getFullYear() &&
    date.getMonth()    === now.getMonth()    &&
    date.getDate()     === now.getDate()
  )
}

function isYesterday(dateStr: string): boolean {
  const date = new Date(dateStr)
  const yesterday = new Date()
  yesterday.setDate(yesterday.getDate() - 1)
  return (
    date.getFullYear() === yesterday.getFullYear() &&
    date.getMonth()    === yesterday.getMonth()    &&
    date.getDate()     === yesterday.getDate()
  )
}

export function groupSessionsByDate(sessions: SessionSummary[]): SessionGroup[] {
  const today:     SessionSummary[] = []
  const yesterday: SessionSummary[] = []
  const older:     SessionSummary[] = []

  for (const s of sessions) {
    if (isToday(s.updated_at))         today.push(s)
    else if (isYesterday(s.updated_at)) yesterday.push(s)
    else                               older.push(s)
  }

  const groups: SessionGroup[] = []
  if (today.length)     groups.push({ label: 'Today',     sessions: today     })
  if (yesterday.length) groups.push({ label: 'Yesterday', sessions: yesterday })
  if (older.length)     groups.push({ label: 'Older',     sessions: older     })
  return groups
}

export function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr)
  const now  = new Date()
  const diffMs  = now.getTime() - date.getTime()
  const diffMin = Math.floor(diffMs / 60_000)
  if (diffMin < 1)  return 'just now'
  if (diffMin < 60) return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr}h ago`
  const diffDay = Math.floor(diffHr / 24)
  if (diffDay < 7)  return `${diffDay}d ago`
  return date.toLocaleDateString()
}
