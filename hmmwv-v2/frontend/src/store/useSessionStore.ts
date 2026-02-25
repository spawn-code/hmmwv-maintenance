import { create } from 'zustand'
import type { Session, SessionSummary, SessionMessage, SourceRef, ImageRef } from '../types'
import { sessionsApi } from '../api/sessions'

interface SessionState {
  sessions: SessionSummary[]
  activeSessionId: string | null
  activeSession:   Session | null
  isLoading:       boolean

  loadSessions:   () => Promise<void>
  createSession:  (title?: string, variant?: string, category?: string) => Promise<Session>
  selectSession:  (id: string) => Promise<void>
  updateSession:  (id: string, patch: Partial<Pick<Session, 'title' | 'vehicle_variant' | 'maintenance_category'>>) => Promise<void>
  deleteSession:  (id: string) => Promise<void>

  // Optimistic local updates (called by chat router after stream completes)
  appendMessages: (
    sessionId: string,
    userContent: string,
    assistantContent: string,
    sources: SourceRef[],
    images: ImageRef[],
    timestamp: string,
  ) => void

  // Update title of a session summary in the list (after auto-title from server)
  refreshSessionSummary: (id: string) => Promise<void>
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessions:        [],
  activeSessionId: null,
  activeSession:   null,
  isLoading:       false,

  loadSessions: async () => {
    set({ isLoading: true })
    try {
      const sessions = await sessionsApi.list()
      set({ sessions, isLoading: false })
    } catch {
      set({ isLoading: false })
    }
  },

  createSession: async (title = 'New Chat', variant = '', category = '') => {
    const session = await sessionsApi.create(title, variant, category)
    // Add to top of list as a summary
    const summary: SessionSummary = {
      id:                   session.id,
      title:                session.title,
      vehicle_variant:      session.vehicle_variant,
      maintenance_category: session.maintenance_category,
      created_at:           session.created_at,
      updated_at:           session.updated_at,
      message_count:        0,
    }
    set(state => ({
      sessions:        [summary, ...state.sessions],
      activeSessionId: session.id,
      activeSession:   session,
    }))
    return session
  },

  selectSession: async (id: string) => {
    if (get().activeSessionId === id) return
    set({ isLoading: true })
    try {
      const session = await sessionsApi.get(id)
      set({ activeSession: session, activeSessionId: id, isLoading: false })
    } catch {
      set({ isLoading: false })
    }
  },

  updateSession: async (id, patch) => {
    const updated = await sessionsApi.update(id, patch)
    set(state => ({
      sessions: state.sessions.map(s =>
        s.id === id ? { ...s, ...patch, updated_at: updated.updated_at } : s
      ),
      activeSession: state.activeSession?.id === id
        ? { ...state.activeSession, ...patch }
        : state.activeSession,
    }))
  },

  deleteSession: async (id) => {
    await sessionsApi.delete(id)
    set(state => {
      const sessions = state.sessions.filter(s => s.id !== id)
      const wasActive = state.activeSessionId === id
      return {
        sessions,
        activeSessionId: wasActive ? null : state.activeSessionId,
        activeSession:   wasActive ? null : state.activeSession,
      }
    })
  },

  appendMessages: (sessionId, userContent, assistantContent, sources, images, timestamp) => {
    const userMsg: SessionMessage = { role: 'user', content: userContent, timestamp, sources: [], images: [] }
    const assistantMsg: SessionMessage = { role: 'assistant', content: assistantContent, timestamp, sources, images }

    set(state => {
      // Update activeSession messages
      const activeSession = state.activeSession?.id === sessionId
        ? {
            ...state.activeSession,
            messages:   [...(state.activeSession?.messages ?? []), userMsg, assistantMsg],
            updated_at: timestamp,
          }
        : state.activeSession

      // Update summary message_count and updated_at
      const sessions = state.sessions.map(s =>
        s.id === sessionId
          ? { ...s, message_count: s.message_count + 2, updated_at: timestamp }
          : s
      )

      return { activeSession, sessions }
    })
  },

  refreshSessionSummary: async (id) => {
    try {
      const session = await sessionsApi.get(id)
      set(state => ({
        sessions: state.sessions.map(s =>
          s.id === id ? { ...s, title: session.title } : s
        ),
        activeSession: state.activeSession?.id === id
          ? { ...state.activeSession, title: session.title }
          : state.activeSession,
      }))
    } catch {
      // ignore
    }
  },
}))
