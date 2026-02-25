import apiFetch from './client'
import type { Session, SessionSummary } from '../types'

export const sessionsApi = {
  list: () =>
    apiFetch<SessionSummary[]>('/sessions'),

  create: (title = 'New Chat', vehicle_variant = '', maintenance_category = '') =>
    apiFetch<Session>('/sessions', {
      method: 'POST',
      body: JSON.stringify({ title, vehicle_variant, maintenance_category }),
    }),

  get: (id: string) =>
    apiFetch<Session>(`/sessions/${id}`),

  update: (id: string, patch: { title?: string; vehicle_variant?: string; maintenance_category?: string }) =>
    apiFetch<Session>(`/sessions/${id}`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),

  delete: (id: string) =>
    apiFetch<{ ok: boolean }>(`/sessions/${id}`, { method: 'DELETE' }),

  clearAll: () =>
    apiFetch<{ deleted: number }>('/sessions', { method: 'DELETE' }),
}
