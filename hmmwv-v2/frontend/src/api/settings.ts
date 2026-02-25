import apiFetch from './client'
import type { SettingsModel, KnowledgeStats, OllamaModelsResponse } from '../types'

export const settingsApi = {
  get: () =>
    apiFetch<SettingsModel>('/settings'),

  update: (patch: Partial<SettingsModel>) =>
    apiFetch<SettingsModel>('/settings', {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),

  getOllamaModels: (url: string) =>
    apiFetch<OllamaModelsResponse>(`/settings/ollama/models?url=${encodeURIComponent(url)}`),

  getVariants: () =>
    apiFetch<{ variants: string[] }>('/settings/variants'),

  getCategories: () =>
    apiFetch<{ categories: string[] }>('/settings/categories'),
}

export const knowledgeApi = {
  stats: () =>
    apiFetch<KnowledgeStats>('/knowledge/stats'),

  index: () =>
    apiFetch<{ indexed: number; chunks_added: number }>('/knowledge/index', { method: 'POST' }),

  renderPages: (dpi = 150) =>
    apiFetch<{ message: string; total_pdfs: number }>(
      `/knowledge/render-pages?dpi=${dpi}`,
      { method: 'POST' },
    ),

  clear: () =>
    apiFetch<{ ok: boolean }>('/knowledge/index', { method: 'DELETE' }),
}
