import { create } from 'zustand'
import type { SettingsModel } from '../types'
import { settingsApi } from '../api/settings'

interface SettingsState {
  settings:     SettingsModel | null
  isLoading:    boolean
  isSaving:     boolean
  ollamaModels: string[]
  ollamaConnected: boolean

  loadSettings:      () => Promise<void>
  saveSettings:      (patch: Partial<SettingsModel>) => Promise<void>
  fetchOllamaModels: (url: string) => Promise<void>
}

const DEFAULT_SETTINGS: SettingsModel = {
  provider:             'Ollama (Local)',
  ollama_url:           'http://localhost:11434',
  ollama_model:         'gpt-oss:latest',
  openai_url:           'https://api.openai.com/v1',
  openai_model:         'gpt-4o',
  openai_api_key:       '',
  anthropic_api_key:    '',
  anthropic_model:      'claude-opus-4-6',
  youtube_api_key:      '',
  youtube_enabled:      true,
  youtube_max_results:  3,
  agent_mode:           false,
  agent1_enabled:       false,
  agent1_provider:      'Ollama (Local)',
  agent1_model:         'gpt-oss:latest',
  agent2_provider:      'Ollama (Local)',
  agent2_model:         'gpt-oss:latest',
  agent3_provider:      'Ollama (Local)',
  agent3_model:         'gpt-oss:latest',
  agent4_provider:      'Ollama (Local)',
  agent4_model:         'gpt-oss:latest',
  agent5_enabled:       true,
  agent5_provider:      'Ollama (Local)',
  agent5_model:         'gpt-oss:latest',
  agent6_provider:      'Ollama (Local)',
  agent6_model:         'gpt-oss:latest',
}

export const useSettingsStore = create<SettingsState>((set) => ({
  settings:        DEFAULT_SETTINGS,
  isLoading:       false,
  isSaving:        false,
  ollamaModels:    [],
  ollamaConnected: false,

  loadSettings: async () => {
    set({ isLoading: true })
    try {
      const settings = await settingsApi.get()
      set({ settings: { ...DEFAULT_SETTINGS, ...settings }, isLoading: false })
    } catch {
      set({ isLoading: false })
    }
  },

  saveSettings: async (patch) => {
    set({ isSaving: true })
    try {
      const updated = await settingsApi.update(patch)
      set({ settings: { ...DEFAULT_SETTINGS, ...updated }, isSaving: false })
    } catch {
      set({ isSaving: false })
    }
  },

  fetchOllamaModels: async (url) => {
    try {
      const resp = await settingsApi.getOllamaModels(url)
      set({ ollamaModels: resp.models, ollamaConnected: resp.connected })
    } catch {
      set({ ollamaModels: [], ollamaConnected: false })
    }
  },
}))
