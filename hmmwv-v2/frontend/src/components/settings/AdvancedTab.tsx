import { useState, useEffect } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useSettingsStore } from '../../store/useSettingsStore'
import type { SettingsModel } from '../../types'

const PROVIDERS  = ['Ollama (Local)', 'OpenAI Compatible', 'Anthropic']
const AGENT_NAMES: Record<string, string> = {
  agent1: 'Agent 1 – Retriever',
  agent2: 'Agent 2 – Procedure Writer',
  agent3: 'Agent 3 – Safety Officer',
  agent4: 'Agent 4 – Parts Specialist',
  agent5: 'Agent 5 – Simplifier',
  agent6: 'Agent 6 – Editor/Synthesizer',
}

export default function AdvancedTab() {
  const { settings, isSaving, saveSettings } = useSettingsStore()
  const [local,    setLocal]    = useState<Partial<SettingsModel>>({})
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    if (settings) setLocal(settings)
  }, [settings])

  if (!settings) return null
  const merged = { ...settings, ...local }

  const set = (patch: Partial<SettingsModel>) => setLocal(prev => ({ ...prev, ...patch }))

  const handleSave = async () => {
    await saveSettings(local)
    setLocal({})
  }

  const inputStyle = {
    backgroundColor: 'var(--color-bg-base)',
    border: '1px solid var(--color-border)',
    color: 'var(--color-text)',
    padding: '0.4rem 0.6rem',
    borderRadius: '4px',
    fontSize: '0.75rem',
    outline: 'none',
    width: '100%',
  } as const
  const labelStyle = {
    color: 'var(--color-muted)',
    fontSize: '0.7rem',
    display: 'block' as const,
    marginBottom: '0.2rem',
    fontFamily: 'var(--font-mono)',
  }

  return (
    <div className="space-y-6">
      {/* Default agent mode */}
      <div>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={merged.agent_mode ?? false}
            onChange={e => set({ agent_mode: e.target.checked })}
          />
          <div>
            <div className="text-sm" style={{ color: 'var(--color-bright)' }}>Enable Deep Analysis by default</div>
            <div className="text-xs" style={{ color: 'var(--color-muted)' }}>
              When checked, the multi-agent pipeline runs for every query (slower but more thorough).
              You can still toggle this per-query in the chat input.
            </div>
          </div>
        </label>
      </div>

      {/* Per-agent config */}
      <div>
        <p className="text-xs font-mono mb-3" style={{ color: 'var(--color-muted)' }}>
          AGENT CONFIGURATION (DEEP ANALYSIS MODE)
        </p>
        <div className="space-y-2">
          {Object.entries(AGENT_NAMES).map(([agentKey, agentName]) => {
            const provKey  = `${agentKey}_provider` as keyof SettingsModel
            const modelKey = `${agentKey}_model`    as keyof SettingsModel
            const isOpen   = expanded === agentKey

            return (
              <div
                key={agentKey}
                className="rounded"
                style={{ border: '1px solid var(--color-border)' }}
              >
                <button
                  onClick={() => setExpanded(isOpen ? null : agentKey)}
                  className="w-full flex items-center gap-2 px-3 py-2.5 cursor-pointer"
                  style={{
                    backgroundColor: isOpen ? 'var(--color-bg-surface2)' : 'var(--color-bg-surface)',
                    borderRadius: isOpen ? '4px 4px 0 0' : '4px',
                    background: 'none',
                  }}
                >
                  {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  <span className="text-xs font-mono" style={{ color: 'var(--color-text)' }}>
                    {agentName}
                  </span>
                  <span className="ml-auto text-xs" style={{ color: 'var(--color-muted)' }}>
                    {(merged[provKey] as string) ?? 'Ollama (Local)'} — {(merged[modelKey] as string) ?? ''}
                  </span>
                </button>

                {isOpen && (
                  <div
                    className="px-4 py-3 space-y-3"
                    style={{ borderTop: '1px solid var(--color-border)' }}
                  >
                    <div>
                      <label style={labelStyle}>PROVIDER</label>
                      <select
                        style={inputStyle}
                        value={(merged[provKey] as string) ?? 'Ollama (Local)'}
                        onChange={e => set({ [provKey]: e.target.value } as Partial<SettingsModel>)}
                      >
                        {PROVIDERS.map(p => <option key={p} value={p}>{p}</option>)}
                      </select>
                    </div>
                    <div>
                      <label style={labelStyle}>MODEL</label>
                      <input
                        style={inputStyle}
                        value={(merged[modelKey] as string) ?? ''}
                        onChange={e => set({ [modelKey]: e.target.value } as Partial<SettingsModel>)}
                        placeholder="model name"
                      />
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <button
        onClick={handleSave}
        disabled={isSaving || Object.keys(local).length === 0}
        className="px-4 py-2 rounded text-sm font-mono cursor-pointer disabled:opacity-40"
        style={{ backgroundColor: 'var(--color-olive)', color: '#000', fontWeight: 700 }}
      >
        {isSaving ? 'Saving…' : 'Save Settings'}
      </button>
    </div>
  )
}
