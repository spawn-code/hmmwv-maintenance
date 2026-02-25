import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Loader } from 'lucide-react'
import { useSettingsStore } from '../../store/useSettingsStore'
import type { SettingsModel } from '../../types'

const PROVIDERS = ['Ollama (Local)', 'OpenAI Compatible', 'Anthropic']

export default function AIProviderTab() {
  const { settings, isSaving, saveSettings, fetchOllamaModels, ollamaModels, ollamaConnected } =
    useSettingsStore()

  const [local, setLocal] = useState<Partial<SettingsModel>>({})
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'ok' | 'fail'>('idle')

  useEffect(() => {
    if (settings) setLocal(settings)
  }, [settings])

  if (!settings) return <div className="p-6 text-muted text-sm">Loading…</div>

  const merged = { ...settings, ...local }
  const provider = merged.provider ?? 'Ollama (Local)'

  const set = (patch: Partial<SettingsModel>) => setLocal(prev => ({ ...prev, ...patch }))

  const handleSave = async () => {
    await saveSettings(local)
    setLocal({})
  }

  const handleTestOllama = async () => {
    setTestStatus('testing')
    try {
      await fetchOllamaModels(merged.ollama_url ?? 'http://localhost:11434')
      setTestStatus(ollamaConnected ? 'ok' : 'fail')
    } catch {
      setTestStatus('fail')
    }
  }

  const inputClass = "w-full px-3 py-2 rounded text-sm outline-none"
  const inputStyle = {
    backgroundColor: 'var(--color-bg-base)',
    border: '1px solid var(--color-border)',
    color: 'var(--color-text)',
  }
  const labelStyle = { color: 'var(--color-muted)', fontSize: '0.75rem', marginBottom: '0.25rem', display: 'block' as const }

  return (
    <div className="space-y-6">
      {/* Provider selector */}
      <div>
        <label style={labelStyle}>AI PROVIDER</label>
        <div className="flex gap-2">
          {PROVIDERS.map(p => (
            <button
              key={p}
              onClick={() => set({ provider: p as SettingsModel['provider'] })}
              className="px-3 py-1.5 rounded text-xs font-mono cursor-pointer transition-colors"
              style={{
                backgroundColor: merged.provider === p ? 'var(--color-amber)' : 'var(--color-bg-surface2)',
                color:           merged.provider === p ? '#000' : 'var(--color-muted)',
                border: `1px solid ${merged.provider === p ? 'var(--color-amber)' : 'var(--color-border)'}`,
                fontWeight: merged.provider === p ? 700 : 400,
              }}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Ollama fields */}
      {provider === 'Ollama (Local)' && (
        <>
          <div>
            <label style={labelStyle}>OLLAMA URL</label>
            <div className="flex gap-2">
              <input
                className={inputClass}
                style={inputStyle}
                value={merged.ollama_url ?? ''}
                onChange={e => set({ ollama_url: e.target.value })}
                placeholder="http://localhost:11434"
              />
              <button
                onClick={handleTestOllama}
                disabled={testStatus === 'testing'}
                className="px-3 py-2 rounded text-xs font-mono cursor-pointer shrink-0"
                style={{
                  backgroundColor: 'var(--color-bg-surface2)',
                  border: '1px solid var(--color-border)',
                  color: 'var(--color-text)',
                }}
              >
                {testStatus === 'testing' ? <Loader size={12} className="animate-spin" /> : 'Test'}
              </button>
              {testStatus === 'ok'   && <CheckCircle size={16} style={{ color: 'var(--color-olive)',  alignSelf: 'center' }} />}
              {testStatus === 'fail' && <XCircle     size={16} style={{ color: 'var(--color-danger)', alignSelf: 'center' }} />}
            </div>
          </div>

          <div>
            <label style={labelStyle}>MODEL</label>
            {ollamaModels.length > 0 ? (
              <select
                className={inputClass}
                style={inputStyle}
                value={merged.ollama_model ?? ''}
                onChange={e => set({ ollama_model: e.target.value })}
              >
                {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            ) : (
              <input
                className={inputClass}
                style={inputStyle}
                value={merged.ollama_model ?? ''}
                onChange={e => set({ ollama_model: e.target.value })}
                placeholder="e.g. llama3.2:latest"
              />
            )}
          </div>
        </>
      )}

      {/* OpenAI-compatible fields */}
      {provider === 'OpenAI Compatible' && (
        <>
          <div>
            <label style={labelStyle}>API BASE URL</label>
            <input
              className={inputClass}
              style={inputStyle}
              value={merged.openai_url ?? ''}
              onChange={e => set({ openai_url: e.target.value })}
              placeholder="https://api.openai.com/v1"
            />
          </div>
          <div>
            <label style={labelStyle}>API KEY</label>
            <input
              className={inputClass}
              style={inputStyle}
              type="password"
              value={merged.openai_api_key ?? ''}
              onChange={e => set({ openai_api_key: e.target.value })}
              placeholder="sk-…"
            />
          </div>
          <div>
            <label style={labelStyle}>MODEL</label>
            <input
              className={inputClass}
              style={inputStyle}
              value={merged.openai_model ?? ''}
              onChange={e => set({ openai_model: e.target.value })}
              placeholder="gpt-4o"
            />
          </div>
        </>
      )}

      {/* Anthropic fields */}
      {provider === 'Anthropic' && (
        <>
          <div>
            <label style={labelStyle}>API KEY</label>
            <input
              className={inputClass}
              style={inputStyle}
              type="password"
              value={merged.anthropic_api_key ?? ''}
              onChange={e => set({ anthropic_api_key: e.target.value })}
              placeholder="sk-ant-…"
            />
          </div>
          <div>
            <label style={labelStyle}>MODEL</label>
            <input
              className={inputClass}
              style={inputStyle}
              value={merged.anthropic_model ?? ''}
              onChange={e => set({ anthropic_model: e.target.value })}
              placeholder="claude-opus-4-6"
            />
          </div>
        </>
      )}

      {/* YouTube fields */}
      <div
        className="rounded p-4"
        style={{ backgroundColor: 'var(--color-bg-surface2)', border: '1px solid var(--color-border)' }}
      >
        <div className="flex items-center justify-between mb-3">
          <label className="text-xs font-mono" style={{ color: 'var(--color-muted)' }}>YOUTUBE SEARCH</label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={merged.youtube_enabled ?? false}
              onChange={e => set({ youtube_enabled: e.target.checked })}
            />
            <span className="text-xs" style={{ color: 'var(--color-text)' }}>Enabled</span>
          </label>
        </div>
        {merged.youtube_enabled && (
          <div>
            <label style={labelStyle}>YOUTUBE API KEY</label>
            <input
              className={inputClass}
              style={inputStyle}
              type="password"
              value={merged.youtube_api_key ?? ''}
              onChange={e => set({ youtube_api_key: e.target.value })}
              placeholder="AIza…"
            />
          </div>
        )}
      </div>

      {/* Save */}
      <button
        onClick={handleSave}
        disabled={isSaving || Object.keys(local).length === 0}
        className="px-4 py-2 rounded text-sm font-mono cursor-pointer transition-opacity disabled:opacity-40"
        style={{ backgroundColor: 'var(--color-olive)', color: '#000', fontWeight: 700 }}
      >
        {isSaving ? 'Saving…' : 'Save Settings'}
      </button>
    </div>
  )
}
