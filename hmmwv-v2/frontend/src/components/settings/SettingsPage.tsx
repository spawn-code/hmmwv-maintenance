import { useState, useEffect } from 'react'
import { Cpu, Database, Sliders, ArrowLeft } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useSettingsStore } from '../../store/useSettingsStore'
import AIProviderTab    from './AIProviderTab'
import KnowledgeBaseTab from './KnowledgeBaseTab'
import AdvancedTab      from './AdvancedTab'

type Tab = 'ai' | 'knowledge' | 'advanced'

const TABS: { id: Tab; label: string; Icon: typeof Cpu }[] = [
  { id: 'ai',        label: 'AI Provider',    Icon: Cpu      },
  { id: 'knowledge', label: 'Knowledge Base', Icon: Database },
  { id: 'advanced',  label: 'Advanced',       Icon: Sliders  },
]

export default function SettingsPage() {
  const { loadSettings } = useSettingsStore()
  const [activeTab, setActiveTab] = useState<Tab>('ai')
  const navigate = useNavigate()

  useEffect(() => { loadSettings() }, [loadSettings])

  return (
    <div className="flex flex-col h-full">
      {/* Topbar */}
      <div
        className="flex items-center gap-4 px-6 py-3 border-b shrink-0"
        style={{ borderColor: 'var(--color-border)' }}
      >
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-1.5 text-xs cursor-pointer hover:opacity-80"
          style={{ color: 'var(--color-muted)', background: 'none', border: 'none', padding: 0 }}
        >
          <ArrowLeft size={13} />
          Back
        </button>
        <span
          className="font-mono text-sm tracking-widest uppercase"
          style={{ color: 'var(--color-muted)' }}
        >
          SETTINGS
        </span>
      </div>

      {/* Tab bar */}
      <div
        className="flex border-b shrink-0"
        style={{ borderColor: 'var(--color-border)' }}
      >
        {TABS.map(({ id, label, Icon }) => {
          const active = activeTab === id
          return (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className="flex items-center gap-2 px-5 py-3 text-xs font-mono cursor-pointer transition-colors"
              style={{
                color:        active ? 'var(--color-amber)' : 'var(--color-muted)',
                borderBottom: active ? '2px solid var(--color-amber)' : '2px solid transparent',
                borderTop:    'none',
                borderLeft:   'none',
                borderRight:  'none',
                marginBottom: '-1px',
                background:   'none',
                outline:      'none',
              }}
            >
              <Icon size={13} />
              {label.toUpperCase()}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto px-6 py-6 max-w-2xl">
        {activeTab === 'ai'        && <AIProviderTab />}
        {activeTab === 'knowledge' && <KnowledgeBaseTab />}
        {activeTab === 'advanced'  && <AdvancedTab />}
      </div>
    </div>
  )
}
