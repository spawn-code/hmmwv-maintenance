import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Shield } from 'lucide-react'
import SidebarTabs from '../sidebar/SidebarTabs'
import SessionList from '../sidebar/SessionList'

type Tab = 'chat' | 'docs' | 'settings'

export default function Sidebar() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const navigate  = useNavigate()
  const location  = useLocation()

  const handleTab = (tab: Tab) => {
    setActiveTab(tab)
    if (tab === 'settings') navigate('/settings')
    else                    navigate('/')
  }

  // Sync active tab with route
  const effectiveTab: Tab =
    location.pathname === '/settings' ? 'settings' :
    activeTab === 'settings'          ? 'chat' :
    activeTab

  return (
    <aside
      className="flex flex-col w-64 shrink-0 border-r"
      style={{
        backgroundColor: 'var(--color-bg-sidebar)',
        borderColor:     'var(--color-border)',
      }}
    >
      {/* Brand header */}
      <div
        className="flex items-center gap-3 px-4 py-4 border-b"
        style={{ borderColor: 'var(--color-border)' }}
      >
        <div
          className="flex items-center justify-center w-9 h-9 rounded"
          style={{ backgroundColor: 'var(--color-amber)', color: '#000' }}
        >
          <Shield size={18} strokeWidth={2.5} />
        </div>
        <div>
          <div
            className="font-mono font-bold text-sm tracking-widest"
            style={{ color: 'var(--color-bright)' }}
          >
            MECHASSIST
          </div>
          <div
            className="font-mono text-xs tracking-wide"
            style={{ color: 'var(--color-muted)' }}
          >
            HMMWV DIAGNOSTICS
          </div>
        </div>
      </div>

      {/* Icon tab row */}
      <SidebarTabs active={effectiveTab} onSelect={handleTab} />

      {/* Tab content area */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {effectiveTab === 'chat' && <SessionList />}
        {effectiveTab === 'docs' && (
          <div className="p-4" style={{ color: 'var(--color-muted)' }}>
            <p className="text-xs">Document browser coming soon.</p>
          </div>
        )}
      </div>

      {/* Status footer */}
      <div
        className="flex items-center gap-2 px-4 py-3 border-t"
        style={{ borderColor: 'var(--color-border)' }}
      >
        <span
          className="inline-block w-2 h-2 rounded-full"
          style={{ backgroundColor: 'var(--color-olive)' }}
        />
        <span className="text-xs font-mono" style={{ color: 'var(--color-muted)' }}>
          CLOUD AI
        </span>
      </div>
    </aside>
  )
}
