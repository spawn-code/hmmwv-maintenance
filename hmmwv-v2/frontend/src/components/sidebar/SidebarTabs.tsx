import { MessageSquare, FileText, Settings } from 'lucide-react'

type Tab = 'chat' | 'docs' | 'settings'

interface Props {
  active:   Tab
  onSelect: (tab: Tab) => void
}

const TABS: { id: Tab; Icon: typeof MessageSquare; label: string }[] = [
  { id: 'chat',     Icon: MessageSquare, label: 'Chat' },
  { id: 'docs',     Icon: FileText,      label: 'Docs' },
  { id: 'settings', Icon: Settings,      label: 'Settings' },
]

export default function SidebarTabs({ active, onSelect }: Props) {
  return (
    <div
      className="flex border-b"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {TABS.map(({ id, Icon, label }) => {
        const isActive = active === id
        return (
          <button
            key={id}
            onClick={() => onSelect(id)}
            title={label}
            className="flex-1 flex flex-col items-center gap-1 py-3 transition-colors cursor-pointer"
            style={{
              color:        isActive ? 'var(--color-amber)' : 'var(--color-muted)',
              borderBottom: isActive ? '2px solid var(--color-amber)' : '2px solid transparent',
              borderTop:    'none',
              borderLeft:   'none',
              borderRight:  'none',
              marginBottom: '-1px',
              background:   'none',
              outline:      'none',
            }}
          >
            <Icon size={16} strokeWidth={isActive ? 2.5 : 1.5} />
            <span className="text-xs font-mono tracking-wider">{label.toUpperCase()}</span>
          </button>
        )
      })}
    </div>
  )
}
