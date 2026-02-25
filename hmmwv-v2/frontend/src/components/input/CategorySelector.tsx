interface Props {
  value:    string
  onChange: (v: string) => void
}

const CATEGORIES = [
  '',
  'Engine & Powertrain',
  'Electrical Systems',
  'Fuel System',
  'Brake System',
  'Steering & Suspension',
  'CTIS (Central Tire Inflation)',
  'Transmission & Drive Train',
  'Cooling System',
  'Body & Accessories',
  'Preventive Maintenance',
  'Troubleshooting',
  'General',
]

export default function CategorySelector({ value, onChange }: Props) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="text-xs px-2 py-1 rounded cursor-pointer"
      style={{
        backgroundColor: 'var(--color-bg-surface2)',
        border: '1px solid var(--color-border)',
        color: value ? 'var(--color-amber)' : 'var(--color-muted)',
        fontFamily: 'var(--font-mono)',
        outline: 'none',
      }}
    >
      <option value="">CATEGORY</option>
      {CATEGORIES.filter(Boolean).map(c => (
        <option key={c} value={c}>{c}</option>
      ))}
    </select>
  )
}
