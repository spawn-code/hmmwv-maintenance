interface Props {
  value:    string
  onChange: (v: string) => void
}

const VARIANTS = [
  '',
  'M998',
  'M998A1',
  'M1038',
  'M1038A1',
  'M1025',
  'M1025A1',
  'M1025A2',
  'M1026',
  'M1026A1',
  'M1043',
  'M1043A1',
  'M1043A2',
  'M1044',
  'M1044A1',
  'M1045',
  'M1045A1',
  'M1045A2',
  'M1046',
  'M1046A1',
  'M1097',
  'M1097A1',
  'M1097A2',
  'M1113',
  'M1114',
  'M1116',
  'M1151',
  'M1151A1',
  'M1152',
  'M1152A1',
  'M1165',
  'M1165A1',
  'M1167',
]

export default function VehicleSelector({ value, onChange }: Props) {
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
      <option value="">VARIANT</option>
      {VARIANTS.filter(Boolean).map(v => (
        <option key={v} value={v}>{v}</option>
      ))}
    </select>
  )
}
