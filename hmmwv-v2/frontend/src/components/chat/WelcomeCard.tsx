interface Props {
  onCategoryClick: (text: string) => void
}

const CATEGORIES = [
  'Engine & drivetrain issues',
  'Electrical system diagnostics',
  'Brake & steering problems',
  'CTIS (Central Tire Inflation System)',
  'Body & accessories maintenance',
]

export default function WelcomeCard({ onCategoryClick }: Props) {
  return (
    <div
      className="mx-auto w-full max-w-2xl rounded"
      style={{
        backgroundColor: 'var(--color-bg-surface)',
        border: '1px solid var(--color-border)',
        padding: '1.75rem 2rem',
        marginTop: '3rem',
      }}
    >
      {/* Title */}
      <h1
        className="font-mono font-bold text-lg tracking-wide mb-3"
        style={{ color: 'var(--color-bright)' }}
      >
        🔧 HMMWV MechAssist Ready
      </h1>

      <p className="text-sm mb-4" style={{ color: 'var(--color-text)' }}>
        I'm your HMMWV diagnostic and repair assistant. I can help you with:
      </p>

      {/* Category buttons */}
      <ul className="mb-5 space-y-1">
        {CATEGORIES.map(cat => (
          <li key={cat}>
            <button
              onClick={() => onCategoryClick(cat)}
              className="text-sm text-left cursor-pointer underline-offset-2 hover:underline transition-opacity hover:opacity-80"
              style={{ color: 'var(--color-amber)', background: 'none', border: 'none', padding: 0 }}
            >
              {cat}
            </button>
          </li>
        ))}
      </ul>

      {/* Prompt */}
      <p className="text-sm mb-4">
        <span style={{ color: 'var(--color-amber)' }}>Tell me what's going on...</span>
      </p>

      {/* Safety note */}
      <p
        className="text-xs italic"
        style={{ color: 'var(--color-muted)' }}
      >
        ⚠️ Always ensure the vehicle is parked on level ground, engine off, and
        parking brake set before performing maintenance.
      </p>
    </div>
  )
}
