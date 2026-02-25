/**
 * Three animated dots shown while waiting for the first streaming token.
 */
export default function StreamingDots() {
  return (
    <div className="flex items-center gap-1.5 px-1">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="inline-block w-1.5 h-1.5 rounded-full animate-pulse-dot"
          style={{
            backgroundColor: 'var(--color-amber)',
            animationDelay: `${i * 0.2}s`,
          }}
        />
      ))}
    </div>
  )
}
