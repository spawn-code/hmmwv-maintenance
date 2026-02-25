import { useEffect, useRef } from 'react'

/**
 * Automatically scrolls a container to the bottom whenever `deps` change.
 * Stops auto-scrolling if the user has manually scrolled up (> 100px from bottom).
 */
export function useAutoScroll(deps: unknown[]) {
  const ref        = useRef<HTMLDivElement>(null)
  const userScrolled = useRef(false)

  // Detect manual scroll up
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const handler = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
      userScrolled.current = distFromBottom > 100
    }
    el.addEventListener('scroll', handler, { passive: true })
    return () => el.removeEventListener('scroll', handler)
  }, [])

  // Auto-scroll when deps change (new tokens / messages)
  useEffect(() => {
    if (userScrolled.current) return
    const el = ref.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  // Reset user-scroll flag when a new streaming session begins
  const resetScroll = () => {
    userScrolled.current = false
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }

  return { ref, resetScroll }
}
