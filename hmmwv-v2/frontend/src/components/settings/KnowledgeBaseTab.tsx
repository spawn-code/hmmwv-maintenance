import { useEffect, useState } from 'react'
import { Database, RefreshCw, Image } from 'lucide-react'
import { knowledgeApi } from '../../api/settings'
import type { KnowledgeStats } from '../../types'

export default function KnowledgeBaseTab() {
  const [stats,     setStats]     = useState<KnowledgeStats | null>(null)
  const [loading,   setLoading]   = useState(true)
  const [indexing,  setIndexing]  = useState(false)
  const [rendering, setRendering] = useState(false)
  const [msg,       setMsg]       = useState<string | null>(null)

  const loadStats = async () => {
    setLoading(true)
    try {
      setStats(await knowledgeApi.stats())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { loadStats() }, [])

  const handleIndex = async () => {
    setIndexing(true)
    setMsg(null)
    try {
      const r = await knowledgeApi.index()
      setMsg(`Indexed ${r.chunks_added} new chunks from ${r.indexed} PDFs.`)
      await loadStats()
    } catch {
      setMsg('Indexing failed.')
    } finally {
      setIndexing(false)
    }
  }

  const handleRenderPages = async () => {
    setRendering(true)
    setMsg(null)
    try {
      const r = await knowledgeApi.renderPages(150)
      setMsg(r.message)
    } catch {
      setMsg('Failed to start page rendering.')
    } finally {
      setRendering(false)
    }
  }

  const statRow = (label: string, value: string | number) => (
    <div className="flex justify-between py-2" style={{ borderBottom: '1px solid var(--color-border)' }}>
      <span className="text-xs font-mono" style={{ color: 'var(--color-muted)' }}>{label}</span>
      <span className="text-xs font-mono" style={{ color: 'var(--color-bright)' }}>{value}</span>
    </div>
  )

  return (
    <div className="space-y-6">
      <div
        className="rounded p-4"
        style={{ backgroundColor: 'var(--color-bg-surface2)', border: '1px solid var(--color-border)' }}
      >
        <div className="flex items-center gap-2 mb-4">
          <Database size={14} style={{ color: 'var(--color-amber)' }} />
          <span className="text-xs font-mono tracking-widest" style={{ color: 'var(--color-muted)' }}>
            KNOWLEDGE BASE STATS
          </span>
        </div>

        {loading ? (
          <p className="text-xs" style={{ color: 'var(--color-muted)' }}>Loading…</p>
        ) : stats ? (
          <>
            {statRow('Total chunks (BM25 index)', stats.total_chunks.toLocaleString())}
            {statRow('Source PDFs',               stats.num_sources)}
            {statRow('Unprocessed PDFs',          stats.unprocessed_pdfs)}
          </>
        ) : (
          <p className="text-xs" style={{ color: 'var(--color-danger)' }}>Failed to load stats.</p>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-3 flex-wrap">
        <button
          onClick={handleIndex}
          disabled={indexing}
          className="flex items-center gap-2 px-4 py-2 rounded text-sm font-mono cursor-pointer disabled:opacity-40"
          style={{
            backgroundColor: 'var(--color-bg-surface2)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text)',
          }}
        >
          <RefreshCw size={13} className={indexing ? 'animate-spin' : ''} />
          {indexing ? 'Indexing…' : 'Index New PDFs'}
        </button>

        <button
          onClick={handleRenderPages}
          disabled={rendering}
          className="flex items-center gap-2 px-4 py-2 rounded text-sm font-mono cursor-pointer disabled:opacity-40"
          style={{
            backgroundColor: 'var(--color-bg-surface2)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text)',
          }}
          title="Pre-render all PDF pages as PNG images — captures vector diagrams that are invisible to the standard image extractor"
        >
          <Image size={13} className={rendering ? 'animate-spin' : ''} />
          {rendering ? 'Starting…' : 'Pre-Render Diagrams'}
        </button>
      </div>

      {msg && (
        <p className="text-xs rounded px-3 py-2"
          style={{
            backgroundColor: 'var(--color-bg-surface2)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-olive)',
          }}
        >
          {msg}
        </p>
      )}

      <div
        className="rounded p-4 text-xs"
        style={{
          backgroundColor: 'var(--color-bg-surface2)',
          border: '1px solid var(--color-border)',
          color: 'var(--color-muted)',
        }}
      >
        <p className="mb-1 font-mono" style={{ color: 'var(--color-amber)' }}>HOW TO ADD NEW DOCUMENTS</p>
        <p>1. Place PDF files in the <code className="font-mono">knowledge_base/</code> directory.</p>
        <p>2. Click "Index New PDFs" above.</p>
        <p>3. New documents will be searchable immediately after indexing.</p>
      </div>

      <div
        className="rounded p-4 text-xs space-y-1"
        style={{
          backgroundColor: 'var(--color-bg-surface2)',
          border: '1px solid var(--color-border)',
          color: 'var(--color-muted)',
        }}
      >
        <p className="mb-1 font-mono" style={{ color: 'var(--color-amber)' }}>ABOUT DIAGRAM EXTRACTION</p>
        <p>
          Military Technical Manuals draw exploded views and wiring diagrams as <strong style={{ color: 'var(--color-bright)' }}>vector graphics</strong>,
          not embedded images. The standard extractor only captures raster photos.
        </p>
        <p className="mt-1">
          Click <strong style={{ color: 'var(--color-bright)' }}>Pre-Render Diagrams</strong> to render every PDF page
          as a PNG (150 dpi). This runs in the background (~1–3 s/page) and permanently
          improves diagram coverage for all future queries.
        </p>
      </div>
    </div>
  )
}
