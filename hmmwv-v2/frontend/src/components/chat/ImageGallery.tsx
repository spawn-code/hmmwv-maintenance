import { useState } from 'react'
import { X, ChevronLeft, ChevronRight, Image } from 'lucide-react'
import type { ImageRef } from '../../types'

interface Props {
  images: ImageRef[]
}

export default function ImageGallery({ images }: Props) {
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null)

  if (!images || images.length === 0) return null

  const openLightbox  = (i: number) => setLightboxIdx(i)
  const closeLightbox = () => setLightboxIdx(null)
  const prev = () => setLightboxIdx(i => i == null ? null : (i - 1 + images.length) % images.length)
  const next = () => setLightboxIdx(i => i == null ? null : (i + 1) % images.length)

  const current = lightboxIdx != null ? images[lightboxIdx] : null

  return (
    <div className="mt-3">
      <div
        className="flex items-center gap-1 mb-2 text-xs"
        style={{ color: 'var(--color-muted)' }}
      >
        <Image size={12} />
        <span>{images.length} diagram{images.length !== 1 ? 's' : ''}</span>
      </div>

      {/* Thumbnail strip */}
      <div className="flex gap-2 flex-wrap">
        {images.map((img, i) => (
          <button
            key={i}
            onClick={() => openLightbox(i)}
            className="rounded overflow-hidden cursor-pointer hover:opacity-80 transition-opacity"
            style={{ border: '1px solid var(--color-border)' }}
          >
            <img
              src={img.url}
              alt={`Diagram from ${img.source} p.${img.page}`}
              className="w-24 h-16 object-contain"
              style={{ backgroundColor: 'var(--color-bg-surface2)' }}
            />
          </button>
        ))}
      </div>

      {/* Lightbox */}
      {current && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ backgroundColor: 'rgba(0,0,0,0.9)' }}
          onClick={closeLightbox}
        >
          <button
            className="absolute top-4 right-4 p-2"
            onClick={closeLightbox}
            style={{ color: 'var(--color-text)' }}
          >
            <X size={20} />
          </button>

          {images.length > 1 && (
            <>
              <button
                className="absolute left-4 p-2"
                onClick={e => { e.stopPropagation(); prev() }}
                style={{ color: 'var(--color-text)' }}
              >
                <ChevronLeft size={28} />
              </button>
              <button
                className="absolute right-4 p-2"
                onClick={e => { e.stopPropagation(); next() }}
                style={{ color: 'var(--color-text)' }}
              >
                <ChevronRight size={28} />
              </button>
            </>
          )}

          <div
            className="max-w-4xl max-h-screen p-4"
            onClick={e => e.stopPropagation()}
          >
            <img
              src={current.url}
              alt={`Diagram from ${current.source} p.${current.page}`}
              className="max-w-full max-h-screen object-contain rounded"
            />
            <p
              className="text-center text-xs mt-2"
              style={{ color: 'var(--color-muted)' }}
            >
              {current.source} — page {current.page}
              {lightboxIdx != null && ` (${lightboxIdx + 1}/${images.length})`}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
