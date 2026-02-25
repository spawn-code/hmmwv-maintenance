import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  server: {
    port: 5173,
    proxy: {
      '/sessions':  { target: 'http://localhost:8000', changeOrigin: true },
      '/settings':  { target: 'http://localhost:8000', changeOrigin: true },
      '/knowledge': { target: 'http://localhost:8000', changeOrigin: true },
      '/chat':      { target: 'http://localhost:8000', changeOrigin: true },
      '/images':    { target: 'http://localhost:8000', changeOrigin: true },
      '/health':    { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
