/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode, command }) => {
  // Fail fast: a production bundle with no VITE_API_URL would silently call the
  // visitor's own localhost.
  if (command === 'build' && mode === 'production') {
    const env = loadEnv(mode, process.cwd(), 'VITE_')
    if (!env.VITE_API_URL && !process.env.VITE_API_URL && !process.env.ALLOW_DEFAULT_API_URL) {
      console.warn(
        '\n[citymind] VITE_API_URL is not set: the production bundle will target http://127.0.0.1:8000/api. ' +
          'Set VITE_API_URL (see .env.example) for any non-local deployment.\n',
      )
    }
  }
  return {
    plugins: [react()],
    build: {
      rollupOptions: {
        output: {
          manualChunks(id: string) {
            if (id.includes('node_modules/recharts') || id.includes('node_modules/d3-') || id.includes('node_modules/victory-vendor')) return 'charts'
            if (id.includes('node_modules/react') || id.includes('node_modules/scheduler') || id.includes('node_modules/react-router')) return 'react'
          },
        },
      },
    },
    test: {
      environment: 'jsdom',
      globals: true,
      // Let the responsive-CSS sanity test read the real stylesheet text.
      css: { include: [/styles\/index\.css/] },
      setupFiles: './src/test/setup.ts',
    },
  }
})
