import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    sourcemap: 'hidden',
    rollupOptions: {
      input: { sidebar: resolve(__dirname, 'sidebar.html') },
      output: {
        entryFileNames: 'sidebar.js',
        chunkFileNames: 'sidebar-[hash].js',
        assetFileNames: (info) =>
          info.name?.endsWith('.css') ? 'sidebar.css' : 'assets/[name][extname]',
        manualChunks: (id) => {
          const normalizedId = id.replace(/\\/g, '/')

          if (!normalizedId.includes('/node_modules/')) return undefined

          if (
            normalizedId.includes('/node_modules/react/') ||
            normalizedId.includes('/node_modules/react-dom/')
          ) {
            return 'vendor-react'
          }

          if (normalizedId.includes('/node_modules/@codemirror/merge/')) {
            return 'vendor-editor-merge'
          }

          if (normalizedId.includes('/node_modules/@codemirror/lang-')) {
            return undefined
          }

          if (normalizedId.includes('/node_modules/@lezer/')) {
            return undefined
          }

          if (normalizedId.includes('/node_modules/@codemirror/')) {
            return 'vendor-editor-core'
          }

          if (
            normalizedId.includes('/node_modules/lucide-react/') ||
            normalizedId.includes('/node_modules/dompurify/') ||
            normalizedId.includes('/node_modules/zustand/')
          ) {
            return 'vendor-ui'
          }

          return 'vendor-misc'
        },
      },
    },
  },
  base: './',
})
