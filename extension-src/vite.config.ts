import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    rollupOptions: {
      input: { sidebar: resolve(__dirname, 'sidebar.html') },
      output: {
        entryFileNames: 'sidebar.js',
        chunkFileNames: 'sidebar-[hash].js',
        assetFileNames: (info) =>
          info.name?.endsWith('.css') ? 'sidebar.css' : 'assets/[name][extname]',
      },
    },
  },
  base: './',
})
