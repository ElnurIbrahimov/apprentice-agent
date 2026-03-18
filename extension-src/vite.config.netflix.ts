import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'src/netflix-inject.ts'),
      name: 'AuraNetflixInject',
      formats: ['iife'],
      fileName: () => 'netflix-inject.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
    minify: true,  // Keep it small — runs in page context
  },
})
