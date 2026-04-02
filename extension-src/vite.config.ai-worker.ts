import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'src/workers/ai-worker.ts'),
      name: 'AIWorker',
      formats: ['iife'],
      fileName: () => 'ai-worker.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
    minify: 'esbuild',
  },
})
