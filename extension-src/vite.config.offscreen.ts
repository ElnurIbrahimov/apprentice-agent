import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: '../extension',
    emptyOutDir: false,
    sourcemap: 'hidden',
    lib: {
      entry: resolve(__dirname, 'offscreen.ts'),
      name: 'offscreen',
      fileName: () => 'offscreen.js',
      formats: ['iife'],
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
})
