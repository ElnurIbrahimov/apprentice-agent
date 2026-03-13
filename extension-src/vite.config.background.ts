import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: '../extension',
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'background.ts'),
      name: 'background',
      fileName: () => 'background.js',
      formats: ['iife'],
    },
    rollupOptions: {
      output: {
        // Service worker — single IIFE file, no code splitting
        inlineDynamicImports: true,
      },
    },
  },
})
