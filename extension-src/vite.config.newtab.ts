import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'src/newtab.ts'),
      name: 'AuraNewTab',
      formats: ['iife'],
      fileName: () => 'newtab.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
    minify: 'esbuild',
  },
})
