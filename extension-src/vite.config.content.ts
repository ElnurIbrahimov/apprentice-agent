import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    sourcemap: 'hidden',
    lib: {
      entry: resolve(__dirname, 'src/content.ts'),
      name: 'AuraContent',
      formats: ['iife'],
      fileName: () => 'content.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
    minify: 'esbuild',
  },
})
