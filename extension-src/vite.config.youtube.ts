import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    outDir: resolve(__dirname, '../extension'),
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'src/youtube-inject.ts'),
      name: 'AuraYoutubeInject',
      formats: ['iife'],
      fileName: () => 'youtube-inject.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
    minify: true,  // Keep it small — runs in page context
  },
})
