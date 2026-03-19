/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // NextGen dark theme
        'chat-bg': '#030303',
        'chat-sidebar': '#0a0a0c',
        'chat-user': '#030303',
        'chat-assistant': 'rgba(20, 20, 25, 0.4)',
        'chat-border': 'rgba(255, 255, 255, 0.06)',
        'chat-text': '#ededed',
        'chat-text-secondary': '#a1a1aa',
        'chat-text-tertiary': '#444444',
        'chat-accent': '#7c3aed',
        'chat-accent-hover': '#6d28d9',
        // Extended palette for glows and accents
        'aura-purple': '#8b5cf6',
        'aura-purple-light': '#a78bfa',
        'aura-blue': '#3b82f6',
        'aura-glow': 'rgba(139, 92, 246, 0.4)',
        // Surface elevation
        'surface-0': '#09090b',
        'surface-1': '#121214',
        'surface-2': '#1a1a1d',
        'surface-3': '#1e1e21',
        'surface-4': '#232326',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'breathe': 'breathe 2s ease-in-out infinite',
        'fade-in': 'fade-in 0.5s ease-out forwards',
        'slide-up': 'slide-up 0.5s ease-out forwards',
        'slide-up-fade': 'slide-up-fade 0.4s ease-out forwards',
        'scale-in': 'scale-in 0.3s ease-out forwards',
        'progress-fill': 'progress-fill 1s ease-out forwards',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': {
            boxShadow: '0 0 20px rgba(139, 92, 246, 0.3), 0 0 40px rgba(139, 92, 246, 0.1)',
            transform: 'scale(1)',
          },
          '50%': {
            boxShadow: '0 0 30px rgba(139, 92, 246, 0.5), 0 0 60px rgba(139, 92, 246, 0.2)',
            transform: 'scale(1.02)',
          },
        },
        'breathe': {
          '0%, 100%': {
            opacity: '1',
            transform: 'scale(1)',
          },
          '50%': {
            opacity: '0.7',
            transform: 'scale(1.1)',
          },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': {
            opacity: '0',
            transform: 'translateY(20px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        'slide-up-fade': {
          '0%': {
            opacity: '0',
            transform: 'translateY(10px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        'scale-in': {
          '0%': {
            opacity: '0',
            transform: 'scale(0.9)',
          },
          '100%': {
            opacity: '1',
            transform: 'scale(1)',
          },
        },
        'progress-fill': {
          '0%': { width: '0%' },
          '100%': { width: 'var(--progress-width, 0%)' },
        },
        'glow-pulse': {
          '0%, 100%': {
            boxShadow: '0 0 15px var(--glow-color, rgba(16, 163, 127, 0.3))',
          },
          '50%': {
            boxShadow: '0 0 25px var(--glow-color, rgba(16, 163, 127, 0.5))',
          },
        },
      },
      backgroundImage: {
        'radial-subtle': 'radial-gradient(ellipse at center, rgba(68, 70, 84, 0.5) 0%, transparent 70%)',
        'radial-purple': 'radial-gradient(ellipse at center, rgba(139, 92, 246, 0.15) 0%, transparent 60%)',
        'gradient-purple-blue': 'linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%)',
      },
      boxShadow: {
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.4)',
        'glow-purple-lg': '0 0 30px rgba(139, 92, 246, 0.5)',
        'glow-green': '0 0 15px rgba(16, 163, 127, 0.4)',
        'glow-blue': '0 0 15px rgba(59, 130, 246, 0.4)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.2)',
      },
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [],
}
