/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ChatGPT-like dark theme
        'chat-bg': '#343541',
        'chat-sidebar': '#202123',
        'chat-user': '#343541',
        'chat-assistant': '#444654',
        'chat-border': '#4d4d4f',
        'chat-text': '#ececf1',
        'chat-text-secondary': '#8e8ea0',
        'chat-accent': '#10a37f',
        'chat-accent-hover': '#1a7f64',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
