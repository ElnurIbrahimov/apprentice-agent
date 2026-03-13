/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#060612',
        s1: '#0e0c1e',
        s2: '#161428',
        s3: '#1e1c32',
        primary: {
          DEFAULT: '#7c3aed',
          dark: '#6d28d9',
          darker: '#5b21b6',
          light: '#a78bfa',
          lighter: '#e0d6ff',
        },
        fg: '#f0eff8',
        muted: '#7a7a9d',
        disabled: '#3a3a5e',
        success: '#10b981',
        danger: '#ef4444',
      },
      borderRadius: {
        sm: '6px',
        md: '10px',
        lg: '14px',
        xl: '18px',
        pill: '20px',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', "'Segoe UI'", 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
