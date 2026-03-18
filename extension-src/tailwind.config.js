/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        s1: 'var(--s1)',
        s2: 'var(--s2)',
        s3: 'var(--s3)',
        primary: {
          DEFAULT: 'var(--p)',
          dark: 'var(--p2)',
          darker: 'var(--p3)',
          light: 'var(--pl)',
          lighter: 'var(--pl2)',
        },
        fg: 'var(--tx)',
        muted: 'var(--mu)',
        disabled: 'var(--di)',
        success: 'var(--gr)',
        danger: 'var(--rd)',
      },
      borderRadius: {
        sm: 'var(--r-sm)',
        md: 'var(--r-md)',
        lg: 'var(--r-lg)',
        xl: 'var(--r-xl)',
        pill: 'var(--r-pill)',
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', "'Segoe UI'", 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
