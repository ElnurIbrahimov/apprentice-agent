/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: 'tsconfig.json',
      diagnostics: false,
    }],
  },
  globals: {
    'import.meta': { url: 'file:///mock' },
  },
  setupFiles: ['./src/__tests__/setup.ts'],
  testMatch: ['**/src/__tests__/**/*.test.ts', '**/src/__tests__/**/*.test.tsx'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
};
