/**
 * Detect import statements in code and generate additional esm.sh import map entries
 * for packages not in the pre-bundled set.
 */

const PRE_BUNDLED = new Set([
  'react', 'react-dom', 'react-dom/client', 'recharts', 'lucide-react',
  'framer-motion', 'three', 'd3', '@tanstack/react-query', 'zustand',
  'clsx', 'date-fns',
]);

/**
 * Extract npm package names from import statements.
 * Handles: import X from 'pkg', import { X } from 'pkg', import 'pkg'
 */
export function extractImports(code: string): string[] {
  const imports: Set<string> = new Set();
  // Match ES module imports
  const regex = /import\s+(?:[\s\S]*?\s+from\s+)?['"]([^'"./][^'"]*)['"]/g;
  let match;
  while ((match = regex.exec(code)) !== null) {
    let pkg = match[1];
    // Handle scoped packages: @scope/pkg/subpath -> @scope/pkg
    if (pkg.startsWith('@')) {
      const parts = pkg.split('/');
      pkg = parts.slice(0, 2).join('/');
    } else {
      // Regular packages: pkg/subpath -> pkg
      pkg = pkg.split('/')[0];
    }
    imports.add(pkg);
  }
  return Array.from(imports);
}

/**
 * Generate import map entries for packages not in the pre-bundled set.
 * Returns entries like: { "some-lib": "https://esm.sh/some-lib?external=react,react-dom" }
 */
export function generateDynamicImports(code: string): Record<string, string> {
  const detected = extractImports(code);
  const dynamic: Record<string, string> = {};
  for (const pkg of detected) {
    if (PRE_BUNDLED.has(pkg)) continue;
    // Add ?external=react,react-dom for React-dependent packages
    dynamic[pkg] = `https://esm.sh/${pkg}?external=react,react-dom`;
  }
  return dynamic;
}
