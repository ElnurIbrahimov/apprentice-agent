/**
 * importScanner — detects npm package imports in source code
 * and identifies missing packages vs. a package.json.
 */

const NODE_BUILTINS = new Set([
  'assert', 'buffer', 'child_process', 'cluster', 'console', 'constants',
  'crypto', 'dgram', 'dns', 'domain', 'events', 'fs', 'http', 'http2',
  'https', 'module', 'net', 'os', 'path', 'perf_hooks', 'process',
  'punycode', 'querystring', 'readline', 'repl', 'stream', 'string_decoder',
  'sys', 'timers', 'tls', 'tty', 'url', 'util', 'v8', 'vm', 'worker_threads',
  'zlib',
]);

/** Extract npm package names from source code */
export function scanImports(code: string): string[] {
  const packages = new Set<string>();

  // import ... from 'package'
  // import 'package'
  const esImports = code.matchAll(/(?:import\s+(?:[\s\S]*?\s+from\s+)?['"])([^'"./][^'"]*)['"]/g);
  for (const m of esImports) addPackage(packages, m[1]);

  // require('package')
  const requires = code.matchAll(/require\s*\(\s*['"]([^'"./][^'"]*)['"]\s*\)/g);
  for (const m of requires) addPackage(packages, m[1]);

  // import('package')
  const dynamicImports = code.matchAll(/import\s*\(\s*['"]([^'"./][^'"]*)['"]\s*\)/g);
  for (const m of dynamicImports) addPackage(packages, m[1]);

  return Array.from(packages).sort();
}

function addPackage(set: Set<string>, specifier: string) {
  // Extract package name (handle scoped packages like @org/pkg)
  const parts = specifier.split('/');
  let pkgName: string;
  if (parts[0].startsWith('@') && parts.length >= 2) {
    pkgName = `${parts[0]}/${parts[1]}`;
  } else {
    pkgName = parts[0];
  }
  if (!NODE_BUILTINS.has(pkgName) && !pkgName.startsWith('node:')) {
    set.add(pkgName);
  }
}

/** Compare scanned imports against package.json dependencies */
export function getMissingPackages(
  imports: string[],
  packageJson: { dependencies?: Record<string, string>; devDependencies?: Record<string, string> },
): string[] {
  const installed = new Set([
    ...Object.keys(packageJson.dependencies || {}),
    ...Object.keys(packageJson.devDependencies || {}),
  ]);
  return imports.filter(pkg => !installed.has(pkg));
}

/** Scan all files in a project for missing packages */
export function scanProjectImports(
  files: Record<string, string>,
  packageJson?: { dependencies?: Record<string, string>; devDependencies?: Record<string, string> },
): { allImports: string[]; missing: string[] } {
  const allImports = new Set<string>();
  for (const [path, content] of Object.entries(files)) {
    if (/\.(js|jsx|ts|tsx|mjs|cjs)$/.test(path)) {
      for (const pkg of scanImports(content)) {
        allImports.add(pkg);
      }
    }
  }
  const imports = Array.from(allImports).sort();
  const missing = packageJson ? getMissingPackages(imports, packageJson) : imports;
  return { allImports: imports, missing };
}
