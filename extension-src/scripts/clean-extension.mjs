// Clean stale build artifacts from ../extension before each build.
// Replaces the old rimraf-glob call, which was inconsistent on Windows.
import { readdirSync, rmSync, statSync, existsSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const EXT_DIR = resolve(here, '..', '..', 'extension');

if (!existsSync(EXT_DIR)) {
  process.exit(0);
}

const PATTERNS = [
  /^sidebar-[A-Za-z0-9_-]+\.js$/,
  /^sidebar-[A-Za-z0-9_-]+\.js\.map$/,
  /^pyodide-worker\.js$/,
  /^pyodide-worker\.js\.map$/,
];

const EXACT_DIRS = ['assets'];

let removed = 0;
for (const name of readdirSync(EXT_DIR)) {
  const full = join(EXT_DIR, name);
  try {
    const s = statSync(full);
    if (s.isDirectory() && EXACT_DIRS.includes(name)) {
      rmSync(full, { recursive: true, force: true });
      removed++;
    } else if (s.isFile() && PATTERNS.some((p) => p.test(name))) {
      rmSync(full, { force: true });
      removed++;
    }
  } catch {
    // ignore
  }
}

if (removed > 0) {
  console.log(`[clean-extension] removed ${removed} stale build artifact(s)`);
}
