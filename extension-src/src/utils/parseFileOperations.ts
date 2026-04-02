export interface FileOperation {
  type: 'create' | 'update' | 'delete';
  path: string;
  content?: string;
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '').replace(/^\/+/, '').trim();
}

export function parseFileOperations(response: string, fallbackPath = 'index.html', allowFallback = true): FileOperation[] {
  const operations: FileOperation[] = [];
  const fileBlockRegex = /===FILE:\s*([^\n=]+?)===\s*([\s\S]*?)===END FILE===/g;
  const deleteBlockRegex = /===DELETE:\s*([^\n=]+?)===/g;

  for (const match of response.matchAll(fileBlockRegex)) {
    const path = normalizePath(match[1]);
    if (!path) continue;
    operations.push({
      type: 'update',
      path,
      content: match[2].replace(/^\r?\n/, ''),
    });
  }

  for (const match of response.matchAll(deleteBlockRegex)) {
    const path = normalizePath(match[1]);
    if (!path) continue;
    operations.push({
      type: 'delete',
      path,
    });
  }

  if (operations.length > 0) {
    const seen = new Set<string>();
    return operations.map((operation) => {
      const key = `${operation.type}:${operation.path}`;
      const normalizedType = operation.type === 'update' && !seen.has(`update:${operation.path}`)
        ? 'create'
        : operation.type;
      seen.add(key);
      return {
        ...operation,
        type: normalizedType as FileOperation['type'],
      };
    });
  }

  if (!allowFallback) return [];

  const trimmed = response.trim();
  if (!trimmed) return [];

  return [
    {
      type: 'update',
      path: normalizePath(fallbackPath) || 'index.html',
      content: trimmed,
    },
  ];
}
