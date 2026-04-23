import { useEffect, useMemo, useState, useCallback } from 'react';
import {
  BeakerIcon,
  PlayIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface MethodParamSpec {
  name: string;
  type: string;
  required: boolean;
  default: unknown;
}

interface MethodSpec {
  name: string;
  doc: string | null;
  params: MethodParamSpec[];
}

interface ToolSpec {
  name: string;
  description: string;
  loaded: boolean;
  methods: MethodSpec[];
}

interface InvokeResponse {
  success: boolean;
  tool: string;
  method: string;
  result?: unknown;
  error?: string;
  elapsed_ms?: number;
}

/** Coerce the string from a form input into the shape the tool expects. */
function coerce(raw: string, type: string, fallback: unknown): unknown {
  if (raw === '' && fallback !== null && fallback !== undefined) return fallback;
  if (raw === '') return undefined;
  const t = (type || '').toLowerCase();
  if (t === 'int' || t === 'integer') {
    const n = parseInt(raw, 10);
    return Number.isNaN(n) ? raw : n;
  }
  if (t === 'float' || t === 'number') {
    const n = parseFloat(raw);
    return Number.isNaN(n) ? raw : n;
  }
  if (t === 'bool' || t === 'boolean') {
    const lower = raw.trim().toLowerCase();
    if (['true', '1', 'yes', 'y'].includes(lower)) return true;
    if (['false', '0', 'no', 'n'].includes(lower)) return false;
    return raw;
  }
  if (t === 'list' || t === 'dict' || t.startsWith('list[') || t.startsWith('dict[')) {
    try {
      return JSON.parse(raw);
    } catch {
      // Treat as comma-separated for lists, fallback to raw for dicts
      if (t.startsWith('list') || t === 'list') {
        return raw.split(',').map((s) => s.trim()).filter(Boolean);
      }
      return raw;
    }
  }
  return raw;
}

function ParamInput({
  param,
  value,
  onChange,
}: {
  param: MethodParamSpec;
  value: string;
  onChange: (next: string) => void;
}) {
  const t = (param.type || '').toLowerCase();
  const isBool = t === 'bool' || t === 'boolean';
  const placeholder =
    param.default !== null && param.default !== undefined
      ? `default: ${JSON.stringify(param.default)}`
      : `${param.type}${param.required ? ' · required' : ''}`;

  if (isBool) {
    return (
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 rounded-md bg-chat-bg border border-chat-border text-chat-text text-sm focus:border-chat-accent outline-none"
      >
        <option value="">{placeholder}</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    );
  }

  const multiline = t === 'dict' || t.startsWith('dict[') || t === 'list' || t.startsWith('list[');
  if (multiline) {
    return (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={3}
        className="w-full px-3 py-2 rounded-md bg-chat-bg border border-chat-border text-chat-text font-mono text-xs focus:border-chat-accent outline-none resize-y"
      />
    );
  }

  return (
    <input
      type={t === 'int' || t === 'integer' || t === 'float' || t === 'number' ? 'number' : 'text'}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full px-3 py-2 rounded-md bg-chat-bg border border-chat-border text-chat-text text-sm focus:border-chat-accent outline-none"
    />
  );
}

export function ToolPlayground() {
  const [registry, setRegistry] = useState<ToolSpec[]>([]);
  const [loadingRegistry, setLoadingRegistry] = useState(true);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<string | null>(null);
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<InvokeResponse | null>(null);

  // Load registry
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingRegistry(true);
      setRegistryError(null);
      try {
        const res = await apiFetch('/api/tools/registry');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as ToolSpec[];
        if (!cancelled) {
          setRegistry(data);
          // Auto-select first loaded tool with methods
          const firstReady = data.find((t) => t.methods.length > 0);
          if (firstReady) {
            setSelectedTool(firstReady.name);
            setSelectedMethod(firstReady.methods[0]?.name ?? null);
          }
        }
      } catch (err) {
        if (!cancelled) setRegistryError(err instanceof Error ? err.message : 'Failed to load');
      } finally {
        if (!cancelled) setLoadingRegistry(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredRegistry = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return registry;
    return registry.filter(
      (t) =>
        t.name.includes(q) ||
        t.description.toLowerCase().includes(q) ||
        t.methods.some((m) => m.name.includes(q)),
    );
  }, [registry, search]);

  const currentTool = useMemo(
    () => registry.find((t) => t.name === selectedTool) ?? null,
    [registry, selectedTool],
  );
  const currentMethod = useMemo(
    () => currentTool?.methods.find((m) => m.name === selectedMethod) ?? null,
    [currentTool, selectedMethod],
  );

  // Reset form when method changes
  useEffect(() => {
    setFormValues({});
    setResponse(null);
  }, [selectedTool, selectedMethod]);

  const handleSelectTool = useCallback((name: string) => {
    setSelectedTool(name);
    const tool = registry.find((t) => t.name === name);
    setSelectedMethod(tool?.methods[0]?.name ?? null);
  }, [registry]);

  const handleRun = useCallback(async () => {
    if (!currentTool || !currentMethod) return;

    // Build kwargs from form values, skipping empties
    const args: Record<string, unknown> = {};
    for (const p of currentMethod.params) {
      const raw = formValues[p.name] ?? '';
      if (raw === '' && !p.required) continue;
      const v = coerce(raw, p.type, p.default);
      if (v !== undefined) args[p.name] = v;
    }

    setRunning(true);
    setResponse(null);
    try {
      const res = await apiFetch('/api/tools/invoke', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tool: currentTool.name, method: currentMethod.name, args }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        setResponse({
          success: false,
          tool: currentTool.name,
          method: currentMethod.name,
          error: `HTTP ${res.status}${text ? ` — ${text.slice(0, 300)}` : ''}`,
        });
      } else {
        const data = (await res.json()) as InvokeResponse;
        setResponse(data);
      }
    } catch (err) {
      setResponse({
        success: false,
        tool: currentTool.name,
        method: currentMethod.name,
        error: err instanceof Error ? err.message : 'Request failed',
      });
    } finally {
      setRunning(false);
    }
  }, [currentTool, currentMethod, formValues]);

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <BeakerIcon className="w-5 h-5 text-violet-400" />
        <h3 className="text-chat-text font-medium">Tool Playground</h3>
        <span className="ml-auto text-xs text-chat-text-secondary">
          {registry.length} tool{registry.length === 1 ? '' : 's'} · read-only invokes only
        </span>
      </div>

      {loadingRegistry && (
        <div className="text-sm text-chat-text-secondary py-6 text-center">Loading tool registry...</div>
      )}

      {registryError && !loadingRegistry && (
        <div className="flex items-start gap-2 p-3 rounded-md bg-red-900/20 border border-red-800/40 text-sm text-red-300">
          <ExclamationTriangleIcon className="w-4 h-4 mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-medium">Registry unavailable</div>
            <div className="text-red-400/80 text-xs mt-0.5">{registryError}</div>
          </div>
        </div>
      )}

      {!loadingRegistry && !registryError && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Tool list */}
          <div className="md:col-span-1 space-y-2">
            <div className="relative">
              <MagnifyingGlassIcon className="w-4 h-4 absolute left-2.5 top-2.5 text-chat-text-secondary" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search tools..."
                className="w-full pl-8 pr-3 py-2 rounded-md bg-chat-bg border border-chat-border text-chat-text text-sm focus:border-chat-accent outline-none"
              />
            </div>
            <div className="max-h-[420px] overflow-y-auto space-y-1 pr-1">
              {filteredRegistry.map((tool) => {
                const isSelected = tool.name === selectedTool;
                const disabled = tool.methods.length === 0;
                return (
                  <button
                    key={tool.name}
                    onClick={() => handleSelectTool(tool.name)}
                    disabled={disabled}
                    className={`w-full text-left px-3 py-2 rounded-md border transition-colors ${
                      isSelected
                        ? 'bg-violet-900/30 border-violet-700/60 text-chat-text'
                        : disabled
                        ? 'bg-chat-bg/40 border-chat-border/40 text-chat-text-secondary/60 cursor-not-allowed'
                        : 'bg-chat-bg border-chat-border text-chat-text hover:border-chat-accent'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs truncate">{tool.name}</span>
                      <span
                        className={`ml-auto text-[10px] px-1.5 py-0.5 rounded-full ${
                          tool.loaded
                            ? 'bg-emerald-900/40 text-emerald-300'
                            : 'bg-gray-700/40 text-gray-400'
                        }`}
                      >
                        {tool.loaded ? 'loaded' : 'lazy'}
                      </span>
                    </div>
                    <div className="text-[11px] text-chat-text-secondary mt-0.5 line-clamp-2">
                      {tool.description}
                    </div>
                    <div className="text-[10px] text-chat-text-secondary/70 mt-1">
                      {tool.methods.length} method{tool.methods.length === 1 ? '' : 's'}
                    </div>
                  </button>
                );
              })}
              {filteredRegistry.length === 0 && (
                <div className="text-xs text-chat-text-secondary text-center py-4">
                  No tools match "{search}"
                </div>
              )}
            </div>
          </div>

          {/* Method + form + response */}
          <div className="md:col-span-2 space-y-3">
            {!currentTool && (
              <div className="text-sm text-chat-text-secondary py-8 text-center flex flex-col items-center gap-2">
                <SparklesIcon className="w-6 h-6 text-violet-400/60" />
                Select a tool on the left to see its methods
              </div>
            )}

            {currentTool && (
              <>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-chat-text font-mono text-sm">{currentTool.name}</span>
                    <span className="text-xs text-chat-text-secondary">·</span>
                    <span className="text-xs text-chat-text-secondary">
                      {currentTool.methods.length} method{currentTool.methods.length === 1 ? '' : 's'}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {currentTool.methods.map((m) => (
                      <button
                        key={m.name}
                        onClick={() => setSelectedMethod(m.name)}
                        className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors ${
                          selectedMethod === m.name
                            ? 'bg-violet-900/40 text-violet-200 border border-violet-700/60'
                            : 'bg-chat-bg text-chat-text-secondary border border-chat-border hover:border-chat-accent'
                        }`}
                      >
                        {m.name}
                      </button>
                    ))}
                  </div>
                </div>

                {currentMethod && (
                  <>
                    {currentMethod.doc && (
                      <div className="text-xs text-chat-text-secondary leading-relaxed px-1">
                        {currentMethod.doc}
                      </div>
                    )}

                    <div className="space-y-2">
                      {currentMethod.params.length === 0 && (
                        <div className="text-xs text-chat-text-secondary italic">
                          No parameters
                        </div>
                      )}
                      {currentMethod.params.map((p) => (
                        <div key={p.name} className="space-y-1">
                          <label className="flex items-baseline gap-2 text-xs">
                            <span className="font-mono text-chat-text">{p.name}</span>
                            <span className="text-chat-text-secondary">{p.type}</span>
                            {p.required && (
                              <span className="text-red-400">*</span>
                            )}
                          </label>
                          <ParamInput
                            param={p}
                            value={formValues[p.name] ?? ''}
                            onChange={(v) => setFormValues((prev) => ({ ...prev, [p.name]: v }))}
                          />
                        </div>
                      ))}
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleRun}
                        disabled={running || !currentTool || !currentMethod}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-violet-700/60 hover:bg-violet-600/70 text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        <PlayIcon className="w-4 h-4" />
                        {running ? 'Running...' : 'Run'}
                      </button>
                      {response && (
                        <span className="text-xs text-chat-text-secondary flex items-center gap-1">
                          <ClockIcon className="w-3 h-3" />
                          {response.elapsed_ms}ms ·{' '}
                          <span
                            className={response.success ? 'text-emerald-400' : 'text-red-400'}
                          >
                            {response.success ? 'success' : 'failed'}
                          </span>
                        </span>
                      )}
                    </div>

                    {response && (
                      <div className="space-y-1">
                        <div className="text-xs text-chat-text-secondary">Result</div>
                        <pre className="text-xs font-mono bg-chat-bg border border-chat-border rounded-md p-3 overflow-auto max-h-[320px] text-chat-text whitespace-pre-wrap break-words">
                          {response.error
                            ? response.error
                            : JSON.stringify(response.result, null, 2)}
                        </pre>
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
