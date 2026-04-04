import { type UseModelSelectorReturn } from '../hooks/useModelSelector';

interface ModelSelectorProps {
  hook: UseModelSelectorReturn;
  /** Visual variant. 'inline' is the compact row version (most panels).
   *  'block' renders a labelled block with a border (WritePanel style). */
  variant?: 'inline' | 'block';
}

export function ModelSelector({ hook, variant = 'inline' }: ModelSelectorProps) {
  const { selectedModel, setSelectedModel, availableModels, showModelMenu, setShowModelMenu, modelMenuRef } = hook;

  const trigger = (
    <button
      type="button"
      onClick={() => setShowModelMenu(p => !p)}
      className="flex items-center gap-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-2.5 py-1.5 rounded-md"
      style={{
        background: 'var(--border-subtle)',
        border: variant === 'block' ? '1px solid var(--border-default)' : undefined,
      }}
    >
      <span className="max-w-[180px] truncate">
        {selectedModel ? selectedModel.split('/').pop() : 'Auto (recommended)'}
      </span>
      <svg className="w-3 h-3 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    </button>
  );

  const dropdown = showModelMenu && (
    <div
      style={{
        position: 'absolute',
        top: '100%',
        left: 0,
        marginTop: 4,
        width: 240,
        maxHeight: 260,
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 10,
        overflow: 'hidden',
        zIndex: 50,
        boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
      }}
    >
      <div style={{ maxHeight: 260, overflowY: 'auto', padding: 4 }}>
        <button
          onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
          className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
          style={{
            color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
            background: !selectedModel ? 'var(--surface-3)' : 'transparent',
          }}
        >
          Auto (recommended)
        </button>
        {availableModels.map(m => (
          <button
            key={m}
            onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
            style={{
              color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
              background: selectedModel === m ? 'var(--surface-3)' : 'transparent',
            }}
          >
            {m}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div ref={modelMenuRef} className="relative inline-block">
      {trigger}
      {dropdown}
    </div>
  );
}
