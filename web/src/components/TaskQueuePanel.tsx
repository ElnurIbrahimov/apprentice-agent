import { useState, useCallback, useRef } from 'react';
import { PlusIcon, PlayIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface Task {
  id: string;
  text: string;
  status: 'pending' | 'running' | 'done' | 'error';
  result?: string;
}

const STATUS_STYLE: Record<string, { dot: string; text: string }> = {
  pending: { dot: 'bg-gray-400', text: 'text-chat-text-secondary' },
  running: { dot: 'bg-blue-400 animate-pulse', text: 'text-blue-400' },
  done:    { dot: 'bg-green-400', text: 'text-green-400' },
  error:   { dot: 'bg-red-400', text: 'text-red-400' },
};

export function TaskQueuePanel() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef(false);

  const addTask = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    setTasks(prev => [...prev, { id: `task-${Date.now()}`, text, status: 'pending' }]);
    setInput('');
  }, [input]);

  const removeTask = useCallback((id: string) => {
    setTasks(prev => prev.filter(t => t.id !== id));
  }, []);

  const runAll = useCallback(async () => {
    setIsRunning(true);
    abortRef.current = false;

    const pending = tasks.filter(t => t.status === 'pending');
    for (const task of pending) {
      if (abortRef.current) break;

      setTasks(prev => prev.map(t => t.id === task.id ? { ...t, status: 'running' } : t));

      try {
        const res = await apiFetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: task.text }),
        });

        if (res.ok) {
          const data = await res.json();
          setTasks(prev => prev.map(t =>
            t.id === task.id ? { ...t, status: 'done', result: data.response?.slice(0, 200) } : t
          ));
        } else {
          setTasks(prev => prev.map(t =>
            t.id === task.id ? { ...t, status: 'error', result: `HTTP ${res.status}` } : t
          ));
        }
      } catch (e: any) {
        setTasks(prev => prev.map(t =>
          t.id === task.id ? { ...t, status: 'error', result: e.message } : t
        ));
      }
    }

    setIsRunning(false);
  }, [tasks]);

  const stopQueue = useCallback(() => {
    abortRef.current = true;
  }, []);

  const clearDone = useCallback(() => {
    setTasks(prev => prev.filter(t => t.status === 'pending' || t.status === 'running'));
  }, []);

  const pendingCount = tasks.filter(t => t.status === 'pending').length;
  const doneCount = tasks.filter(t => t.status === 'done' || t.status === 'error').length;

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--surface-0)' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">Task Queue</h2>
        <div className="flex items-center gap-2">
          {doneCount > 0 && (
            <button onClick={clearDone} className="text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors">
              Clear done
            </button>
          )}
          {isRunning ? (
            <button onClick={stopQueue} className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-red-600/20 text-red-400 text-xs hover:bg-red-600/30 transition-colors">
              Stop
            </button>
          ) : (
            <button
              onClick={runAll}
              disabled={pendingCount === 0}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-chat-accent text-white text-xs disabled:opacity-40 hover:opacity-90 transition-opacity"
            >
              <PlayIcon className="w-3.5 h-3.5" />
              Run All ({pendingCount})
            </button>
          )}
        </div>
      </div>

      {/* Task list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {tasks.length === 0 && (
          <div className="text-center py-12 text-chat-text-secondary text-sm">
            Add tasks below and run them sequentially.
          </div>
        )}
        {tasks.map(task => {
          const style = STATUS_STYLE[task.status];
          return (
            <div key={task.id} className="flex items-start gap-2.5 p-3 rounded-lg border border-chat-border/20" style={{ background: 'var(--surface-1)' }}>
              <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${style.dot}`} />
              <div className="flex-1 min-w-0">
                <p className={`text-xs ${task.status === 'done' ? 'line-through text-chat-text-secondary/50' : 'text-chat-text'}`}>
                  {task.text}
                </p>
                {task.result && (
                  <p className={`text-[10px] mt-1 truncate ${style.text}`}>{task.result}</p>
                )}
              </div>
              {task.status === 'pending' && (
                <button onClick={() => removeTask(task.id)} className="text-chat-text-secondary/40 hover:text-red-400 transition-colors p-0.5">
                  <XMarkIcon className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Input */}
      <div className="p-3 border-t border-chat-border flex-shrink-0">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') addTask(); }}
            placeholder="Add a task..."
            className="flex-1 px-3 py-2 rounded-lg border border-chat-border text-chat-text text-sm outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 text-[16px] sm:text-sm"
            style={{ background: 'var(--surface-2)' }}
          />
          <button
            onClick={addTask}
            disabled={!input.trim()}
            className="p-2 rounded-lg bg-chat-accent text-white disabled:opacity-40 hover:opacity-90 transition-opacity"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
