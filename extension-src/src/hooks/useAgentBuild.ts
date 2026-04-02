/**
 * useAgentBuild — connects WebCreatorPanel build mode to the backend
 * agent-powered build service at /api/agent/build.
 *
 * The backend runs the full agentic loop (write_file, edit_file, etc.)
 * and broadcasts file updates + progress via the artifacts WebSocket.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { HTTP, getAuthHeaders } from '../api';

const ARTIFACTS_WS_URL = HTTP.replace(/^http/, 'ws') + '/api/artifacts/stream';

export interface AgentBuildFile {
  path: string;
  content: string;
}

export interface AgentBuildPlanItem {
  path: string;
  purpose: string;
  priority: number;
}

export type AgentBuildStatus = 'idle' | 'planning' | 'building' | 'completed' | 'error' | 'cancelled';

interface AgentBuildState {
  status: AgentBuildStatus;
  taskId: string | null;
  plan: AgentBuildPlanItem[];
  filesCreated: string[];
  currentFile: string;
  progressStep: number;
  progressTotal: number;
  progressMessage: string;
  error: string;
}

interface AgentBuildActions {
  /** Request a build plan from the backend agent */
  requestPlan: (description: string, framework: string, seedFiles?: AgentBuildFile[]) => Promise<AgentBuildPlanItem[]>;
  /** Start a build with an approved plan */
  startBuild: (description: string, framework: string, plan: AgentBuildPlanItem[], seedFiles?: AgentBuildFile[], model?: string) => Promise<string>;
  /** Cancel the active build */
  cancelBuild: () => Promise<void>;
  /** Reset all state */
  reset: () => void;
}

interface BuildWsEvent {
  type: string;
  task_id?: string;
  filename?: string;
  code?: string;
  artifact_type?: string;
  timestamp?: number;
  step?: number;
  total?: number;
  current_file?: string;
  message?: string;
  files_created?: number | string[];
  paths?: string[];
  error?: string;
  plan?: AgentBuildPlanItem[];
  framework?: string;
  workspace?: string;
}

export function useAgentBuild(
  onFileUpdate?: (filename: string, code: string, artifactType: string) => void,
): [AgentBuildState, AgentBuildActions] {
  const [state, setState] = useState<AgentBuildState>({
    status: 'idle',
    taskId: null,
    plan: [],
    filesCreated: [],
    currentFile: '',
    progressStep: 0,
    progressTotal: 0,
    progressMessage: '',
    error: '',
  });

  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const taskIdRef = useRef<string | null>(null);
  const onFileUpdateRef = useRef(onFileUpdate);
  onFileUpdateRef.current = onFileUpdate;

  // Connect to artifact WebSocket when a build is active
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    const socket = new WebSocket(ARTIFACTS_WS_URL);

    socket.onmessage = (ev) => {
      let d: BuildWsEvent;
      try { d = JSON.parse(ev.data); } catch { return; }

      // Only process events for our active task
      const activeTaskId = taskIdRef.current;

      if (d.type === 'artifact_update' && d.filename && d.code != null) {
        onFileUpdateRef.current?.(d.filename, d.code, d.artifact_type || 'html');
      }

      if (d.task_id && activeTaskId && d.task_id !== activeTaskId) return;

      if (d.type === 'build_progress') {
        setState(prev => ({
          ...prev,
          currentFile: d.current_file || prev.currentFile,
          progressStep: d.step ?? prev.progressStep,
          progressTotal: d.total ?? prev.progressTotal,
          progressMessage: d.message || prev.progressMessage,
        }));
      } else if (d.type === 'build_complete') {
        const paths = d.paths || [];
        setState(prev => ({
          ...prev,
          status: 'completed',
          filesCreated: paths.length > 0 ? paths : prev.filesCreated,
          currentFile: '',
          progressMessage: `Build complete: ${d.files_created || paths.length} files created`,
        }));
      } else if (d.type === 'build_error') {
        setState(prev => ({
          ...prev,
          status: 'error',
          error: d.error || 'Build failed',
          currentFile: '',
          progressMessage: d.error || 'Build failed',
        }));
      } else if (d.type === 'build_cancelled') {
        setState(prev => ({
          ...prev,
          status: 'cancelled',
          currentFile: '',
          progressMessage: 'Build cancelled',
        }));
      }
    };

    socket.onclose = () => {
      wsRef.current = null;
      if (taskIdRef.current) {
        retryRef.current = setTimeout(connectWs, 3000);
      }
    };

    wsRef.current = socket;
  }, []);

  const disconnectWs = useCallback(() => {
    if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null; }
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
  }, []);

  // Auto-connect when build is active, disconnect when idle
  useEffect(() => {
    if (state.status === 'building') {
      connectWs();
    } else if (state.status === 'idle') {
      disconnectWs();
    }
    return () => { disconnectWs(); };
  }, [state.status, connectWs, disconnectWs]);

  const requestPlan = useCallback(async (
    description: string,
    framework: string,
    seedFiles?: AgentBuildFile[],
  ): Promise<AgentBuildPlanItem[]> => {
    setState(prev => ({ ...prev, status: 'planning', error: '', progressMessage: 'Planning files...' }));
    try {
      const resp = await fetch(`${HTTP}/api/agent/build/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          description,
          framework,
          files: (seedFiles || []).map(f => ({ path: f.path, content: f.content })),
        }),
      });
      if (!resp.ok) throw new Error(`Plan request failed: ${resp.status}`);
      const data = await resp.json();
      const plan: AgentBuildPlanItem[] = data.plan || [];
      setState(prev => ({
        ...prev,
        status: plan.length > 0 ? 'idle' : 'error',
        plan,
        error: plan.length === 0 ? 'Plan returned no files' : '',
        progressMessage: plan.length > 0 ? `Planned ${plan.length} files` : 'No files planned',
      }));
      return plan;
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        status: 'error',
        error: err.message || 'Planning failed',
        progressMessage: err.message || 'Planning failed',
      }));
      return [];
    }
  }, []);

  const startBuild = useCallback(async (
    description: string,
    framework: string,
    plan: AgentBuildPlanItem[],
    seedFiles?: AgentBuildFile[],
    model?: string,
  ): Promise<string> => {
    setState(prev => ({
      ...prev,
      status: 'building',
      plan,
      filesCreated: [],
      currentFile: plan[0]?.path || '',
      progressStep: 0,
      progressTotal: plan.length,
      progressMessage: 'Starting agent build...',
      error: '',
    }));

    try {
      const resp = await fetch(`${HTTP}/api/agent/build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          description,
          framework,
          files: (seedFiles || []).map(f => ({ path: f.path, content: f.content })),
          plan: plan.map(p => ({ path: p.path, purpose: p.purpose, priority: p.priority })),
          model: model || undefined,
          max_iterations: 24,
        }),
      });
      if (!resp.ok) throw new Error(`Build request failed: ${resp.status}`);
      const data = await resp.json();
      const taskId = data.task_id;
      taskIdRef.current = taskId;
      setState(prev => ({ ...prev, taskId }));
      return taskId;
    } catch (err: any) {
      setState(prev => ({
        ...prev,
        status: 'error',
        error: err.message || 'Failed to start build',
        progressMessage: err.message || 'Failed to start build',
      }));
      return '';
    }
  }, []);

  const cancelBuild = useCallback(async () => {
    const taskId = taskIdRef.current;
    if (!taskId) return;
    try {
      await fetch(`${HTTP}/api/agent/build/${taskId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
      });
    } catch { /* ignore cancel errors */ }
    setState(prev => ({
      ...prev,
      status: 'cancelled',
      currentFile: '',
      progressMessage: 'Build cancelled',
    }));
    taskIdRef.current = null;
  }, []);

  const reset = useCallback(() => {
    taskIdRef.current = null;
    disconnectWs();
    setState({
      status: 'idle',
      taskId: null,
      plan: [],
      filesCreated: [],
      currentFile: '',
      progressStep: 0,
      progressTotal: 0,
      progressMessage: '',
      error: '',
    });
  }, [disconnectWs]);

  return [state, { requestPlan, startBuild, cancelBuild, reset }];
}
