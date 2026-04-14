import { HTTP, getAuthHeaders } from '../api';
import { sendMsg } from '../ext';
import type { AgentStep } from '../types';
import {
  attachCDP,
  detachCDP,
  cdpScreenshot,
  cdpClick,
  cdpType,
  cdpScroll,
  cdpResolveSelector,
} from './cdp';

const MAX_STEPS = 15;
const POST_ACTION_DELAY_MS = 600;
const POST_ACTION_DELAY_CDP_MS = 200;
const POST_NAV_DELAY_MS = 2500;
// Keep screenshots well under the backend's 4.5 MB base64 cap.
// ~3 MB of base64 ≈ 2.2 MB PNG/JPEG — plenty for a viewport screenshot at quality 75.
const MAX_SCREENSHOT_BASE64_LEN = 3_000_000;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

interface DomElement {
  index: number;
  type: string;
  text: string;
  selector: string;
  href?: string;
}

interface DomSnapshot {
  ok: boolean;
  dom?: DomElement[];
  url?: string;
  title?: string;
}

interface AgentAction {
  action: AgentStep['action'];
  selector?: string;
  text?: string;
  url?: string;
  amount?: number;
  description?: string;
}

interface RunAgentLoopArgs {
  task: string;
  model: string | null;
  signal: AbortSignal;
  onStep: (step: AgentStep) => void;
  onComplete: (reason: 'done' | 'max-steps' | 'stopped' | 'error', errorMessage?: string) => void;
  /** When true, attach chrome.debugger and use CDP for input + vision. */
  powerMode?: boolean;
}

async function getActiveTabId(): Promise<number | null> {
  return new Promise((resolve) => {
    try {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        resolve(tabs?.[0]?.id ?? null);
      });
    } catch {
      resolve(null);
    }
  });
}

/**
 * Run the browser agent loop for a given task.
 *
 * The loop is reactive: snapshot DOM → ask backend for next action →
 * execute on page → repeat. Bounded by MAX_STEPS. Aborts immediately
 * when `signal` is aborted, including any in-flight fetch.
 */
export async function runAgentLoop({
  task,
  model,
  signal,
  onStep,
  onComplete,
  powerMode = false,
}: RunAgentLoopArgs): Promise<void> {
  const history: AgentAction[] = [];
  let stepCount = 0;

  // Power Mode: attach chrome.debugger once at loop start. If attach fails,
  // silently fall back to content-script mode — no step emitted, the loop
  // just uses the slower path. The user only sees the fallback if they
  // notice the missing yellow banner.
  let cdpTabId: number | null = null;
  if (powerMode) {
    const tabId = await getActiveTabId();
    if (tabId != null) {
      const attached = await attachCDP(tabId);
      if (attached.ok) {
        cdpTabId = tabId;
      } else {
        console.warn('[Aura/agent] Power Mode attach failed, using fallback:', attached.error);
      }
    }
  }

  const finish = async (reason: 'done' | 'max-steps' | 'stopped' | 'error', errorMessage?: string) => {
    if (cdpTabId != null) {
      await detachCDP(cdpTabId).catch(() => {});
      cdpTabId = null;
    }
    onComplete(reason, errorMessage);
  };

  try {
    while (stepCount < MAX_STEPS) {
      if (signal.aborted) {
        await finish('stopped');
        return;
      }
      stepCount++;

      let dom = (await sendMsg({ type: 'AGENT_DOM' })) as DomSnapshot | null;
      if (!dom?.ok || !dom.dom) {
        onStep({
          stepNum: stepCount,
          action: 'done',
          description: 'Could not read page DOM',
          result: 'error',
          error: 'AGENT_DOM failed',
          timestamp: Date.now(),
        });
        await finish('error', 'Could not read page DOM');
        return;
      }

      const domStr = dom.dom
        .map((e) => `[${e.index}] ${e.type} "${e.text}" -> ${e.selector}`)
        .join('\n');

      // In Power Mode, grab a screenshot to give the model vision context.
      // Drop oversized images silently rather than crashing the step.
      let screenshot: string | undefined;
      if (cdpTabId != null) {
        const shot = await cdpScreenshot(cdpTabId);
        if (shot.ok && shot.data && shot.data.length <= MAX_SCREENSHOT_BASE64_LEN) {
          screenshot = shot.data;
        } else if (shot.ok && shot.data && shot.data.length > MAX_SCREENSHOT_BASE64_LEN) {
          console.warn('[Aura/agent] dropping oversized screenshot', shot.data.length);
        }
      }

      const prompt =
        `Task: "${task}"\nURL: ${dom.url}\nTitle: ${dom.title}\n` +
        `History: ${JSON.stringify(history.slice(-5))}\n\n` +
        `Interactive elements on page:\n${domStr.slice(0, 3000)}\n\n` +
        (screenshot ? `A current screenshot of the viewport is attached. Use it to disambiguate elements and verify progress.\n\n` : '') +
        `Respond ONLY with valid JSON (no markdown, no explanation):\n` +
        `{"action":"click"|"type"|"scroll"|"navigate"|"done","selector":"","text":"","url":"","amount":300,"description":""}`;

      // Re-check abort right before the fetch — the stop button may have fired
      // while we were waiting on AGENT_DOM or the screenshot.
      if (signal.aborted) {
        await finish('stopped');
        return;
      }

      const r = await fetch(`${HTTP}/api/agent/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ prompt, model, screenshot }),
        signal,
      });

      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      const action = (await r.json()) as AgentAction;

      if (action.action === 'done') {
        onStep({
          stepNum: stepCount,
          action: 'done',
          description: action.description || 'Task complete',
          result: 'ok',
          timestamp: Date.now(),
        });
        await finish('done');
        return;
      }

      if (action.action === 'navigate' && action.url) {
        const navRes = await sendMsg({ type: 'AGENT_NAV', url: action.url });
        onStep({
          stepNum: stepCount,
          action: 'navigate',
          selector: action.url,
          description: action.description || `Navigate to ${action.url}`,
          result: navRes?.ok ? 'ok' : 'error',
          error: navRes?.ok ? undefined : String(navRes?.error || 'nav failed'),
          timestamp: Date.now(),
        });
        await sleep(POST_NAV_DELAY_MS);
        // Validate the page actually changed — if not, log a warning step and continue.
        const after = (await sendMsg({ type: 'AGENT_DOM' })) as DomSnapshot | null;
        if (after?.ok && after.url === dom.url && after.title === dom.title) {
          onStep({
            stepNum: stepCount,
            action: 'navigate',
            description: 'Warning: page did not change after navigate',
            result: 'error',
            error: 'no navigation detected',
            timestamp: Date.now(),
          });
        }
      } else {
        let result: { ok?: boolean; error?: string } | null = null;

        // Power Mode: use CDP primitives instead of content-script EXEC_ACTION
        // when we have a selector to resolve.
        if (cdpTabId != null && action.selector) {
          const resolved = await cdpResolveSelector(cdpTabId, action.selector);
          if (resolved.ok && resolved.data) {
            const { x, y } = resolved.data;
            if (action.action === 'click') {
              result = await cdpClick(cdpTabId, x, y);
            } else if (action.action === 'type' && action.text) {
              // Focus by clicking first, then type.
              await cdpClick(cdpTabId, x, y);
              result = await cdpType(cdpTabId, action.text);
            } else if (action.action === 'scroll') {
              result = await cdpScroll(cdpTabId, action.amount ?? 300);
            }
          }
        }

        // Fallback to content-script execution if Power Mode didn't apply
        // or the CDP attempt failed — must check .ok, not just truthiness,
        // because a failed CDP result is a truthy `{ok: false, error}` object.
        if (!result?.ok) {
          result = await sendMsg({ type: 'AGENT_EXEC', action });
        }

        // Selector-stale retry: refresh DOM once and try again via content script.
        if (!result?.ok) {
          const refreshed = (await sendMsg({ type: 'AGENT_DOM' })) as DomSnapshot | null;
          if (refreshed?.ok) {
            dom = refreshed;
            result = await sendMsg({ type: 'AGENT_EXEC', action });
          }
        }

        onStep({
          stepNum: stepCount,
          action: action.action,
          selector: action.selector,
          description: action.description || `${action.action} ${action.selector || ''}`.trim(),
          result: result?.ok ? 'ok' : 'error',
          error: result?.ok ? undefined : String(result?.error || 'action failed'),
          timestamp: Date.now(),
        });
        // CDP actions land instantly; content-script EXEC_ACTION fires DOM
        // mutation events that need a paint frame to settle.
        await sleep(cdpTabId != null ? POST_ACTION_DELAY_CDP_MS : POST_ACTION_DELAY_MS);
      }

      history.push(action);
    }

    await finish('max-steps');
  } catch (err: unknown) {
    if (signal.aborted || (err as { name?: string })?.name === 'AbortError') {
      await finish('stopped');
      return;
    }
    const message = err instanceof Error ? err.message : String(err);
    onStep({
      stepNum: stepCount,
      action: 'done',
      description: 'Agent loop error',
      result: 'error',
      error: message,
      timestamp: Date.now(),
    });
    await finish('error', message);
  }
}
