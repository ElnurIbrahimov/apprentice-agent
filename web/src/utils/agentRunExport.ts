/**
 * Export an agent run (plan + executor transcript + tool trace) as a single
 * Markdown document for sharing, review, or archive.
 */

import type { Message } from '../types';
import type { AgentPlan } from './agentPlan';

export interface RunExportInput {
  goal: string;
  plan: AgentPlan | null;
  messages: Message[];
  elapsedSeconds: number;
  tokensEstimated: number;
  plannerModel: string | null;
  executorModel: string | null;
  completedAt: number;
}

export function buildRunMarkdown(input: RunExportInput): string {
  const lines: string[] = [];
  const { goal, plan, messages, elapsedSeconds, tokensEstimated, plannerModel, executorModel, completedAt } = input;

  lines.push(`# Agent run — ${goal || '(no goal)'}`);
  lines.push('');
  lines.push(`*Completed ${new Date(completedAt).toLocaleString()}*`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- **Goal:** ${goal}`);
  lines.push(`- **Elapsed:** ${formatElapsed(elapsedSeconds)}`);
  lines.push(`- **Tokens (est.):** ${tokensEstimated.toLocaleString()}`);
  lines.push(`- **Planner model:** ${plannerModel ?? 'Auto'}`);
  lines.push(`- **Executor model:** ${executorModel ?? 'Auto'}`);
  lines.push('');

  if (plan && plan.steps.length) {
    lines.push('## Plan');
    lines.push('');
    for (const [i, step] of plan.steps.entries()) {
      const status = step.status ?? 'pending';
      lines.push(`### ${i + 1}. ${step.title} — \`${status}\``);
      if (step.rationale) lines.push(`> ${step.rationale}`);
      if (step.tool) lines.push(`- **Tool:** \`${step.tool}\``);
      if (step.inputs && Object.keys(step.inputs).length) {
        lines.push('- **Inputs:**');
        lines.push('  ```json');
        lines.push('  ' + JSON.stringify(step.inputs, null, 2).split('\n').join('\n  '));
        lines.push('  ```');
      }
      lines.push('');
    }
  }

  if (messages.length) {
    lines.push('## Transcript');
    lines.push('');
    for (const msg of messages) {
      if (msg.role === 'user') {
        lines.push('**User:**');
        lines.push('');
        lines.push(blockquote(msg.content));
        lines.push('');
      } else if (msg.role === 'assistant') {
        lines.push('**Assistant:**');
        lines.push('');
        lines.push(msg.content);
        lines.push('');
        if (msg.toolTrace && msg.toolTrace.length) {
          lines.push('**Tool trace:**');
          lines.push('');
          for (const t of msg.toolTrace) {
            const when = new Date(t.timestamp).toLocaleTimeString();
            const elapsed = typeof t.elapsed_ms === 'number' ? ` (${(t.elapsed_ms / 1000).toFixed(1)}s)` : '';
            lines.push(`- \`${t.tool}\` — ${t.event}${elapsed} @ ${when}`);
            if (t.detail) {
              lines.push('  ```');
              lines.push('  ' + t.detail.split('\n').join('\n  '));
              lines.push('  ```');
            }
          }
          lines.push('');
        }
        if (msg.citations && msg.citations.length) {
          lines.push('**Sources:**');
          lines.push('');
          for (const c of msg.citations) {
            lines.push(`- [${c.title || c.url}](${c.url})${c.snippet ? ` — ${c.snippet}` : ''}`);
          }
          lines.push('');
        }
      }
    }
  }

  lines.push('---');
  lines.push('*Exported from Aura.*');
  return lines.join('\n');
}

function blockquote(text: string): string {
  return text.split('\n').map((l) => `> ${l}`).join('\n');
}

function formatElapsed(total: number): string {
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}m ${s.toString().padStart(2, '0')}s`;
}

/** Trigger a browser download of the markdown. */
export function downloadRunMarkdown(input: RunExportInput): void {
  const md = buildRunMarkdown(input);
  const filename = `agent-run-${sanitizeName(input.goal)}-${Date.now()}.md`;
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 250);
}

function sanitizeName(s: string): string {
  return s.replace(/[^a-z0-9-_]+/gi, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'untitled';
}
