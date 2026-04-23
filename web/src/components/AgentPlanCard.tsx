/**
 * Editable plan card used in the Plan phase of AgentPanel.
 *
 * Renders an agent plan (produced by the LLM in planning mode) as an
 * ordered list of step cards. Each step's title / rationale / tool hint
 * is editable; steps can be reordered, deleted, or added. A final
 * Approve button hands the plan off to the executor.
 */

import { useState } from 'react';
import {
  ArrowUpIcon, ArrowDownIcon, TrashIcon, PlusIcon,
  CheckCircleIcon, PencilIcon, XMarkIcon,
} from '@heroicons/react/24/outline';
import type { AgentPlan, PlanStep } from '../utils/agentPlan';
import { addStep, moveStep, removeStep, updateStep } from '../utils/agentPlan';

interface Props {
  plan: AgentPlan;
  onChange: (next: AgentPlan) => void;
  onApprove: () => void;
  onDiscard: () => void;
  disabled?: boolean;
}

export function AgentPlanCard({ plan, onChange, onApprove, onDiscard, disabled }: Props) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-chat-text-secondary">Proposed plan</div>
          <div className="text-sm text-chat-text mt-0.5 font-medium">{plan.goal}</div>
        </div>
        <button
          type="button"
          onClick={onDiscard}
          disabled={disabled}
          className="text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
        >
          Discard
        </button>
      </div>

      <ol className="flex flex-col gap-2">
        {plan.steps.map((step, i) => (
          <StepRow
            key={step.id}
            step={step}
            index={i}
            total={plan.steps.length}
            disabled={disabled}
            onEdit={(patch) => onChange(updateStep(plan, step.id, patch))}
            onUp={() => onChange(moveStep(plan, step.id, -1))}
            onDown={() => onChange(moveStep(plan, step.id, +1))}
            onRemove={() => onChange(removeStep(plan, step.id))}
          />
        ))}
      </ol>

      <div className="flex items-center gap-2 mt-1">
        <button
          type="button"
          onClick={() => onChange(addStep(plan))}
          disabled={disabled}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border text-chat-text-secondary hover:text-chat-text transition-colors"
          style={{ borderColor: 'var(--border-default)' }}
        >
          <PlusIcon className="w-3.5 h-3.5" />
          Add step
        </button>

        <div className="flex-1" />

        <button
          type="button"
          onClick={onApprove}
          disabled={disabled || plan.steps.length === 0}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            background: 'var(--chat-accent)',
            boxShadow: '0 4px 12px rgba(124,58,237,0.35)',
          }}
        >
          <CheckCircleIcon className="w-4 h-4" />
          Approve &amp; execute
        </button>
      </div>
    </div>
  );
}

interface StepRowProps {
  step: PlanStep;
  index: number;
  total: number;
  disabled?: boolean;
  onEdit: (patch: Partial<PlanStep>) => void;
  onUp: () => void;
  onDown: () => void;
  onRemove: () => void;
}

function StepRow({ step, index, total, disabled, onEdit, onUp, onDown, onRemove }: StepRowProps) {
  const [editing, setEditing] = useState(false);
  const canUp = index > 0;
  const canDown = index < total - 1;

  return (
    <li
      className="rounded-xl p-3 flex gap-3"
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
      }}
    >
      <div
        className="flex-shrink-0 flex items-center justify-center rounded-full text-xs font-bold"
        style={{
          width: 26, height: 26,
          background: 'rgba(124,58,237,0.18)',
          color: '#c4b5fd',
        }}
      >
        {index + 1}
      </div>

      <div className="flex-1 min-w-0">
        {editing ? (
          <div className="flex flex-col gap-1.5">
            <input
              value={step.title}
              onChange={(e) => onEdit({ title: e.target.value })}
              placeholder="Step title"
              className="w-full bg-transparent border rounded px-2 py-1 text-sm text-chat-text outline-none focus:border-chat-accent"
              style={{ borderColor: 'var(--border-default)' }}
              autoFocus
            />
            <textarea
              value={step.rationale}
              onChange={(e) => onEdit({ rationale: e.target.value })}
              placeholder="Why this step?"
              rows={2}
              className="w-full bg-transparent border rounded px-2 py-1 text-xs text-chat-text-secondary outline-none focus:border-chat-accent resize-none"
              style={{ borderColor: 'var(--border-default)' }}
            />
            <input
              value={step.tool ?? ''}
              onChange={(e) => onEdit({ tool: e.target.value || undefined })}
              placeholder="Tool hint (optional) — e.g. web_search"
              className="w-full bg-transparent border rounded px-2 py-1 text-xs text-chat-text-secondary outline-none focus:border-chat-accent font-mono"
              style={{ borderColor: 'var(--border-default)' }}
            />
            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => setEditing(false)}
                className="text-xs text-chat-accent hover:underline"
              >
                Done editing
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-chat-text">{step.title}</span>
              {step.tool && (
                <span
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                  style={{ background: 'rgba(59,130,246,0.18)', color: '#93c5fd' }}
                >
                  {step.tool}
                </span>
              )}
              <StatusPill status={step.status ?? 'pending'} />
            </div>
            {step.rationale && (
              <div className="text-xs text-chat-text-secondary mt-1 leading-relaxed">{step.rationale}</div>
            )}
          </>
        )}
      </div>

      {!editing && !disabled && (
        <div className="flex-shrink-0 flex items-start gap-0.5 opacity-60 group-hover:opacity-100 transition-opacity">
          <IconButton title="Edit" onClick={() => setEditing(true)}>
            <PencilIcon className="w-3.5 h-3.5" />
          </IconButton>
          <IconButton title="Move up" onClick={onUp} disabled={!canUp}>
            <ArrowUpIcon className="w-3.5 h-3.5" />
          </IconButton>
          <IconButton title="Move down" onClick={onDown} disabled={!canDown}>
            <ArrowDownIcon className="w-3.5 h-3.5" />
          </IconButton>
          <IconButton title="Delete" onClick={onRemove}>
            <TrashIcon className="w-3.5 h-3.5" />
          </IconButton>
        </div>
      )}
      {editing && (
        <IconButton title="Cancel" onClick={() => setEditing(false)}>
          <XMarkIcon className="w-3.5 h-3.5" />
        </IconButton>
      )}
    </li>
  );
}

function StatusPill({ status }: { status: NonNullable<PlanStep['status']> }) {
  const styles: Record<string, { bg: string; color: string; label: string }> = {
    pending:  { bg: 'rgba(255,255,255,0.06)', color: '#9ca3af', label: 'pending' },
    running:  { bg: 'rgba(245,158,11,0.18)', color: '#fbbf24', label: 'running' },
    done:     { bg: 'rgba(34,197,94,0.18)', color: '#86efac', label: 'done' },
    failed:   { bg: 'rgba(239,68,68,0.18)', color: '#fca5a5', label: 'failed' },
    skipped:  { bg: 'rgba(148,163,184,0.18)', color: '#cbd5e1', label: 'skipped' },
  };
  const s = styles[status] ?? styles.pending;
  return (
    <span
      className="text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded"
      style={{ background: s.bg, color: s.color }}
    >
      {s.label}
    </span>
  );
}

function IconButton({
  children, onClick, disabled, title,
}: { children: React.ReactNode; onClick: () => void; disabled?: boolean; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="p-1 rounded text-chat-text-secondary hover:text-chat-text hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
    >
      {children}
    </button>
  );
}
