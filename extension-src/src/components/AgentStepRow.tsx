import React from 'react';
import { MousePointerClick, Keyboard, MoveVertical, ExternalLink, Check, Circle, type LucideIcon } from 'lucide-react';
import type { AgentStep } from '../types';

const ICONS: Record<AgentStep['action'], LucideIcon> = {
  click: MousePointerClick,
  type: Keyboard,
  scroll: MoveVertical,
  navigate: ExternalLink,
  done: Check,
};

interface Props {
  step: AgentStep;
}

export default function AgentStepRow({ step }: Props) {
  const Icon: LucideIcon = ICONS[step.action] ?? Circle;
  const isError = step.result === 'error';
  const color = isError ? 'var(--rd)' : 'var(--pl)';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '6px 10px',
        borderRadius: 8,
        background: isError ? 'rgba(239,68,68,0.08)' : 'var(--s2)',
        border: `1px solid ${isError ? 'rgba(239,68,68,0.3)' : 'var(--b1)'}`,
        fontSize: 11.5,
        lineHeight: 1.4,
      }}
    >
      <span
        style={{
          display: 'inline-flex',
          width: 18,
          height: 18,
          alignItems: 'center',
          justifyContent: 'center',
          color,
          flexShrink: 0,
        }}
      >
        <Icon size={13} />
      </span>
      <span style={{ color: 'var(--mu)', fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>
        {step.stepNum}
      </span>
      <span
        style={{
          flex: 1,
          minWidth: 0,
          color: 'var(--tx)',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={step.description}
      >
        {step.description}
      </span>
      {isError && step.error && (
        <span
          style={{
            color: 'var(--rd)',
            fontSize: 10.5,
            flexShrink: 0,
            maxWidth: 120,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={step.error}
        >
          {step.error}
        </span>
      )}
    </div>
  );
}
