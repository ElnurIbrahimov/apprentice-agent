import React from 'react';
import { render, screen } from '@testing-library/react';
import ApprovalCard from '../../../components/hands/ApprovalCard';

describe('ApprovalCard age rendering', () => {
  test('shows "just now" for fresh approvals', () => {
    render(
      <ApprovalCard
        approval={{
          request_id: 'req-1',
          hand_name: 'my-hand',
          tool_name: 'browser_tool',
          args: {},
          timestamp: Date.now(),
          age_seconds: 10,
        }}
        onApprove={jest.fn()}
        onDeny={jest.fn()}
      />
    );
    expect(screen.getByText(/just now/)).toBeTruthy();
  });

  test('shows minutes for approvals older than 60s', () => {
    render(
      <ApprovalCard
        approval={{
          request_id: 'req-1',
          hand_name: 'my-hand',
          tool_name: 'browser_tool',
          args: {},
          timestamp: Date.now(),
          age_seconds: 300,
        }}
        onApprove={jest.fn()}
        onDeny={jest.fn()}
      />
    );
    expect(screen.getByText(/5m ago/)).toBeTruthy();
  });

  test('shows hours for approvals older than 3600s', () => {
    render(
      <ApprovalCard
        approval={{
          request_id: 'req-1',
          hand_name: 'my-hand',
          tool_name: 'browser_tool',
          args: {},
          timestamp: Date.now(),
          age_seconds: 7200,
        }}
        onApprove={jest.fn()}
        onDeny={jest.fn()}
      />
    );
    expect(screen.getByText(/2h ago/)).toBeTruthy();
  });

  test('shows days for approvals older than 86400s', () => {
    render(
      <ApprovalCard
        approval={{
          request_id: 'req-1',
          hand_name: 'my-hand',
          tool_name: 'browser_tool',
          args: {},
          timestamp: Date.now(),
          age_seconds: 172800,
        }}
        onApprove={jest.fn()}
        onDeny={jest.fn()}
      />
    );
    expect(screen.getByText(/2d ago/)).toBeTruthy();
  });
});