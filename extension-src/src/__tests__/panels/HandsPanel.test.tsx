import React from 'react';
import { render, act } from '@testing-library/react';
import { useStore } from '../../store';
import HandsPanel from '../../panels/HandsPanel';

let mockState: any = null;
let capturedCallback: ((prev: any) => void) | null = null;

jest.mock('../../store', () => {
  const actual = jest.requireActual('../../store');
  const stub: any = (selector: any) => {
    return selector(mockState);
  };
  stub._state = null;
  stub.getState = () => mockState || actual.useStore.getState();
  stub.setState = actual.useStore.setState;
  stub.subscribe = (cb: (prev: any) => void) => {
    capturedCallback = cb;
    return () => { capturedCallback = null; };
  };
  stub.mockImplementation = (fn: any) => {
    mockState = fn(actual.useStore.getState());
    return stub;
  };
  stub.mockReturnValue = (state: any) => {
    mockState = state;
    return stub;
  };
  stub.mockClear = () => { mockState = null; };
  return { ...actual, useStore: stub };
});

function makeMockState(overrides = {}) {
  return {
    hands: [] as any[],
    handApprovals: [] as any[],
    handHistory: [] as any[],
    handTemplates: [] as any[],
    handLiveTrace: [] as any[],
    handsError: null as string | null,
    handsLoaded: true as boolean,
    handsPollingActive: false as boolean,
    handsPollingInterval: 15000 as number,
    handsLastLoaded: null as number | null,
    wsReady: true,
    backendStatus: 'online' as 'online' | 'offline' | 'connecting',
    loadHands: jest.fn().mockResolvedValue(undefined),
    loadHandApprovals: jest.fn().mockResolvedValue(undefined),
    loadHandHistory: jest.fn().mockResolvedValue(undefined),
    loadHandTemplates: jest.fn().mockResolvedValue(undefined),
    setHandsPollingActive: jest.fn(),
    setHandsLastLoaded: jest.fn(),
    setHandsPollingInterval: jest.fn(),
    history: [] as any[],
    ...overrides,
  };
}

describe('HandsPanel polling lifecycle', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockState = null;
    capturedCallback = null;
    (useStore as unknown as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
    mockState = null;
    capturedCallback = null;
  });

  test('loads hands, approvals, history, and templates exactly once on mount', () => {
    const loadHands = jest.fn().mockResolvedValue(undefined);
    const loadHandApprovals = jest.fn().mockResolvedValue(undefined);
    const loadHandHistory = jest.fn().mockResolvedValue(undefined);
    const loadHandTemplates = jest.fn().mockResolvedValue(undefined);

    (useStore as unknown as jest.Mock).mockReturnValue(makeMockState({
      loadHands,
      loadHandApprovals,
      loadHandHistory,
      loadHandTemplates,
    }));

    render(<HandsPanel />);

    expect(loadHands).toHaveBeenCalledTimes(1);
    expect(loadHandApprovals).toHaveBeenCalledTimes(1);
    expect(loadHandHistory).toHaveBeenCalledTimes(1);
    expect(loadHandTemplates).toHaveBeenCalledTimes(1);
  });

  test('sets polling active and configures interval on mount', () => {
    const setHandsPollingActive = jest.fn();
    const setHandsPollingInterval = jest.fn();

    (useStore as unknown as jest.Mock).mockReturnValue(makeMockState({
      setHandsPollingActive,
      setHandsPollingInterval,
    }));

    render(<HandsPanel />);

    expect(setHandsPollingActive).toHaveBeenCalledWith(true);
    expect(setHandsPollingInterval).toHaveBeenCalledWith(15000);
  });

  test('updates interval to 60s when ws disconnects', () => {
    const setHandsPollingInterval = jest.fn();
    const loadHands = jest.fn().mockResolvedValue(undefined);
    const loadHandApprovals = jest.fn().mockResolvedValue(undefined);
    const loadHandHistory = jest.fn().mockResolvedValue(undefined);
    const loadHandTemplates = jest.fn().mockResolvedValue(undefined);

    mockState = makeMockState({
      setHandsPollingInterval,
      loadHands,
      loadHandApprovals,
      loadHandHistory,
      loadHandTemplates,
      wsReady: true,
      backendStatus: 'online' as const,
    });

    render(<HandsPanel />);

    setHandsPollingInterval.mockClear();

    const previousState = { ...mockState };

    mockState = makeMockState({
      setHandsPollingInterval,
      loadHands,
      loadHandApprovals,
      loadHandHistory,
      loadHandTemplates,
      wsReady: false,
      backendStatus: 'online' as const,
    });

    act(() => {
      capturedCallback?.(previousState);
    });

    expect(setHandsPollingInterval).toHaveBeenCalledWith(60000);
  });
});