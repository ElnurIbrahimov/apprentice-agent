import React from 'react';
import { render, act } from '@testing-library/react';
import { useStore } from '../../store';
import HandsPanel from '../../panels/HandsPanel';

jest.mock('../../store', () => {
  const actual = jest.requireActual('../../store');
  return {
    ...actual,
    useStore: Object.assign(jest.fn(), {
      getState: actual.useStore.getState,
      setState: actual.useStore.setState,
      subscribe: actual.useStore.subscribe,
    }),
  };
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
    runHand: jest.fn().mockResolvedValue(undefined),
    pauseHand: jest.fn().mockResolvedValue(undefined),
    activateHand: jest.fn().mockResolvedValue(undefined),
    deactivateHand: jest.fn().mockResolvedValue(undefined),
    deleteHand: jest.fn().mockResolvedValue(undefined),
    approveHand: jest.fn().mockResolvedValue(undefined),
    createHand: jest.fn().mockResolvedValue(undefined),
    createHandFromTemplate: jest.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

describe('HandsPanel polling lifecycle', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    (useStore as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('loads hands immediately on mount', async () => {
    const loadHands = jest.fn().mockResolvedValue(undefined);
    const loadHandApprovals = jest.fn().mockResolvedValue(undefined);
    const loadHandHistory = jest.fn().mockResolvedValue(undefined);
    const setHandsPollingActive = jest.fn();
    const setHandsPollingInterval = jest.fn();

    (useStore as jest.Mock).mockImplementation((selector) => {
      return selector(makeMockState({
        loadHands,
        loadHandApprovals,
        loadHandHistory,
        setHandsPollingActive,
        setHandsPollingInterval,
      }));
    });

    render(<HandsPanel />);

    await act(async () => {
      jest.runAllTimers();
    });

    expect(loadHands).toHaveBeenCalled();
  });

  test('sets polling active on mount', async () => {
    const setHandsPollingActive = jest.fn();

    (useStore as jest.Mock).mockImplementation((selector) => {
      return selector(makeMockState({ setHandsPollingActive }));
    });

    render(<HandsPanel />);

    await act(async () => {
      jest.runAllTimers();
    });

    expect(setHandsPollingActive).toHaveBeenCalledWith(true);
  });
});

describe('HandsPanel polling lifecycle', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    (useStore as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('loads hands immediately on mount', async () => {
    let loadHands: jest.Mock;
    let loadHandApprovals: jest.Mock;
    let loadHandHistory: jest.Mock;
    let setHandsPollingActive: jest.Mock;
    let setHandsLastLoaded: jest.Mock;
    let setHandsPollingInterval: jest.Mock;

    (useStore as jest.Mock).mockImplementation((selector) => {
      loadHands = jest.fn().mockResolvedValue(undefined);
      loadHandApprovals = jest.fn().mockResolvedValue(undefined);
      loadHandHistory = jest.fn().mockResolvedValue(undefined);
      setHandsPollingActive = jest.fn();
      setHandsLastLoaded = jest.fn();
      setHandsPollingInterval = jest.fn();
      const state = {
        hands: [],
        handsError: null,
        handsLoaded: true,
        handsPollingActive: false,
        handsPollingInterval: 15000,
        handsLastLoaded: null,
        wsReady: true,
        backendStatus: 'online' as const,
        loadHands,
        loadHandApprovals,
        loadHandHistory,
        setHandsPollingActive,
        setHandsLastLoaded,
        setHandsPollingInterval,
        loadHandTemplates: jest.fn(),
        handsPollingActive: false,
      };
      return selector(state);
    });

    render(<HandsPanel />);

    await act(async () => {
      jest.runAllTimers();
    });

    expect(loadHands).toHaveBeenCalled();
  });

  test('sets polling active on mount', async () => {
    let setHandsPollingActive: jest.Mock;

    (useStore as jest.Mock).mockImplementation((selector) => {
      setHandsPollingActive = jest.fn();
      const state = {
        hands: [],
        handsError: null,
        handsLoaded: true,
        handsPollingActive: false,
        handsPollingInterval: 15000,
        handsLastLoaded: null,
        wsReady: true,
        backendStatus: 'online' as const,
        loadHands: jest.fn().mockResolvedValue(undefined),
        loadHandApprovals: jest.fn().mockResolvedValue(undefined),
        loadHandHistory: jest.fn().mockResolvedValue(undefined),
        setHandsPollingActive,
        setHandsLastLoaded: jest.fn(),
        setHandsPollingInterval: jest.fn(),
        loadHandTemplates: jest.fn(),
      };
      return selector(state);
    });

    render(<HandsPanel />);

    await act(async () => {
      jest.runAllTimers();
    });

    expect(setHandsPollingActive).toHaveBeenCalledWith(true);
  });
});