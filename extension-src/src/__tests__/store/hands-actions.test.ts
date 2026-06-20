/**
 * Tests for the Hands action factory. Mocks apiFetch; verifies endpoint
 * selection, state transitions, and the optimistic event/approval/trace apply
 * paths.
 */

jest.mock('../../api', () => ({
  HTTP: 'http://mock',
  apiFetch: jest.fn(),
  API_KEY: 'test-key',
}));

import { createHandsActions } from '../../store/hands-actions';
import { apiFetch } from '../../api';

const mockApiFetch = apiFetch as jest.MockedFunction<typeof apiFetch>;

function makeState(overrides: any = {}) {
  const state: any = {
    hands: [],
    handApprovals: [],
    handHistory: [],
    handTemplates: [],
    handLiveTrace: [],
    handsLoaded: false,
    handsError: null,
    ...overrides,
  };
  const set = (partial: any) => {
    const patch = typeof partial === 'function' ? partial(state) : partial;
    Object.assign(state, patch);
  };
  const get = () => state;
  return { state, set, get };
}

beforeEach(() => {
  mockApiFetch.mockReset();
});

describe('hands-actions', () => {
  test('loadHands populates list and marks loaded', async () => {
    const { state, set, get } = makeState();
    mockApiFetch.mockResolvedValueOnce({ hands: [{ name: 'h1', total_runs: 0 }] });
    const actions = createHandsActions(set, get);
    await actions.loadHands();
    expect(mockApiFetch).toHaveBeenCalledWith('http://mock/api/hands');
    expect(state.hands).toHaveLength(1);
    expect(state.handsLoaded).toBe(true);
    expect(state.handsError).toBeNull();
  });

test('loadHands records error when apiFetch throws', async () => {
    (apiFetch as jest.Mock).mockRejectedValue(new Error('boom'));
    let state: any = {};
    const set = jest.fn((u) => { state = typeof u === 'function' ? u(state) : { ...state, ...u }; });
    const get = jest.fn(() => ({ loadHands: jest.fn() }));
    const actions = createHandsActions(set, get);
    await actions.loadHands();
    expect(state.handsLoaded).toBe(true);
    expect(state.handsError).toBe('Failed to load hands after retries');
  });

  test('runHand posts to /run and refreshes list', async () => {
    const { set, get } = makeState();
    mockApiFetch.mockResolvedValueOnce({}).mockResolvedValueOnce({ hands: [] });
    const actions = createHandsActions(set, get);
    Object.assign(get(), actions);
    await actions.runHand('demo hand');
    expect(mockApiFetch).toHaveBeenNthCalledWith(1, 'http://mock/api/hands/demo%20hand/run', { method: 'POST' });
    expect(mockApiFetch).toHaveBeenNthCalledWith(2, 'http://mock/api/hands');
  });

  test('deleteHand posts to delete endpoint and reloads', async () => {
    const { state, set, get } = makeState();
    state.hands = [{ name: 'h1' }, { name: 'h2' }];
    mockApiFetch.mockResolvedValueOnce({});
    const loadHands = jest.fn().mockResolvedValue(undefined);
    const getMock = jest.fn(() => ({ loadHands }));
    const actions = createHandsActions(set, getMock);
    await actions.deleteHand('h1');
    expect(mockApiFetch).toHaveBeenCalledWith('http://mock/api/hands/h1', { method: 'DELETE' });
    expect(loadHands).toHaveBeenCalled();
  });

  test('applyHandApprovalRequest de-duplicates by request_id', () => {
    const { state, set, get } = makeState();
    const actions = createHandsActions(set, get);
    const req = { request_id: '42', hand_name: 'h', tool_name: 't', args: {} };
    actions.applyHandApprovalRequest(req);
    actions.applyHandApprovalRequest(req);
    expect(state.handApprovals).toHaveLength(1);
  });

  test('applyHandActionTrace caps the live trace at the max window', () => {
    const { state, set, get } = makeState();
    const actions = createHandsActions(set, get);
    for (let i = 0; i < 30; i++) {
      actions.applyHandActionTrace({ hand: 'h', step: i, description: 's', timestamp: i });
    }
    expect(state.handLiveTrace).toHaveLength(20);
    expect(state.handLiveTrace[0].step).toBe(10);
    expect(state.handLiveTrace[19].step).toBe(29);
  });

  test('applyHandEvent updates stats and prepends a history entry', () => {
    const { state, set, get } = makeState();
    state.hands = [{ name: 'h', total_runs: 2, last_run_ts: 0, state: 'idle' }];
    const actions = createHandsActions(set, get);
    actions.applyHandEvent({ hand: 'h', success: true, summary: 'ok', duration_seconds: 1.5 });
    expect(state.hands[0].total_runs).toBe(3);
    expect(state.hands[0].state).toBe('cooldown');
    expect(state.handHistory[0].action_data.duration_ms).toBe(1500);
  });

  test('approveHand posts boolean and drops the specific approval by request_id', async () => {
    const { state, set, get } = makeState();
    state.handApprovals = [
      { request_id: 'r1', hand_name: 'h1' },
      { request_id: 'r2', hand_name: 'h1' },
    ];
    mockApiFetch.mockResolvedValueOnce({});
    const actions = createHandsActions(set, get);
    await actions.approveHand('h1', 'r1', true);
    const call = mockApiFetch.mock.calls[0];
    expect(call[0]).toBe('http://mock/api/hands/h1/approve');
    expect(JSON.parse((call[1] as any).body)).toEqual({ approved: true, request_id: 'r1' });
    expect(state.handApprovals.map((a: any) => a.request_id)).toEqual(['r2']);
  });
});

describe('hands-actions regression tests', () => {
  beforeEach(() => {
    mockApiFetch.mockReset();
  });

  test('approveHand removes the specific approval by request_id, not hand_name', async () => {
    const { state, set, get } = makeState({
      handApprovals: [
        { request_id: 'req-1', hand_name: 'hand-a', tool_name: 'tool1', args: {}, timestamp: Date.now(), age_seconds: 10 },
        { request_id: 'req-2', hand_name: 'hand-a', tool_name: 'tool2', args: {}, timestamp: Date.now(), age_seconds: 20 },
      ],
    });
    mockApiFetch.mockResolvedValueOnce({});
    const actions = createHandsActions(set, get);

    await actions.approveHand('hand-a', 'req-1', true);

    expect(state.handApprovals).toHaveLength(1);
    expect(state.handApprovals[0].request_id).toBe('req-2');
  });

  test('loadHands sets handsLoaded=true even when fetch fails', async () => {
    const { state, set, get } = makeState();
    mockApiFetch.mockRejectedValue(null);
    const actions = createHandsActions(set, get);

    await actions.loadHands();

    expect(state.handsLoaded).toBe(true);
    expect(state.handsError).toBe('Failed to load hands after retries');
  });

test('deleteHand calls loadHands after successful delete', async () => {
    (apiFetch as jest.Mock).mockResolvedValue(undefined);
    const loadHands = jest.fn().mockResolvedValue(undefined);
    let state: any = { hands: [{ name: 'hand-to-delete', state: 'idle', total_runs: 0, last_run_ts: 0 }] };
    const set = jest.fn((u) => { state = typeof u === 'function' ? u(state) : { ...state, ...u }; });
    const get = jest.fn(() => ({ loadHands }));
    const actions = createHandsActions(set, get);
    await actions.deleteHand('hand-to-delete');
    expect(loadHands).toHaveBeenCalled();
  });
});