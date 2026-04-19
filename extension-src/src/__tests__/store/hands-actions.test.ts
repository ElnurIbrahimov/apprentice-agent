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

function makeState() {
  const state: any = {
    hands: [],
    handApprovals: [],
    handHistory: [],
    handTemplates: [],
    handLiveTrace: [],
    handsLoaded: false,
    handsError: null,
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
    const { state, set, get } = makeState();
    mockApiFetch.mockRejectedValueOnce(new Error('boom'));
    const actions = createHandsActions(set, get);
    await actions.loadHands();
    expect(state.handsError).toBe('boom');
    expect(state.handsLoaded).toBe(true);
  });

  test('runHand posts to /run and refreshes list', async () => {
    const { set, get } = makeState();
    mockApiFetch.mockResolvedValueOnce({}).mockResolvedValueOnce({ hands: [] });
    const actions = createHandsActions(set, get);
    // loadHands is pulled off get() so we attach actions to the same state
    Object.assign(get(), actions);
    await actions.runHand('demo hand');
    expect(mockApiFetch).toHaveBeenNthCalledWith(1, 'http://mock/api/hands/demo%20hand/run', { method: 'POST' });
    expect(mockApiFetch).toHaveBeenNthCalledWith(2, 'http://mock/api/hands');
  });

  test('deleteHand removes the hand optimistically without waiting for reload', async () => {
    const { state, set, get } = makeState();
    state.hands = [{ name: 'h1' }, { name: 'h2' }];
    mockApiFetch.mockResolvedValueOnce({});
    const actions = createHandsActions(set, get);
    await actions.deleteHand('h1');
    expect(state.hands).toEqual([{ name: 'h2' }]);
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

  test('approveHand posts boolean and drops that hand from pending approvals', async () => {
    const { state, set, get } = makeState();
    state.handApprovals = [{ request_id: 'r1', hand_name: 'h1' }, { request_id: 'r2', hand_name: 'h2' }];
    mockApiFetch.mockResolvedValueOnce({});
    const actions = createHandsActions(set, get);
    await actions.approveHand('h1', 'r1', true);
    const call = mockApiFetch.mock.calls[0];
    expect(call[0]).toBe('http://mock/api/hands/h1/approve');
    expect(JSON.parse((call[1] as any).body)).toEqual({ approved: true });
    expect(state.handApprovals.map((a: any) => a.hand_name)).toEqual(['h2']);
  });
});
