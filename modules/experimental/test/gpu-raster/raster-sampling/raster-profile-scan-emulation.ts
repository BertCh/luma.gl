// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU emulation of the reduction order of the segmented scans in `gpu-raster-profile-scan.ts`:
 * Hillis-Steele inside 256-item blocks, a chunked carry pass over block aggregates, and a final
 * `combine(carry, prefix)`. Every float32 operation is rounded with `Math.fround`, and the combine
 * functions mirror the WGSL line by line, so tests can bound the tree-order rounding against the
 * serial oracle without a GPU.
 */

const BLOCK_SIZE = 256;
const f = Math.fround;

/** Segmented profile state; mirrors the WGSL `ScanState` of the profile scan. */
export type ProfileState = {
  flag: number;
  hasValue: number;
  firstValue: number;
  lastValue: number;
  gain: number;
  loss: number;
  minimum: number;
  maximum: number;
};

/** Segmented distance state; mirrors the WGSL `ScanState` of the vertex-distance scan. */
export type DistanceState = {flag: number; sum: number};

export function getIdentityProfileState(): ProfileState {
  return {
    flag: 0,
    hasValue: 0,
    firstValue: 0,
    lastValue: 0,
    gain: 0,
    loss: 0,
    minimum: 0,
    maximum: 0
  };
}

export function combineProfileStates(left: ProfileState, right: ProfileState): ProfileState {
  let result = {...right};
  if (right.flag === 0) {
    if (left.hasValue === 0) {
      result = {...right};
    } else if (right.hasValue === 0) {
      result = {...left};
    } else {
      const delta = f(right.firstValue - left.lastValue);
      result.firstValue = left.firstValue;
      result.gain = f(f(left.gain + Math.max(delta, 0)) + right.gain);
      result.loss = f(f(left.loss + Math.max(-delta, 0)) + right.loss);
      result.minimum = Math.min(left.minimum, right.minimum);
      result.maximum = Math.max(left.maximum, right.maximum);
    }
  }
  result.flag = left.flag | right.flag;
  return result;
}

export function getIdentityDistanceState(): DistanceState {
  return {flag: 0, sum: 0};
}

export function combineDistanceStates(left: DistanceState, right: DistanceState): DistanceState {
  const result = {...right};
  if (right.flag === 0) {
    result.sum = f(left.sum + right.sum);
  }
  result.flag = left.flag | right.flag;
  return result;
}

/** Profile state of one sample value; `isStart` marks the first sample of a line. */
export function getProfileItemState(value: number, isStart: boolean): ProfileState {
  const state = getIdentityProfileState();
  state.flag = isStart ? 1 : 0;
  if (Number.isFinite(value)) {
    state.hasValue = 1;
    state.firstValue = state.lastValue = state.minimum = state.maximum = f(value);
  }
  return state;
}

/**
 * Inclusive segmented scan in the exact combine order of the GPU kernels. Returns the inclusive
 * state of every item and the block count.
 */
export function emulateSegmentedScan<State>(
  items: readonly State[],
  identity: () => State,
  combine: (left: State, right: State) => State
): {inclusive: State[]; blockCount: number} {
  const blockCount = Math.max(1, Math.ceil(items.length / BLOCK_SIZE));
  const local: State[][] = [];
  const aggregates: State[] = [];
  for (let block = 0; block < blockCount; block++) {
    // Hillis-Steele: every lane reads the pre-step value of its left neighbour.
    let scratch = Array.from({length: BLOCK_SIZE}, (_, lane) => {
      const index = block * BLOCK_SIZE + lane;
      return index < items.length ? items[index] : identity();
    });
    for (let span = 1; span < BLOCK_SIZE; span <<= 1) {
      const previous = scratch;
      scratch = previous.map((value, lane) =>
        lane >= span ? combine(previous[lane - span], value) : value
      );
    }
    local.push(scratch);
    aggregates.push(scratch[BLOCK_SIZE - 1]);
  }
  // Carry pass: one lane per chunk of consecutive blocks.
  const chunk = Math.ceil(blockCount / BLOCK_SIZE);
  const chunkTotals = Array.from({length: BLOCK_SIZE}, (_, lane) => {
    let total = identity();
    const begin = Math.min(lane * chunk, blockCount);
    const end = Math.min(begin + chunk, blockCount);
    for (let block = begin; block < end; block++) {
      total = combine(total, aggregates[block]);
    }
    return total;
  });
  let scratch = chunkTotals;
  for (let span = 1; span < BLOCK_SIZE; span <<= 1) {
    const previous = scratch;
    scratch = previous.map((value, lane) =>
      lane >= span ? combine(previous[lane - span], value) : value
    );
  }
  const carries: State[] = new Array(blockCount);
  for (let lane = 0; lane < BLOCK_SIZE; lane++) {
    let running = lane > 0 ? scratch[lane - 1] : identity();
    const begin = Math.min(lane * chunk, blockCount);
    const end = Math.min(begin + chunk, blockCount);
    for (let block = begin; block < end; block++) {
      carries[block] = running;
      running = combine(running, aggregates[block]);
    }
  }
  const inclusive: State[] = [];
  for (let index = 0; index < items.length; index++) {
    const block = Math.floor(index / BLOCK_SIZE);
    inclusive.push(combine(carries[block], local[block][index % BLOCK_SIZE]));
  }
  return {inclusive, blockCount};
}
