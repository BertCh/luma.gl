// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL for the parallel segmented scans of {@link GPURasterProfile}.
 *
 * Two scans run over a flat item array in which lines are segments: the vertex-distance scan
 * (a segmented float sum over vertices) and the profile scan (a segmented scan whose state carries
 * running gain, loss, minimum and maximum over the finite samples of a line). Both are
 * reduce-then-scan: a block kernel reduces 256 items per workgroup, one workgroup scans the block
 * aggregates, and a down-sweep kernel rescans each block with its carry-in. The state is a monoid
 * whose segment flag resets the running value at a line start, so segmenting needs no per-line
 * loop and the work is parallel over items.
 *
 * Summation order is a fixed tree (Hillis-Steele inside a block, a fixed chunk walk across
 * blocks), so results are deterministic. They differ from a left-to-right serial sum only by
 * float32 rounding, bounded by {@link getSegmentedScanTreeHeight}.
 *
 * @internal
 */

/** Items per workgroup of the segmented scans. */
export const SEGMENTED_SCAN_WORKGROUP_SIZE = 256;

/** `u32` words stored per block aggregate of the vertex-distance scan. */
export const DISTANCE_SCAN_STATE_WORDS = 2;

/** `u32` words stored per block aggregate of the profile scan. */
export const PROFILE_SCAN_STATE_WORDS = 8;

/** Number of 256-item blocks covering `itemCount` items (at least 1). */
export function getSegmentedScanBlockCount(itemCount: number): number {
  return Math.max(1, Math.ceil(itemCount / SEGMENTED_SCAN_WORKGROUP_SIZE));
}

/**
 * Upper bound on the number of combine levels between any item and a scanned result: the height of
 * the fixed summation tree. A sum of non-negative terms formed by a tree of additions of height
 * `h` has relative error at most about `h * 2^-24` in float32, against `(n - 1) * 2^-24` for a
 * serial left-to-right sum of `n` terms.
 *
 * Derivation: the inclusive block scan has depth 8 (log2 of 256). The block aggregate is the last
 * lane, depth 8. A lane of the carry kernel folds `chunk = ceil(blockCount / 256)` aggregates
 * serially (`chunk - 1` more levels), the lanes are scanned (8 levels), and each lane then walks
 * its chunk again (`chunk - 1` levels). The down-sweep combines the carry with the block-local
 * prefix (1 level). Total: `8 + (chunk - 1) + 8 + (chunk - 1) + 1 = 2 * chunk + 15`.
 */
export function getSegmentedScanTreeHeight(itemCount: number): number {
  const blockCount = getSegmentedScanBlockCount(itemCount);
  const chunk = Math.ceil(blockCount / SEGMENTED_SCAN_WORKGROUP_SIZE);
  return 2 * chunk + 15;
}

/**
 * Workgroup inclusive scan with Hillis-Steele doubling. Needs `ScanState`, `getIdentityScanState`
 * and `combineScanStates` in scope. The caller must reach it in uniform control flow.
 */
const WORKGROUP_SCAN_WGSL = /* wgsl */ `
var<workgroup> scanScratch: array<ScanState, ${SEGMENTED_SCAN_WORKGROUP_SIZE}>;
fn scanWorkgroup(lane: u32, value: ScanState) -> ScanState {
  scanScratch[lane] = value;
  workgroupBarrier();
  for (var spanLength = 1u; spanLength < ${SEGMENTED_SCAN_WORKGROUP_SIZE}u; spanLength = spanLength << 1u) {
    var leftValue = getIdentityScanState();
    if (lane >= spanLength) {
      leftValue = scanScratch[lane - spanLength];
    }
    workgroupBarrier();
    if (lane >= spanLength) {
      scanScratch[lane] = combineScanStates(leftValue, scanScratch[lane]);
    }
    workgroupBarrier();
  }
  return scanScratch[lane];
}
`;

/** Which parts of a scan's module-scope WGSL a kernel needs. */
export type ScanDeclarationOptions = {
  /** Include `storeAggregate`; the kernel binds `aggregates` as `read_write`. */
  writesAggregates: boolean;
  /** Include the item loaders, which need the line and value bindings in scope. */
  loadsItems: boolean;
};

/** Module-scope WGSL of the vertex-distance scan. Needs the profile `lineConstants` before it. */
export function getDistanceScanDeclarations(options: ScanDeclarationOptions): string {
  const {writesAggregates, loadsItems} = options;
  return /* wgsl */ `
struct ScanState {
  flag: u32,
  sum: f32
}
fn getIdentityScanState() -> ScanState {
  return ScanState(0u, 0.0);
}
// Segmented sum: a flagged right operand starts a new segment and replaces the running sum.
fn combineScanStates(left: ScanState, right: ScanState) -> ScanState {
  var result = right;
  if (right.flag == 0u) {
    result.sum = left.sum + right.sum;
  }
  result.flag = left.flag | right.flag;
  return result;
}
fn loadAggregate(blockIndex: u32) -> ScanState {
  let base = aggregatesOffset + ${DISTANCE_SCAN_STATE_WORDS}u * blockIndex;
  return ScanState(aggregates[base], bitcast<f32>(aggregates[base + 1u]));
}
${
  writesAggregates
    ? `fn storeAggregate(blockIndex: u32, value: ScanState) {
  let base = aggregatesOffset + ${DISTANCE_SCAN_STATE_WORDS}u * blockIndex;
  aggregates[base] = value.flag;
  aggregates[base + 1u] = bitcast<u32>(value.sum);
}`
    : ''
}
${
  loadsItems
    ? `// Number of lines whose first vertex is at or before the vertex (lines are sorted by offset).
fn countLinesStartedBy(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (getVertexBegin(middle) <= vertex) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}
// Edge length from the previous vertex; a line's first vertex (or one outside every line) starts
// a segment with distance 0.
fn getVertexScanState(vertex: u32) -> ScanState {
  let startedCount = countLinesStartedBy(vertex);
  if (startedCount == 0u || getVertexBegin(startedCount - 1u) == vertex) {
    return ScanState(1u, 0.0);
  }
  let currentX = linePositions[linePositionsOffset + 2u * vertex];
  let currentY = linePositions[linePositionsOffset + 2u * vertex + 1u];
  let deltaX = currentX - linePositions[linePositionsOffset + 2u * vertex - 2u];
  let deltaY = currentY - linePositions[linePositionsOffset + 2u * vertex - 1u];
  return ScanState(0u, sqrt(deltaX * deltaX + deltaY * deltaY));
}`
    : ''
}
${WORKGROUP_SCAN_WGSL}`;
}

/**
 * Module-scope WGSL of the profile scan over sample values. Needs `LINE_COUNT`,
 * `SAMPLE_CAPACITY`, `getNaN`, `isFiniteValue` and the `sampleOffsets`, `sampleValues` and
 * `aggregates` bindings in scope.
 *
 * The state summarises a run of samples of one line: whether it holds a finite value, the first
 * and last finite value, the gain and loss between consecutive finite values, and their minimum
 * and maximum. Non-finite samples are the identity (skipped), which keeps the exact semantics of a
 * serial walk that skips them.
 */
export function getProfileScanDeclarations(options: ScanDeclarationOptions): string {
  const {writesAggregates, loadsItems} = options;
  return /* wgsl */ `
struct ScanState {
  flag: u32,
  hasValue: u32,
  firstValue: f32,
  lastValue: f32,
  gain: f32,
  loss: f32,
  minimum: f32,
  maximum: f32
}
fn getIdentityScanState() -> ScanState {
  return ScanState(0u, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}
// Segmented profile monoid: a flagged right operand starts a new line and replaces the running
// state. The junction delta is the difference of the last finite value on the left and the first
// finite value on the right, exactly the consecutive finite pair of a serial walk.
fn combineScanStates(left: ScanState, right: ScanState) -> ScanState {
  var result = right;
  if (right.flag == 0u) {
    if (left.hasValue == 0u) {
      result = right;
    } else if (right.hasValue == 0u) {
      result = left;
    } else {
      let delta = right.firstValue - left.lastValue;
      result.firstValue = left.firstValue;
      result.gain = (left.gain + max(delta, 0.0)) + right.gain;
      result.loss = (left.loss + max(-delta, 0.0)) + right.loss;
      result.minimum = min(left.minimum, right.minimum);
      result.maximum = max(left.maximum, right.maximum);
    }
  }
  result.flag = left.flag | right.flag;
  return result;
}
fn loadAggregate(blockIndex: u32) -> ScanState {
  let base = aggregatesOffset + ${PROFILE_SCAN_STATE_WORDS}u * blockIndex;
  return ScanState(
    aggregates[base],
    aggregates[base + 1u],
    bitcast<f32>(aggregates[base + 2u]),
    bitcast<f32>(aggregates[base + 3u]),
    bitcast<f32>(aggregates[base + 4u]),
    bitcast<f32>(aggregates[base + 5u]),
    bitcast<f32>(aggregates[base + 6u]),
    bitcast<f32>(aggregates[base + 7u])
  );
}
${
  writesAggregates
    ? `fn storeAggregate(blockIndex: u32, value: ScanState) {
  let base = aggregatesOffset + ${PROFILE_SCAN_STATE_WORDS}u * blockIndex;
  aggregates[base] = value.flag;
  aggregates[base + 1u] = value.hasValue;
  aggregates[base + 2u] = bitcast<u32>(value.firstValue);
  aggregates[base + 3u] = bitcast<u32>(value.lastValue);
  aggregates[base + 4u] = bitcast<u32>(value.gain);
  aggregates[base + 5u] = bitcast<u32>(value.loss);
  aggregates[base + 6u] = bitcast<u32>(value.minimum);
  aggregates[base + 7u] = bitcast<u32>(value.maximum);
}`
    : ''
}
${
  loadsItems
    ? `// Line of an emitted sample: the last line whose first sample row is at or before it.
fn findSampleLine(sample: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (sampleOffsets[sampleOffsetsOffset + middle] <= sample) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low - 1u;
}
fn getSampleScanState(sample: u32, line: u32) -> ScanState {
  var result = getIdentityScanState();
  if (sampleOffsets[sampleOffsetsOffset + line] == sample) {
    result.flag = 1u;
  }
  let value = sampleValues[sampleValuesOffset + sample];
  if (isFiniteValue(value)) {
    result.hasValue = 1u;
    result.firstValue = value;
    result.lastValue = value;
    result.minimum = value;
    result.maximum = value;
  }
  return result;
}
fn getEmittedSampleCount() -> u32 {
  return min(sampleOffsets[sampleOffsetsOffset + LINE_COUNT], SAMPLE_CAPACITY);
}`
    : ''
}
${WORKGROUP_SCAN_WGSL}`;
}

/** Body of a block kernel: stores the inclusive state of the block's last lane. */
export function getScanBlockBody(loadStateWGSL: string): string {
  return `let lane = localInvocationIndex;
  ${loadStateWGSL}
  let inclusive = scanWorkgroup(lane, itemState);
  if (lane == ${SEGMENTED_SCAN_WORKGROUP_SIZE - 1}u) {
    storeAggregate(workgroupIndex, inclusive);
  }`;
}

/**
 * Body of the one-workgroup carry kernel: replaces each block aggregate with the segmented
 * exclusive prefix of all earlier blocks. Needs a `BLOCK_COUNT` constant. Each lane folds a
 * contiguous chunk of at most `ceil(BLOCK_COUNT / 256)` blocks, the lanes are scanned, and each
 * lane walks its chunk again; the loops run over blocks, never over samples.
 */
export function getScanCarryBody(): string {
  return `let lane = localInvocationIndex;
  let chunk = (BLOCK_COUNT + ${SEGMENTED_SCAN_WORKGROUP_SIZE - 1}u) / ${SEGMENTED_SCAN_WORKGROUP_SIZE}u;
  let chunkBegin = min(lane * chunk, BLOCK_COUNT);
  let chunkEnd = min(chunkBegin + chunk, BLOCK_COUNT);
  var chunkTotal = getIdentityScanState();
  for (var chunkBlock = chunkBegin; chunkBlock < chunkEnd; chunkBlock++) {
    chunkTotal = combineScanStates(chunkTotal, loadAggregate(chunkBlock));
  }
  let inclusive = scanWorkgroup(lane, chunkTotal);
  var running = getIdentityScanState();
  if (lane > 0u) {
    running = scanScratch[lane - 1u];
  }
  for (var chunkBlock = chunkBegin; chunkBlock < chunkEnd; chunkBlock++) {
    let aggregate = loadAggregate(chunkBlock);
    storeAggregate(chunkBlock, running);
    running = combineScanStates(running, aggregate);
  }
  _ = inclusive;`;
}
