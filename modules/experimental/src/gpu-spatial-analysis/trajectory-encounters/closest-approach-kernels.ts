// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';

const OPERATION = 'GPUTrajectoryClosestApproach';

/** u32 words per segment record: box `x0, y0, t0, x1, y1, t1` (f32 bits), track, track first row. @internal */
export const CLOSEST_APPROACH_SEGMENT_WORDS = 8;
/** u32 words per cell entry: segment row and the cell's `x, y, t` coordinates (i32 bits). @internal */
export const CLOSEST_APPROACH_ENTRY_WORDS = 4;
/** u32 words per hit: low track, high track, then enter, exit, minimum distance and closest time (f32 bits). @internal */
export const CLOSEST_APPROACH_HIT_WORDS = 6;
/** Largest number of space-time cells one segment may cover; more raises `entryOverflow`. @internal */
export const CLOSEST_APPROACH_MAXIMUM_CELLS_PER_SEGMENT = 4096;

/** Sizes shared by the closest-approach kernels. @internal */
export type ClosestApproachShape = {
  rowCount: number;
  trackCount: number;
  cellSize: number;
  timeCellSize: number;
  entryCapacity: number;
  hitCapacity: number;
  pairCapacity: number;
  isWordMode: boolean;
};

/** State words: `[hit count, entry total, entry overflow flag]`. @internal */
export const CLOSEST_APPROACH_STATE_WORDS = 3;

function getShapeConstants(shape: ClosestApproachShape): string {
  return `const ROW_COUNT: u32 = ${shape.rowCount}u;
const TRACK_COUNT: u32 = ${shape.trackCount}u;
const CELL_SIZE: f32 = ${getWGSLFloatLiteral(shape.cellSize)};
const TIME_CELL_SIZE: f32 = ${getWGSLFloatLiteral(shape.timeCellSize)};
const ENTRY_CAPACITY: u32 = ${shape.entryCapacity}u;
const HIT_CAPACITY: u32 = ${shape.hitCapacity}u;
const PAIR_CAPACITY: u32 = ${shape.pairCapacity}u;
const SEGMENT_WORDS: u32 = ${CLOSEST_APPROACH_SEGMENT_WORDS}u;
const ENTRY_WORDS: u32 = ${CLOSEST_APPROACH_ENTRY_WORDS}u;
const HIT_WORDS: u32 = ${CLOSEST_APPROACH_HIT_WORDS}u;
const MAXIMUM_CELLS_PER_SEGMENT: u32 = ${CLOSEST_APPROACH_MAXIMUM_CELLS_PER_SEGMENT}u;
const NONE: u32 = 0xffffffffu;
// Cell coordinates are clamped to +-2^30 so the hash arithmetic never meets i32 overflow.
const CELL_LIMIT: f32 = 1073741824.0;
fn getCellCoordinate(value: f32, size: f32) -> i32 {
  return i32(clamp(floor(value / size), -CELL_LIMIT, CELL_LIMIT));
}
// Hash of a space-time cell. Different cells may share a hash; every user compares coordinates.
fn getCellHash(cell: vec3<i32>) -> u32 {
  return (bitcast<u32>(cell.x) * 73856093u) ^ (bitcast<u32>(cell.y) * 19349663u) ^
    (bitcast<u32>(cell.z) * 83492791u);
}`;
}

/** Time difference `t(a) - t(b)` of two rows in seconds, exact in word mode until the f32 conversion. */
function getTimeSource(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  let wordsA = vec2<u32>(timestamps[timestampsOffset + 2u * a], timestamps[timestampsOffset + 2u * a + 1u]);
  let wordsB = vec2<u32>(timestamps[timestampsOffset + 2u * b], timestamps[timestampsOffset + 2u * b + 1u]);
  return timeWordsDifferenceToF32(wordsA, wordsB);
}`
    : /* wgsl */ `
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}`;
}

function getTimestampBinding(
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>,
  isWordMode: boolean
): WGSLKernelBinding {
  return {name: 'timestamps', view: timestamps, type: isWordMode ? 'u32' : 'f32', access: 'read'};
}

/**
 * Per row: the segment to the next row of the same track as a padded space-time box (`x`, `y`
 * expanded by half the distance, so two segments within the distance at one instant have
 * overlapping boxes; time relative to row 0, padded for f32 rounding) and the number of cells it
 * covers. Rows that start no segment (last row of a track, an invalid track, a non-finite position,
 * a zero or negative duration) and segments over the per-segment cell cap get 0 cells; the cap
 * raises the entry overflow word. @internal
 */
export function createClosestApproachSegmentsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    positions: GraphDataView<'float32x2'>;
    timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    trackOffsets: GraphDataView<'uint32'>;
    trackValid?: GraphDataView<'uint32'>;
    distance: GraphDataView<'float32'>;
    segmentRecords: GraphDataView<'uint32'>;
    segmentCounts: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {shape} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `${shape.isWordMode ? 'segments-words' : 'segments'}${props.trackValid ? '-valid' : ''}`,
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      getTimestampBinding(props.timestamps, shape.isWordMode),
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      ...(props.trackValid
        ? [
            {
              name: 'trackValid',
              view: props.trackValid,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'distance', view: props.distance, type: 'f32', access: 'read'},
      {name: 'segmentRecords', view: props.segmentRecords, type: 'u32', access: 'read_write'},
      {name: 'segmentCounts', view: props.segmentCounts, type: 'u32', access: 'read_write'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: shape.rowCount,
    declarations: `${getShapeConstants(shape)}
${getTimeSource(shape.isWordMode)}
fn findTrack(row: u32) -> u32 {
  if (row < trackOffsets[trackOffsetsOffset] || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NONE;
  }
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] > row) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}
fn isFiniteNumber(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`,
    body: `segmentCounts[segmentCountsOffset + index] = 0u;
  let base = segmentRecordsOffset + index * SEGMENT_WORDS;
  segmentRecords[base + 6u] = NONE;
  let track = findTrack(index);
  if (track == NONE || index + 1u >= trackOffsets[trackOffsetsOffset + track + 1u]) { return; }
  ${props.trackValid ? 'if (trackValid[trackValidOffset + track] == 0u) { return; }' : ''}
  let radius = distance[distanceOffset];
  if (!(isFiniteNumber(radius) && radius >= 0.0)) { return; }
  let start = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  let end = vec2<f32>(positions[positionsOffset + index * 2u + 2u], positions[positionsOffset + index * 2u + 3u]);
  if (!(isFiniteNumber(start.x) && isFiniteNumber(start.y) && isFiniteNumber(end.x) && isFiniteNumber(end.y))) {
    return;
  }
  if (!(rowTimeDifference(index + 1u, index) > 0.0)) { return; }
  let startTime = rowTimeDifference(index, 0u);
  let endTime = rowTimeDifference(index + 1u, 0u);
  // Half the distance on each box (plus a relative margin for the f32 corners); the time pad
  // covers the rounding of times measured from row 0 (two roundings of 2^-24 relative).
  let halfDistance = 0.5 * radius;
  let spatialPad = halfDistance + 1.0e-6 * max(max(abs(start.x), abs(end.x)), max(abs(start.y), abs(end.y)));
  let minimum = min(start, end) - vec2<f32>(spatialPad);
  let maximum = max(start, end) + vec2<f32>(spatialPad);
  let timePad = 1.0e-6 * max(abs(startTime), abs(endTime)) + 1.0e-6 * TIME_CELL_SIZE;
  let minimumTime = min(startTime, endTime) - timePad;
  let maximumTime = max(startTime, endTime) + timePad;
  let firstCell = vec3<i32>(getCellCoordinate(minimum.x, CELL_SIZE), getCellCoordinate(minimum.y, CELL_SIZE),
    getCellCoordinate(minimumTime, TIME_CELL_SIZE));
  let lastCell = vec3<i32>(getCellCoordinate(maximum.x, CELL_SIZE), getCellCoordinate(maximum.y, CELL_SIZE),
    getCellCoordinate(maximumTime, TIME_CELL_SIZE));
  let extent = vec3<f32>(lastCell - firstCell) + vec3<f32>(1.0);
  let cellCount = extent.x * extent.y * extent.z;
  if (!(cellCount <= f32(MAXIMUM_CELLS_PER_SEGMENT))) {
    atomicStore(&state[stateOffset + 2u], 1u);
    return;
  }
  segmentRecords[base] = bitcast<u32>(minimum.x);
  segmentRecords[base + 1u] = bitcast<u32>(minimum.y);
  segmentRecords[base + 2u] = bitcast<u32>(minimumTime);
  segmentRecords[base + 3u] = bitcast<u32>(maximum.x);
  segmentRecords[base + 4u] = bitcast<u32>(maximum.y);
  segmentRecords[base + 5u] = bitcast<u32>(maximumTime);
  segmentRecords[base + 6u] = track;
  segmentRecords[base + 7u] = trackOffsets[trackOffsetsOffset + track];
  segmentCounts[segmentCountsOffset + index] = u32(cellCount);`
  });
}

/** Writes the sentinel key `0xffffffff` and the identity value of every entry slot. @internal */
export function createClosestApproachEntryClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    entryKeys: GraphDataView<'uint32'>;
    entryIdentity: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'entry-clear',
    bindings: [
      {name: 'entryKeys', view: props.entryKeys, type: 'u32', access: 'read_write'},
      {name: 'entryIdentity', view: props.entryIdentity, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.entryKeys.length,
    body: `entryKeys[entryKeysOffset + index] = 0xffffffffu;
  entryIdentity[entryIdentityOffset + index] = index;`
  });
}

/** Per segment: one `(cell hash, segment, cell)` entry per covered cell, up to the entry capacity. @internal */
export function createClosestApproachEntriesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    segmentRecords: GraphDataView<'uint32'>;
    segmentCounts: GraphDataView<'uint32'>;
    segmentStarts: GraphDataView<'uint32'>;
    entryKeys: GraphDataView<'uint32'>;
    entryRecords: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'entries',
    bindings: [
      {name: 'segmentRecords', view: props.segmentRecords, type: 'u32', access: 'read'},
      {name: 'segmentCounts', view: props.segmentCounts, type: 'u32', access: 'read'},
      {name: 'segmentStarts', view: props.segmentStarts, type: 'u32', access: 'read'},
      {name: 'entryKeys', view: props.entryKeys, type: 'u32', access: 'read_write'},
      {name: 'entryRecords', view: props.entryRecords, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.rowCount,
    declarations: getShapeConstants(props.shape),
    body: `let count = segmentCounts[segmentCountsOffset + index];
  if (count == 0u) { return; }
  let base = segmentRecordsOffset + index * SEGMENT_WORDS;
  let firstCell = vec3<i32>(
    getCellCoordinate(bitcast<f32>(segmentRecords[base]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 1u]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 2u]), TIME_CELL_SIZE));
  let lastCell = vec3<i32>(
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 3u]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 4u]), CELL_SIZE),
    getCellCoordinate(bitcast<f32>(segmentRecords[base + 5u]), TIME_CELL_SIZE));
  let extent = vec3<u32>(lastCell - firstCell) + vec3<u32>(1u);
  let start = segmentStarts[segmentStartsOffset + index];
  for (var cellIndex = 0u; cellIndex < count; cellIndex++) {
    let slot = start + cellIndex;
    // A wrapped (saturated) prefix or a full scratch stops here; the total reports the overflow.
    if (slot < start || slot >= ENTRY_CAPACITY) { break; }
    let offset = vec3<u32>(cellIndex % extent.x, (cellIndex / extent.x) % extent.y, cellIndex / (extent.x * extent.y));
    let cell = firstCell + vec3<i32>(offset);
    entryKeys[entryKeysOffset + slot] = getCellHash(cell);
    let entry = entryRecordsOffset + slot * ENTRY_WORDS;
    entryRecords[entry] = index;
    entryRecords[entry + 1u] = bitcast<u32>(cell.x);
    entryRecords[entry + 2u] = bitcast<u32>(cell.y);
    entryRecords[entry + 3u] = bitcast<u32>(cell.z);
  }`
  });
}

/**
 * Per cell entry in hash order: tests every later entry of the same hash run that is in the same
 * cell and belongs to another track. A segment pair is tested once, in the cell holding the minimum
 * corner of the intersection of the two boxes (both segments have an entry there). Inside the
 * common time interval both positions are linear, so the separation is `D + V tau` and the squared
 * distance is the quadratic `|V|^2 tau^2 + 2 (D . V) tau + |D|^2`: the closest approach is at
 * `tau = -(D . V) / |V|^2` clamped to the interval, and the time within the distance is the
 * interval between the roots of `quadratic - distance^2`, clamped (MobilityDB `tdwithin`). Times
 * are measured from the earlier segment row with `rowTimeDifference`, so word timestamps stay exact
 * until the final f32 difference, and are reported relative to the first sample of the lower track.
 * @internal
 */
export function createClosestApproachPairsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    sortedEntries: GraphDataView<'uint32'>;
    entryRecords: GraphDataView<'uint32'>;
    segmentRecords: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x2'>;
    timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    distance: GraphDataView<'float32'>;
    state: GraphDataView<'uint32'>;
    hits: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {shape} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: shape.isWordMode ? 'pairs-words' : 'pairs',
    bindings: [
      {name: 'sortedEntries', view: props.sortedEntries, type: 'u32', access: 'read'},
      {name: 'entryRecords', view: props.entryRecords, type: 'u32', access: 'read'},
      {name: 'segmentRecords', view: props.segmentRecords, type: 'u32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      getTimestampBinding(props.timestamps, shape.isWordMode),
      {name: 'distance', view: props.distance, type: 'f32', access: 'read'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
      {name: 'hits', view: props.hits, type: 'u32', access: 'read_write'}
    ],
    invocationCount: shape.entryCapacity,
    declarations: `${getShapeConstants(shape)}
${getTimeSource(shape.isWordMode)}
fn getEntryCell(entry: u32) -> vec3<i32> {
  let base = entryRecordsOffset + entry * ENTRY_WORDS;
  return vec3<i32>(bitcast<i32>(entryRecords[base + 1u]), bitcast<i32>(entryRecords[base + 2u]),
    bitcast<i32>(entryRecords[base + 3u]));
}
fn getBoxMinimum(segment: u32) -> vec3<f32> {
  let base = segmentRecordsOffset + segment * SEGMENT_WORDS;
  return vec3<f32>(bitcast<f32>(segmentRecords[base]), bitcast<f32>(segmentRecords[base + 1u]),
    bitcast<f32>(segmentRecords[base + 2u]));
}
fn getBoxMaximum(segment: u32) -> vec3<f32> {
  let base = segmentRecordsOffset + segment * SEGMENT_WORDS;
  return vec3<f32>(bitcast<f32>(segmentRecords[base + 3u]), bitcast<f32>(segmentRecords[base + 4u]),
    bitcast<f32>(segmentRecords[base + 5u]));
}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}
// Position of segment row 'row' at time 'time' measured from row 'origin'.
fn getSegmentPosition(row: u32, origin: u32, time: f32) -> vec2<f32> {
  let startTime = rowTimeDifference(row, origin);
  let duration = rowTimeDifference(row + 1u, row);
  let fraction = clamp((time - startTime) / duration, 0.0, 1.0);
  return mix(getPosition(row), getPosition(row + 1u), fraction);
}
// Tests segments 'first' and 'second' of different tracks and appends a hit when they come within
// the distance during their common time interval.
fn testSegments(first: u32, second: u32, radius: f32) {
  let firstTrack = segmentRecords[segmentRecordsOffset + first * SEGMENT_WORDS + 6u];
  let secondTrack = segmentRecords[segmentRecordsOffset + second * SEGMENT_WORDS + 6u];
  let lowIsFirst = firstTrack < secondTrack;
  let low = select(second, first, lowIsFirst);
  let high = select(first, second, lowIsFirst);
  // Times are measured from the low segment's start row.
  let lowEnd = rowTimeDifference(low + 1u, low);
  let highStart = rowTimeDifference(high, low);
  let highEnd = rowTimeDifference(high + 1u, low);
  let commonStart = max(0.0, highStart);
  let commonEnd = min(lowEnd, highEnd);
  if (commonStart > commonEnd) { return; }
  let separationStart = getSegmentPosition(high, low, commonStart) - getSegmentPosition(low, low, commonStart);
  let separationEnd = getSegmentPosition(high, low, commonEnd) - getSegmentPosition(low, low, commonEnd);
  let span = commonEnd - commonStart;
  var velocity = vec2<f32>(0.0);
  if (span > 0.0) { velocity = (separationEnd - separationStart) / span; }
  let a = dot(velocity, velocity);
  let b = dot(separationStart, velocity);
  let c = dot(separationStart, separationStart) - radius * radius;
  var enter = 0.0;
  var exit = span;
  var closest = 0.0;
  if (a > 0.0) {
    closest = clamp(-b / a, 0.0, span);
    let discriminant = b * b - a * c;
    if (discriminant < 0.0) { return; }
    let root = sqrt(discriminant);
    enter = max((-b - root) / a, 0.0);
    exit = min((-b + root) / a, span);
    if (enter > exit) { return; }
  } else if (c > 0.0) {
    return;
  }
  let nearest = separationStart + velocity * closest;
  let slot = atomicAdd(&state[stateOffset], 1u);
  if (slot >= HIT_CAPACITY) { return; }
  // Report times relative to the first sample of the lower track.
  let lowTrack = segmentRecords[segmentRecordsOffset + low * SEGMENT_WORDS + 6u];
  let lowFirstRow = segmentRecords[segmentRecordsOffset + low * SEGMENT_WORDS + 7u];
  let base = rowTimeDifference(low, lowFirstRow) + commonStart;
  let highTrack = segmentRecords[segmentRecordsOffset + high * SEGMENT_WORDS + 6u];
  let hit = hitsOffset + slot * HIT_WORDS;
  hits[hit] = lowTrack;
  hits[hit + 1u] = highTrack;
  hits[hit + 2u] = bitcast<u32>(max(base + enter, 0.0));
  hits[hit + 3u] = bitcast<u32>(max(base + exit, 0.0));
  hits[hit + 4u] = bitcast<u32>(length(nearest));
  hits[hit + 5u] = bitcast<u32>(max(base + closest, 0.0));
}`,
    body: `let entryTotal = min(atomicLoad(&state[stateOffset + 1u]), ENTRY_CAPACITY);
  if (index >= entryTotal) { return; }
  let radius = distance[distanceOffset];
  let entry = sortedEntries[sortedEntriesOffset + index];
  let cell = getEntryCell(entry);
  let hash = getCellHash(cell);
  let segment = entryRecords[entryRecordsOffset + entry * ENTRY_WORDS];
  let track = segmentRecords[segmentRecordsOffset + segment * SEGMENT_WORDS + 6u];
  let boxMinimum = getBoxMinimum(segment);
  let boxMaximum = getBoxMaximum(segment);
  for (var next = index + 1u; next < entryTotal; next++) {
    let other = sortedEntries[sortedEntriesOffset + next];
    let otherCell = getEntryCell(other);
    if (getCellHash(otherCell) != hash) { break; }
    if (any(otherCell != cell)) { continue; }
    let otherSegment = entryRecords[entryRecordsOffset + other * ENTRY_WORDS];
    if (segmentRecords[segmentRecordsOffset + otherSegment * SEGMENT_WORDS + 6u] == track) { continue; }
    let corner = max(boxMinimum, getBoxMinimum(otherSegment));
    let farCorner = min(boxMaximum, getBoxMaximum(otherSegment));
    if (any(corner > farCorner)) { continue; }
    let referenceCell = vec3<i32>(getCellCoordinate(corner.x, CELL_SIZE), getCellCoordinate(corner.y, CELL_SIZE),
      getCellCoordinate(corner.z, TIME_CELL_SIZE));
    if (any(referenceCell != cell)) { continue; }
    testSegments(segment, otherSegment, radius);
  }`
  });
}

/**
 * Sort keys of the stable hit chain: enter time, then high track, then low track, giving
 * `(low, high, enter)` order. Slots at and after the hit count sort last. @internal
 */
export function createClosestApproachSortKeyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    variant: 'enter' | 'high' | 'low';
    state: GraphDataView<'uint32'>;
    hits: GraphDataView<'uint32'>;
    /** Previous sort order; absent for the first (enter) pass, which writes the identity. */
    order?: GraphDataView<'uint32'>;
    identity?: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const word = props.variant === 'enter' ? 2 : props.variant === 'high' ? 1 : 0;
  const sentinel = props.variant === 'low' ? 'TRACK_COUNT' : '0u';
  const bindings: WGSLKernelBinding[] = [
    {name: 'state', view: props.state, type: 'u32', access: 'read'},
    {name: 'hits', view: props.hits, type: 'u32', access: 'read'}
  ];
  if (props.order) {
    bindings.push({name: 'order', view: props.order, type: 'u32', access: 'read'});
  }
  if (props.identity) {
    bindings.push({name: 'identity', view: props.identity, type: 'u32', access: 'read_write'});
  }
  bindings.push({name: 'keys', view: props.keys, type: 'u32', access: 'read_write'});
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `sort-key-${props.variant}`,
    bindings,
    invocationCount: props.shape.hitCapacity,
    declarations: getShapeConstants(props.shape),
    body: `let activeCount = min(state[stateOffset], HIT_CAPACITY);
  ${props.identity ? 'identity[identityOffset + index] = index;' : ''}
  let slot = ${props.order ? 'order[orderOffset + index]' : 'index'};
  keys[keysOffset + index] = select(${sentinel}, hits[hitsOffset + slot * HIT_WORDS + ${word}u], slot < activeCount);`
  });
}

/** Gathers the sorted hit columns. Slots at and after the hit count get sentinels. @internal */
export function createClosestApproachSortedHitsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    state: GraphDataView<'uint32'>;
    order: GraphDataView<'uint32'>;
    hits: GraphDataView<'uint32'>;
    sortedHigh: GraphDataView<'uint32'>;
    sortedValues: GraphDataView<'float32'>;
    flags: GraphDataView<'uint32'>;
    rowIds: GraphDataView<'uint32'>;
    sortedLow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'sorted-hits',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'order', view: props.order, type: 'u32', access: 'read'},
      {name: 'hits', view: props.hits, type: 'u32', access: 'read'},
      {name: 'sortedLow', view: props.sortedLow, type: 'u32', access: 'read'},
      {name: 'sortedHigh', view: props.sortedHigh, type: 'u32', access: 'read_write'},
      {name: 'sortedValues', view: props.sortedValues, type: 'f32', access: 'read_write'},
      {name: 'flags', view: props.flags, type: 'u32', access: 'read_write'},
      {name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.hitCapacity,
    declarations: getShapeConstants(props.shape),
    body: `rowIds[rowIdsOffset + index] = index;
  let activeCount = min(state[stateOffset], HIT_CAPACITY);
  if (index >= activeCount) {
    sortedHigh[sortedHighOffset + index] = NONE;
    for (var word = 0u; word < 4u; word++) { sortedValues[sortedValuesOffset + index * 4u + word] = 0.0; }
    flags[flagsOffset + index] = 0u;
    return;
  }
  let hit = hitsOffset + order[orderOffset + index] * HIT_WORDS;
  let high = hits[hit + 1u];
  sortedHigh[sortedHighOffset + index] = high;
  for (var word = 0u; word < 4u; word++) {
    sortedValues[sortedValuesOffset + index * 4u + word] = bitcast<f32>(hits[hit + 2u + word]);
  }
  // The first hit of each (low, high) run starts a pair. The previous high is read from the hit
  // table, not from sortedHigh, which other invocations are writing.
  var isStart = index == 0u;
  if (!isStart) {
    let previous = hitsOffset + order[orderOffset + index - 1u] * HIT_WORDS;
    isStart = sortedLow[sortedLowOffset + index] != sortedLow[sortedLowOffset + index - 1u] ||
      hits[previous + 1u] != high;
  }
  flags[flagsOffset + index] = select(0u, 1u, isStart);`
  });
}

/** Per-pair output columns of {@link createClosestApproachReduceNodes}. @internal */
export type ClosestApproachPairColumns = {
  ids: GraphDataView<'uint32'>;
  partners?: GraphDataView<'uint32'>;
  minimumDistances?: GraphDataView<'float32'>;
  closestTimes?: GraphDataView<'float32'>;
  firstTimes?: GraphDataView<'float32'>;
  durations?: GraphDataView<'float32'>;
};

/**
 * Writes the per-pair columns from the sorted runs: ids and partners, then the minimum distance,
 * its time (ties to the earliest), the first enter time and the summed time within the distance
 * (the segment pairs of one track pair split time at sample instants, so their intervals only
 * share end points). Every slot is rewritten. @internal
 */
export function createClosestApproachReduceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    state: GraphDataView<'uint32'>;
    runStarts: GraphDataView<'uint32'>;
    runCount: GraphDataView<'uint32'>;
    sortedLow: GraphDataView<'uint32'>;
    sortedHigh: GraphDataView<'uint32'>;
    sortedValues: GraphDataView<'float32'>;
    columns: ClosestApproachPairColumns;
  }
): GPUCommandNode<Parameters>[] {
  const {columns} = props;
  const declarations = getShapeConstants(props.shape);
  const live = `let isLive = index < min(runCount[runCountOffset], PAIR_CAPACITY);
  let start = select(0u, runStarts[runStartsOffset + index], isLive);`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-identity`,
      operation: OPERATION,
      variant: 'reduce-identity',
      bindings: [
        {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
        {name: 'runCount', view: props.runCount, type: 'u32', access: 'read'},
        {name: 'sortedLow', view: props.sortedLow, type: 'u32', access: 'read'},
        {name: 'sortedHigh', view: props.sortedHigh, type: 'u32', access: 'read'},
        {name: 'ids', view: columns.ids, type: 'u32', access: 'read_write'},
        ...(columns.partners
          ? [
              {
                name: 'partners',
                view: columns.partners,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: props.shape.pairCapacity,
      declarations,
      body: `${live}
  ids[idsOffset + index] = select(NONE, sortedLow[sortedLowOffset + start], isLive);
  ${columns.partners ? 'partners[partnersOffset + index] = select(NONE, sortedHigh[sortedHighOffset + start], isLive);' : ''}`
    })
  ];
  const statistics = (
    [
      ['minimumDistances', columns.minimumDistances, 'minimumDistance'],
      ['closestTimes', columns.closestTimes, 'closestTime'],
      ['firstTimes', columns.firstTimes, 'firstTime'],
      ['durations', columns.durations, 'duration']
    ] as const
  ).filter(([, view]) => view);
  if (statistics.length > 0) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-statistics`,
        operation: OPERATION,
        variant: `reduce-statistics-${statistics.map(([name]) => name).join('-')}`,
        bindings: [
          {name: 'state', view: props.state, type: 'u32', access: 'read'},
          {name: 'runStarts', view: props.runStarts, type: 'u32', access: 'read'},
          {name: 'runCount', view: props.runCount, type: 'u32', access: 'read'},
          {name: 'sortedValues', view: props.sortedValues, type: 'f32', access: 'read'},
          ...statistics.map(
            ([name, view]): WGSLKernelBinding => ({
              name,
              view: view as GraphDataView,
              type: 'f32',
              access: 'read_write'
            })
          )
        ],
        invocationCount: props.shape.pairCapacity,
        declarations,
        body: `${live}
  var end = start;
  if (isLive) {
    end = min(state[stateOffset], HIT_CAPACITY);
    if (index + 1u < min(runCount[runCountOffset], PAIR_CAPACITY)) { end = runStarts[runStartsOffset + index + 1u]; }
  }
  // Hits of one pair are in enter-time order.
  var firstTime = 0.0;
  var minimumDistance = 0.0;
  var closestTime = 0.0;
  var duration = 0.0;
  if (isLive) {
    firstTime = sortedValues[sortedValuesOffset + start * 4u];
    minimumDistance = 3.402823466e+38;
  }
  for (var hit = start; hit < end; hit++) {
    let base = sortedValuesOffset + hit * 4u;
    let hitDistance = sortedValues[base + 2u];
    let hitClosest = sortedValues[base + 3u];
    if (hitDistance < minimumDistance || (hitDistance == minimumDistance && hitClosest < closestTime)) {
      minimumDistance = hitDistance;
      closestTime = hitClosest;
    }
    duration = duration + (sortedValues[base + 1u] - sortedValues[base]);
  }
  ${statistics.map(([name, , value]) => `${name}[${name}Offset + index] = ${value};`).join('\n  ')}`
      })
    );
  }
  return nodes;
}

/** Writes the two status words: entry scratch overflow (or a segment over the cell cap) and hit scratch overflow. @internal */
export function createClosestApproachStatusNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ClosestApproachShape;
    state: GraphDataView<'uint32'>;
    entryOverflow: GraphDataView<'uint32'>;
    hitOverflow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'status',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'entryOverflow', view: props.entryOverflow, type: 'u32', access: 'read_write'},
      {name: 'hitOverflow', view: props.hitOverflow, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: getShapeConstants(props.shape),
    body: `entryOverflow[entryOverflowOffset] =
    select(0u, 1u, state[stateOffset + 1u] > ENTRY_CAPACITY || state[stateOffset + 2u] != 0u);
  hitOverflow[hitOverflowOffset] = select(0u, 1u, state[stateOffset] > HIT_CAPACITY);`
  });
}
