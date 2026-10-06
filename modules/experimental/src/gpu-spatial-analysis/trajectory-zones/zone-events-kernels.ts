// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUBVH, GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';
import {SPATIAL_JOIN_WGSL_HELPERS} from '../spatial-join/spatial-join-passes';

const OPERATION = 'GPUZoneEvents';
/** Bit marking a candidate query row as a first-row point-in-polygon ray. @internal */
const RAY_FLAG = '0x80000000u';

/** Sizes shared by the zone-event kernels. @internal */
export type ZoneEventsShape = {
  rowCount: number;
  trackCount: number;
  edgeCount: number;
  zoneCount: number;
  candidateCapacity: number;
  maximumEventsPerTrack: number;
};

type Nodes<Parameters> = GPUCommandNode<Parameters>[];

/** WGSL track lookup over a `trackOffsets` binding. @internal */
const TRACK_LOOKUP_WGSL = /* wgsl */ `
const NO_TRACK: u32 = 0xffffffffu;
fn findTrack(row: u32) -> u32 {
  if (row < trackOffsets[trackOffsetsOffset] || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NO_TRACK;
  }
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] > row) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`;

function getTimeSource(isWordMode: boolean): string {
  return isWordMode
    ? /* wgsl */ `${TIME_WORDS_WGSL}
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  let wordsA = vec2<u32>(timestamps[timestampsOffset + 2u * a], timestamps[timestampsOffset + 2u * a + 1u]);
  let wordsB = vec2<u32>(timestamps[timestampsOffset + 2u * b], timestamps[timestampsOffset + 2u * b + 1u]);
  return timeWordsToF32(timeWordsSubtract(wordsA, wordsB));
}`
    : /* wgsl */ `
fn rowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}`;
}

function getTrackConstants(shape: ZoneEventsShape): string {
  return `const TRACK_COUNT: u32 = ${shape.trackCount}u;
const ZONE_COUNT: u32 = ${shape.zoneCount}u;
const EDGE_COUNT: u32 = ${shape.edgeCount}u;
const CANDIDATE_CAPACITY: u32 = ${shape.candidateCapacity}u;
const MAXIMUM_EVENTS_PER_TRACK: u32 = ${shape.maximumEventsPerTrack}u;`;
}

/** Zeroes the counters and the initial-inside parity matrix. @internal */
export function createZoneClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    state: GraphDataView<'uint32'>;
    initialInside: GraphDataView<'uint32'>;
    cellCount: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'clear',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read_write'},
      {name: 'initialInside', view: props.initialInside, type: 'u32', access: 'read_write'}
    ],
    invocationCount: Math.max(props.cellCount, 4),
    body: `if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${props.cellCount}u) { initialInside[initialInsideOffset + index] = 0u; }`
  });
}

/**
 * BVH probe: one invocation per row. A row after the first of its track probes the BVH with its
 * segment's bounds. The first row of a track probes with a ray box toward +x, whose crossings
 * give the initial point-in-polygon parity. @internal
 */
export function createZoneProbeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    positions: GraphDataView<'float32x2'>;
    trackOffsets: GraphDataView<'uint32'>;
    bvh: GPUBVH;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {bvh, shape} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'probe',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
      {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read_write'}
    ],
    invocationCount: shape.rowCount,
    declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
${getTrackConstants(shape)}
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
${TRACK_LOOKUP_WGSL}
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}`,
    body: `let row = index;
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  let isFirst = row == trackOffsets[trackOffsetsOffset + track];
  let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  if (!isFiniteValue(current.x) || !isFiniteValue(current.y)) { return; }
  var queryMinimum = current;
  var queryMaximum = vec2f(FLOAT32_MAXIMUM, current.y);
  if (!isFirst) {
    let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
    if (!isFiniteValue(previous.x) || !isFiniteValue(previous.y)) { return; }
    queryMinimum = min(previous, current);
    queryMaximum = max(previous, current);
  }
  let rowWord = select(row, row | ${RAY_FLAG}, isFirst);
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let edgeRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (edgeRow < EDGE_COUNT) {
        let slot = atomicAdd(&state[stateOffset], 1u);
        if (slot < CANDIDATE_CAPACITY) {
          candidatePairs[candidatePairsOffset + slot * 2u] = rowWord;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = edgeRow;
        }
      }
    }
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }`
  });
}

/**
 * Exact crossing test per candidate. `candidateKinds` is 0 (none), 1 (segment crossing with
 * parameter in `candidateParameters`) or 2 (ray crossing). Every slot is rewritten. @internal
 */
export function createZoneCrossingNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    positions: GraphDataView<'float32x2'>;
    edgeStarts: GraphDataView<'float32x2'>;
    edgeEnds: GraphDataView<'float32x2'>;
    edgeZones: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    candidateKinds: GraphDataView<'uint32'>;
    candidateParameters: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'crossing',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'edgeStarts', view: props.edgeStarts, type: 'f32', access: 'read'},
      {name: 'edgeEnds', view: props.edgeEnds, type: 'f32', access: 'read'},
      {name: 'edgeZones', view: props.edgeZones, type: 'u32', access: 'read'},
      {name: 'candidateKinds', view: props.candidateKinds, type: 'u32', access: 'read_write'},
      {
        name: 'candidateParameters',
        view: props.candidateParameters,
        type: 'f32',
        access: 'read_write'
      }
    ],
    invocationCount: props.shape.candidateCapacity,
    declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
${getTrackConstants(props.shape)}
fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }`,
    body: `candidateKinds[candidateKindsOffset + index] = 0u;
  candidateParameters[candidateParametersOffset + index] = 0.0;
  if (index >= min(state[stateOffset], CANDIDATE_CAPACITY)) { return; }
  let rowWord = candidatePairs[candidatePairsOffset + index * 2u];
  let edge = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (edgeZones[edgeZonesOffset + edge] >= ZONE_COUNT) { return; }
  let a = vec2f(edgeStarts[edgeStartsOffset + edge * 2u], edgeStarts[edgeStartsOffset + edge * 2u + 1u]);
  let b = vec2f(edgeEnds[edgeEndsOffset + edge * 2u], edgeEnds[edgeEndsOffset + edge * 2u + 1u]);
  if (!isFiniteValue(a.x) || !isFiniteValue(a.y) || !isFiniteValue(b.x) || !isFiniteValue(b.y)) { return; }
  let isRay = (rowWord & ${RAY_FLAG}) != 0u;
  let row = rowWord & 0x7fffffffu;
  let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  if (isRay) {
    if ((a.y > current.y) != (b.y > current.y)) {
      let crossingX = a.x + (current.y - a.y) * (b.x - a.x) / (b.y - a.y);
      if (current.x < crossingX) { candidateKinds[candidateKindsOffset + index] = 2u; }
    }
    return;
  }
  let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
  let direction = current - previous;
  let edgeDirection = b - a;
  let denominator = cross2(direction, edgeDirection);
  if (denominator == 0.0) { return; }
  let offset = a - previous;
  let along = cross2(offset, edgeDirection) / denominator;
  let across = cross2(offset, direction) / denominator;
  if (along >= 0.0 && along < 1.0 && across >= 0.0 && across < 1.0) {
    candidateKinds[candidateKindsOffset + index] = 1u;
    candidateParameters[candidateParametersOffset + index] = along;
  }`
  });
}

/** Toggles the initial-inside parity of `(track, zone)` for each ray crossing. @internal */
export function createZoneInitialStateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    trackOffsets: GraphDataView<'uint32'>;
    edgeZones: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    candidateKinds: GraphDataView<'uint32'>;
    initialInside: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'initial-state',
    bindings: [
      {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
      {name: 'candidateKinds', view: props.candidateKinds, type: 'u32', access: 'read'},
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'edgeZones', view: props.edgeZones, type: 'u32', access: 'read'},
      {name: 'initialInside', view: props.initialInside, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    declarations: `${getTrackConstants(props.shape)}
${TRACK_LOOKUP_WGSL}`,
    body: `if (candidateKinds[candidateKindsOffset + index] != 2u) { return; }
  let row = candidatePairs[candidatePairsOffset + index * 2u] & 0x7fffffffu;
  let edge = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  atomicXor(&initialInside[initialInsideOffset + track * ZONE_COUNT + edgeZones[edgeZonesOffset + edge]], 1u);`
  });
}

/**
 * Converts segment crossings to track-relative event times. Writes the owning track (or the
 * sentinel `TRACK_COUNT` for non-events) and counts events in `state[1]`. @internal
 */
export function createZoneEventTimeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    isWordMode: boolean;
    timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    trackOffsets: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    candidateKinds: GraphDataView<'uint32'>;
    candidateParameters: GraphDataView<'float32'>;
    candidateTracks: GraphDataView<'uint32'>;
    candidateTimes: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.isWordMode ? 'event-time-words' : 'event-time',
    bindings: [
      {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
      {name: 'candidateKinds', view: props.candidateKinds, type: 'u32', access: 'read'},
      {name: 'candidateParameters', view: props.candidateParameters, type: 'f32', access: 'read'},
      {
        name: 'timestamps',
        view: props.timestamps,
        type: props.isWordMode ? 'u32' : 'f32',
        access: 'read'
      },
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'candidateTracks', view: props.candidateTracks, type: 'u32', access: 'read_write'},
      {name: 'candidateTimes', view: props.candidateTimes, type: 'f32', access: 'read_write'},
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    declarations: `${getTrackConstants(props.shape)}
${TRACK_LOOKUP_WGSL}
${getTimeSource(props.isWordMode)}`,
    body: `candidateTracks[candidateTracksOffset + index] = TRACK_COUNT;
  candidateTimes[candidateTimesOffset + index] = 0.0;
  if (candidateKinds[candidateKindsOffset + index] != 1u) { return; }
  let row = candidatePairs[candidatePairsOffset + index * 2u];
  let track = findTrack(row);
  if (track == NO_TRACK) { return; }
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let base = rowTimeDifference(row - 1u, firstRow);
  let span = rowTimeDifference(row, row - 1u);
  let eventTime = max(base + candidateParameters[candidateParametersOffset + index] * span, 0.0);
  candidateTracks[candidateTracksOffset + index] = track;
  candidateTimes[candidateTimesOffset + index] = eventTime;
  atomicAdd(&state[stateOffset + 1u], 1u);`
  });
}

/** Writes the three sort-key columns used by the stable LSD sort chain. @internal */
export function createZoneSortKeyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    variant: 'edge' | 'time' | 'track';
    /** Previous sort order; absent for the first (edge) pass, which also writes the identity. */
    order?: GraphDataView<'uint32'>;
    candidatePairs?: GraphDataView<'uint32x2'>;
    candidateKinds?: GraphDataView<'uint32'>;
    candidateTimes?: GraphDataView<'float32'>;
    candidateTracks?: GraphDataView<'uint32'>;
    keys: GraphDataView<'uint32'>;
    identity?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [];
  let body: string;
  if (props.variant === 'edge') {
    bindings.push(
      {name: 'candidatePairs', view: props.candidatePairs!, type: 'u32', access: 'read'},
      {name: 'candidateKinds', view: props.candidateKinds!, type: 'u32', access: 'read'},
      {name: 'identity', view: props.identity!, type: 'u32', access: 'read_write'}
    );
    body = `identity[identityOffset + index] = index;
  let isEvent = candidateKinds[candidateKindsOffset + index] == 1u;
  keys[keysOffset + index] = select(0u, candidatePairs[candidatePairsOffset + index * 2u + 1u], isEvent);`;
  } else if (props.variant === 'time') {
    bindings.push(
      {name: 'order', view: props.order!, type: 'u32', access: 'read'},
      {name: 'candidateTimes', view: props.candidateTimes!, type: 'f32', access: 'read'}
    );
    body = `keys[keysOffset + index] = bitcast<u32>(candidateTimes[candidateTimesOffset + order[orderOffset + index]]);`;
  } else {
    bindings.push(
      {name: 'order', view: props.order!, type: 'u32', access: 'read'},
      {name: 'candidateTracks', view: props.candidateTracks!, type: 'u32', access: 'read'}
    );
    body = `keys[keysOffset + index] = candidateTracks[candidateTracksOffset + order[orderOffset + index]];`;
  }
  bindings.push({name: 'keys', view: props.keys, type: 'u32', access: 'read_write'});
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `sort-key-${props.variant}`,
    bindings,
    invocationCount: props.shape.candidateCapacity,
    body
  });
}

/**
 * Gathers sorted event columns (zone, relative time, segment end row) and writes the default
 * (not kept) flags. Slots at and after the event count receive sentinels. @internal
 */
export function createZoneSortedColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    order: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    edgeZones: GraphDataView<'uint32'>;
    candidateTimes: GraphDataView<'float32'>;
    sortedZones: GraphDataView<'uint32'>;
    sortedTimes: GraphDataView<'float32'>;
    sortedRows: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'sorted-columns',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'order', view: props.order, type: 'u32', access: 'read'},
      {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
      {name: 'edgeZones', view: props.edgeZones, type: 'u32', access: 'read'},
      {name: 'candidateTimes', view: props.candidateTimes, type: 'f32', access: 'read'},
      {name: 'sortedZones', view: props.sortedZones, type: 'u32', access: 'read_write'},
      {name: 'sortedTimes', view: props.sortedTimes, type: 'f32', access: 'read_write'},
      {name: 'sortedRows', view: props.sortedRows, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    body: `if (index >= state[stateOffset + 1u]) {
    sortedZones[sortedZonesOffset + index] = 0xffffffffu;
    sortedTimes[sortedTimesOffset + index] = 0.0;
    sortedRows[sortedRowsOffset + index] = 0xffffffffu;
    return;
  }
  let slot = order[orderOffset + index];
  let edge = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  sortedZones[sortedZonesOffset + index] = edgeZones[edgeZonesOffset + edge];
  sortedTimes[sortedTimesOffset + index] = candidateTimes[candidateTimesOffset + slot];
  sortedRows[sortedRowsOffset + index] = candidatePairs[candidatePairsOffset + slot * 2u];`
  });
}

/** Seeds per `(track, zone)` state from the initial point-in-polygon parity. @internal */
export function createZoneSeedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    initialInside: GraphDataView<'uint32'>;
    walkState: GraphDataView<'uint32'>;
    enterTimes: GraphDataView<'float32'>;
    dwellTimes: GraphDataView<'float32'>;
    visitCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'seed',
    bindings: [
      {name: 'initialInside', view: props.initialInside, type: 'u32', access: 'read'},
      {name: 'walkState', view: props.walkState, type: 'u32', access: 'read_write'},
      {name: 'enterTimes', view: props.enterTimes, type: 'f32', access: 'read_write'},
      {name: 'dwellTimes', view: props.dwellTimes, type: 'f32', access: 'read_write'},
      {name: 'visitCounts', view: props.visitCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let inside = initialInside[initialInsideOffset + index] & 1u;
  walkState[walkStateOffset + index] = inside;
  enterTimes[enterTimesOffset + index] = 0.0;
  dwellTimes[dwellTimesOffset + index] = 0.0;
  visitCounts[visitCountsOffset + index] = inside;`
  });
}

/**
 * Per-track sequential walk over the sorted events: alternates enter and exit per zone from the
 * seeded state, ranks events inside the track, and flags truncation. @internal
 */
export function createZoneWalkNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    sortedTracks: GraphDataView<'uint32'>;
    sortedZones: GraphDataView<'uint32'>;
    walkState: GraphDataView<'uint32'>;
    eventTypes: GraphDataView<'uint32'>;
    eventRanks: GraphDataView<'uint32'>;
    trackEventCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'walk',
    bindings: [
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
      {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
      {name: 'sortedZones', view: props.sortedZones, type: 'u32', access: 'read'},
      {name: 'walkState', view: props.walkState, type: 'u32', access: 'read_write'},
      {name: 'eventTypes', view: props.eventTypes, type: 'u32', access: 'read_write'},
      {name: 'eventRanks', view: props.eventRanks, type: 'u32', access: 'read_write'},
      {name: 'trackEventCounts', view: props.trackEventCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount,
    declarations: getTrackConstants(props.shape),
    body: `let eventTotal = atomicLoad(&state[stateOffset + 1u]);
  var low = 0u;
  var high = eventTotal;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (sortedTracks[sortedTracksOffset + middle] < index) { low = middle + 1u; } else { high = middle; }
  }
  var rank = 0u;
  var event = low;
  loop {
    if (event >= eventTotal || sortedTracks[sortedTracksOffset + event] != index) { break; }
    let cell = index * ZONE_COUNT + sortedZones[sortedZonesOffset + event];
    let next = walkState[walkStateOffset + cell] ^ 1u;
    walkState[walkStateOffset + cell] = next;
    eventTypes[eventTypesOffset + event] = select(1u, 0u, next == 1u);
    eventRanks[eventRanksOffset + event] = rank;
    rank = rank + 1u;
    event = event + 1u;
  }
  trackEventCounts[trackEventCountsOffset + index] = rank;
  if (rank > MAXIMUM_EVENTS_PER_TRACK) { atomicOr(&state[stateOffset + 2u], 1u); }`
  });
}

/** Writes the keep flags and identity rows consumed by the final compaction. @internal */
export function createZoneKeepFlagsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    eventRanks: GraphDataView<'uint32'>;
    keepFlags: GraphDataView<'uint32'>;
    rowIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'keep-flags',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'eventRanks', view: props.eventRanks, type: 'u32', access: 'read'},
      {name: 'keepFlags', view: props.keepFlags, type: 'u32', access: 'read_write'},
      {name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    declarations: getTrackConstants(props.shape),
    body: `rowIds[rowIdsOffset + index] = index;
  let isKept = index < state[stateOffset + 1u] && eventRanks[eventRanksOffset + index] < MAXIMUM_EVENTS_PER_TRACK;
  keepFlags[keepFlagsOffset + index] = select(0u, 1u, isKept);`
  });
}

/**
 * Per-track dwell accumulation in event order. An exit adds `time - enterTime`; an enter records
 * the time and counts a visit. Deterministic because each track is walked sequentially. @internal
 */
export function createZoneDwellNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    sortedTracks: GraphDataView<'uint32'>;
    sortedZones: GraphDataView<'uint32'>;
    sortedTimes: GraphDataView<'float32'>;
    eventTypes: GraphDataView<'uint32'>;
    enterTimes: GraphDataView<'float32'>;
    dwellTimes: GraphDataView<'float32'>;
    visitCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'dwell',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
      {name: 'sortedZones', view: props.sortedZones, type: 'u32', access: 'read'},
      {name: 'sortedTimes', view: props.sortedTimes, type: 'f32', access: 'read'},
      {name: 'eventTypes', view: props.eventTypes, type: 'u32', access: 'read'},
      {name: 'enterTimes', view: props.enterTimes, type: 'f32', access: 'read_write'},
      {name: 'dwellTimes', view: props.dwellTimes, type: 'f32', access: 'read_write'},
      {name: 'visitCounts', view: props.visitCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount,
    declarations: getTrackConstants(props.shape),
    body: `let eventTotal = state[stateOffset + 1u];
  var low = 0u;
  var high = eventTotal;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (sortedTracks[sortedTracksOffset + middle] < index) { low = middle + 1u; } else { high = middle; }
  }
  var event = low;
  loop {
    if (event >= eventTotal || sortedTracks[sortedTracksOffset + event] != index) { break; }
    let cell = index * ZONE_COUNT + sortedZones[sortedZonesOffset + event];
    let eventTime = sortedTimes[sortedTimesOffset + event];
    if (eventTypes[eventTypesOffset + event] == 0u) {
      enterTimes[enterTimesOffset + cell] = eventTime;
      visitCounts[visitCountsOffset + cell] = visitCounts[visitCountsOffset + cell] + 1u;
    } else {
      dwellTimes[dwellTimesOffset + cell] = dwellTimes[dwellTimesOffset + cell] + (eventTime - enterTimes[enterTimesOffset + cell]);
    }
    event = event + 1u;
  }`
  });
}

/** Closes still-open visits at the last sample of each track. @internal */
export function createZoneCloseNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    isWordMode: boolean;
    timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    trackOffsets: GraphDataView<'uint32'>;
    walkState: GraphDataView<'uint32'>;
    enterTimes: GraphDataView<'float32'>;
    dwellTimes: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.isWordMode ? 'close-words' : 'close',
    bindings: [
      {
        name: 'timestamps',
        view: props.timestamps,
        type: props.isWordMode ? 'u32' : 'f32',
        access: 'read'
      },
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'walkState', view: props.walkState, type: 'u32', access: 'read'},
      {name: 'enterTimes', view: props.enterTimes, type: 'f32', access: 'read'},
      {name: 'dwellTimes', view: props.dwellTimes, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount * props.shape.zoneCount,
    declarations: `${getTrackConstants(props.shape)}
${getTimeSource(props.isWordMode)}`,
    body: `if (walkState[walkStateOffset + index] == 0u) { return; }
  let track = index / ZONE_COUNT;
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
  let duration = rowTimeDifference(lastRow, firstRow);
  dwellTimes[dwellTimesOffset + index] = dwellTimes[dwellTimesOffset + index] + (duration - enterTimes[enterTimesOffset + index]);`
  });
}

/** One column copied from sorted events to compact output slots. @internal */
export type ZoneGatherColumn = {
  source: GraphDataView;
  destination: GraphDataView;
  /** Bit pattern written at and after the kept count: `0xffffffff` for IDs, `0` for floats. */
  sentinel: string;
};

/** Copies up to three sorted columns to the first `capacity` compact slots. @internal */
export function createZoneGatherNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    keptTotal: GraphDataView<'uint32'>;
    keptRows: GraphDataView<'uint32'>;
    columns: readonly ZoneGatherColumn[];
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'keptTotal', view: props.keptTotal, type: 'u32', access: 'read'},
    {name: 'keptRows', view: props.keptRows, type: 'u32', access: 'read'}
  ];
  const lines: string[] = [];
  for (const [columnIndex, column] of props.columns.entries()) {
    bindings.push(
      {name: `source${columnIndex}`, view: column.source, type: 'u32', access: 'read'},
      {
        name: `destination${columnIndex}`,
        view: column.destination,
        type: 'u32',
        access: 'read_write'
      }
    );
    lines.push(
      `destination${columnIndex}[destination${columnIndex}Offset + index] = select(${column.sentinel}, source${columnIndex}[source${columnIndex}Offset + row], isLive);`
    );
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gather',
    bindings,
    invocationCount: props.capacity,
    body: `let isLive = index < keptTotal[keptTotalOffset];
  let row = select(0u, keptRows[keptRowsOffset + index], isLive);
  ${lines.join('\n  ')}`
  });
}

/** Writes the overflow flag: candidate buffer overflow or a track beyond its event bound. @internal */
export function createZoneOverflowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    flag: GraphDataView<'uint32'>;
  }
): Nodes<Parameters>[number] {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'overflow',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'flag', view: props.flag, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: getTrackConstants(props.shape),
    body: `let candidateOverflow = state[stateOffset] > CANDIDATE_CAPACITY;
  let trackOverflow = state[stateOffset + 2u] != 0u;
  flag[flagOffset] = select(0u, 1u, candidateOverflow || trackOverflow);`
  });
}

/**
 * Writes the split overflow diagnostics: unclamped candidate count, candidate scratch overflow,
 * per-track event bound overflow and event list capacity overflow. Each output is optional.
 * @internal
 */
export function createZoneDiagnosticsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    eventCapacity: number;
    state: GraphDataView<'uint32'>;
    keptTotal: GraphDataView<'uint32'>;
    candidateCount?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
    trackOverflow?: GraphDataView<'uint32'>;
    eventOverflow?: GraphDataView<'uint32'>;
  }
): Nodes<Parameters>[number] {
  const bindings: WGSLKernelBinding[] = [
    {name: 'state', view: props.state, type: 'u32', access: 'read'},
    {name: 'keptTotal', view: props.keptTotal, type: 'u32', access: 'read'}
  ];
  const lines: string[] = [];
  const outputs = [
    ['candidateCount', props.candidateCount, 'state[stateOffset]'],
    [
      'candidateOverflow',
      props.candidateOverflow,
      'select(0u, 1u, state[stateOffset] > CANDIDATE_CAPACITY)'
    ],
    ['trackOverflow', props.trackOverflow, 'select(0u, 1u, state[stateOffset + 2u] != 0u)'],
    [
      'eventOverflow',
      props.eventOverflow,
      'select(0u, 1u, keptTotal[keptTotalOffset] > EVENT_CAPACITY)'
    ]
  ] as const;
  for (const [name, view, expression] of outputs) {
    if (view) {
      bindings.push({name, view, type: 'u32', access: 'read_write'});
      lines.push(`${name}[${name}Offset] = ${expression};`);
    }
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'diagnostics',
    bindings,
    invocationCount: 1,
    declarations: `${getTrackConstants(props.shape)}
const EVENT_CAPACITY: u32 = ${props.eventCapacity}u;`,
    body: lines.join('\n  ')
  });
}
