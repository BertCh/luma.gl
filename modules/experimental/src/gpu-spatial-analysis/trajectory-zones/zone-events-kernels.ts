// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUBVH, GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';
import {SPATIAL_JOIN_WGSL_HELPERS} from '../spatial-join/spatial-join-passes';
import {NODE_OVERLAPS_WGSL} from '../../utils/wgsl-common';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../utils/exact-geometry-wgsl';

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
  return timeWordsDifferenceToF32(wordsA, wordsB);
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

/** WGSL interval search over an offsets binding: the interval `i` with `offsets[i] <= value < offsets[i + 1]`. @internal */
function getIntervalSearchSource(functionName: string, offsets: string, count: number): string {
  return `fn ${functionName}(value: u32) -> u32 {
  if (value < ${offsets}[${offsets}Offset] || value >= ${offsets}[${offsets}Offset + ${count}u]) { return NO_INTERVAL; }
  var low = 0u;
  var high = ${count}u;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (${offsets}[${offsets}Offset + middle] > value) { high = middle; } else { low = middle + 1u; }
  }
  return low - 1u;
}`;
}

/**
 * Derives one boundary edge per zone vertex from the four-level polygon layout: the edge from the
 * vertex to the next vertex of its ring (the last vertex closes onto the first), tagged with the
 * feature (zone) index of its ring. Vertices outside every ring, rings with fewer than three
 * vertices, features outside the layout and degenerate edges (a repeated closing vertex) get a NaN
 * start and the zone index `0xffffffff`, which the bounds and crossing kernels ignore. @internal
 */
export function createZoneEdgesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    zones: {
      positions: GraphDataView<'float32x2'>;
      featureOffsets: GraphDataView<'uint32'>;
      polygonOffsets: GraphDataView<'uint32'>;
      ringOffsets: GraphDataView<'uint32'>;
    };
    edgeStarts: GraphDataView<'float32x2'>;
    edgeEnds: GraphDataView<'float32x2'>;
    edgeZones: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {zones} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'edges',
    bindings: [
      {name: 'positions', view: zones.positions, type: 'f32', access: 'read'},
      {name: 'featureOffsets', view: zones.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: zones.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringOffsets', view: zones.ringOffsets, type: 'u32', access: 'read'},
      {name: 'edgeStarts', view: props.edgeStarts, type: 'f32', access: 'read_write'},
      {name: 'edgeEnds', view: props.edgeEnds, type: 'f32', access: 'read_write'},
      {name: 'edgeZones', view: props.edgeZones, type: 'u32', access: 'read_write'}
    ],
    invocationCount: zones.positions.length,
    declarations: `const NO_INTERVAL: u32 = 0xffffffffu;
${getIntervalSearchSource('findRing', 'ringOffsets', zones.ringOffsets.length - 1)}
${getIntervalSearchSource('findPolygon', 'polygonOffsets', zones.polygonOffsets.length - 1)}
${getIntervalSearchSource('findFeature', 'featureOffsets', zones.featureOffsets.length - 1)}`,
    body: `let vertex = index;
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));
  var zone = NO_INTERVAL;
  var start = vec2f(notANumber, notANumber);
  var end = vec2f(notANumber, notANumber);
  let ring = findRing(vertex);
  if (ring != NO_INTERVAL) {
    let ringStart = ringOffsets[ringOffsetsOffset + ring];
    let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
    let polygon = findPolygon(ring);
    var feature = NO_INTERVAL;
    if (polygon != NO_INTERVAL) { feature = findFeature(polygon); }
    if (ringEnd - ringStart >= 3u && feature != NO_INTERVAL) {
      let next = select(vertex + 1u, ringStart, vertex + 1u >= ringEnd);
      let a = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
      let b = vec2f(positions[positionsOffset + next * 2u], positions[positionsOffset + next * 2u + 1u]);
      let isRepeat = bitcast<u32>(a.x) == bitcast<u32>(b.x) && bitcast<u32>(a.y) == bitcast<u32>(b.y);
      if (!isRepeat) {
        zone = feature;
        start = a;
        end = b;
      }
    }
  }
  edgeStarts[edgeStartsOffset + vertex * 2u] = start.x;
  edgeStarts[edgeStartsOffset + vertex * 2u + 1u] = start.y;
  edgeEnds[edgeEndsOffset + vertex * 2u] = end.x;
  edgeEnds[edgeEndsOffset + vertex * 2u + 1u] = end.y;
  edgeZones[edgeZonesOffset + vertex] = zone;`
  });
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
${NODE_OVERLAPS_WGSL}`,
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

/*
 * Boundary rule shared by the segment crossing and the initial-state ray, so that the parity of a
 * track's crossings of a zone always equals the change of its inside flag. It is the library's
 * Simulation-of-Simplicity rule in `utils/exact-geometry-wgsl.ts`: every track sample
 * `P = (x, y)` is read as `P' = (x - e^2, y - e)` and the polygon is not perturbed. Because the
 * whole track moves by the same offset, a boundary touch is decided once and consistently, and all
 * remaining decisions are exact signs from `orientSign` (`sosPointSide`, `sosSegmentSide`):
 *
 * - Ray from `P'` toward +x against edge a-b: it crosses when `a.y >= y` differs from `b.y >= y`
 *   (the edge straddles `y - e`) and `P'` is left of an upward edge or right of a downward edge.
 * - Segment `p'q'` against edge c-d crosses when `p'` and `q'` are on opposite sides of c-d and
 *   `c`, `d` are on opposite sides of `p'q'`. Collinear overlaps give equal ties and so no crossing.
 *
 * Consequences: a track that only touches a polygon vertex or edge from outside has no crossing,
 * one that runs through a vertex into the interior has exactly one, and a track that runs along an
 * edge changes state only where it leaves the edge's line. A zero-length segment never crosses.
 * Where `orientSign` cannot certify a sign (code 2, spans over 200 binary orders) the f32 cross
 * product sign is used.
 */

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
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
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
  // Straddle rule of the perturbed point (see the boundary rule above): a.y >= y differs from b.y >= y.
  if (isRay && ((a.y >= current.y) == (b.y >= current.y))) { return; }
  var previous = current;
  if (!isRay) {
    previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
  }
  // One batch of exact signs for both candidate kinds keeps a single orientSign call site:
  // the ray needs orient(a, b, current); the segment test all four.
  let exact = exactOrientSigns(
    select(4u, 1u, isRay),
    array<vec2f, 4>(a, a, previous, previous),
    array<vec2f, 4>(b, b, current, current),
    array<vec2f, 4>(current, previous, a, b)
  );
  if (isRay) {
    let side = sosPointSide(exact.signs.x, a, b);
    if (select((side < 0), (side > 0), (b.y > a.y))) { candidateKinds[candidateKindsOffset + index] = 2u; }
    return;
  }
  if (sosPointSide(exact.signs.y, a, b) == sosPointSide(exact.signs.x, a, b)) { return; }
  if (sosSegmentSide(exact.signs.z, previous, current) == sosSegmentSide(exact.signs.w, previous, current)) { return; }
  // Only the decision above is exact. The parameter places the event in time.
  let direction = current - previous;
  let edgeDirection = b - a;
  let offset = a - previous;
  let denominator = cross2(direction, edgeDirection);
  var along = 0.0;
  if (denominator != 0.0) {
    along = cross2(offset, edgeDirection) / denominator;
  } else {
    // Parallel track and edge: the perturbed crossing lies at the edge endpoint nearest the track.
    let length2 = dot(direction, direction);
    if (length2 > 0.0) { along = dot(offset, direction) / length2; }
  }
  along = select(0.0, clamp(along, 0.0, 1.0), isFiniteValue(along));
  candidateKinds[candidateKindsOffset + index] = 1u;
  candidateParameters[candidateParametersOffset + index] = along;`
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

/**
 * Writes the cell key `track * ZONE_COUNT + zone` of every event in `(track, time, edge row)`
 * order, and the identity slot it is sorted with. Slots at and after the event count get the
 * key `CELL_COUNT`, which sorts after every cell. @internal
 */
export function createZoneCellKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    sortedTracks: GraphDataView<'uint32'>;
    sortedZones: GraphDataView<'uint32'>;
    cellKeys: GraphDataView<'uint32'>;
    identity: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'cell-keys',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
      {name: 'sortedZones', view: props.sortedZones, type: 'u32', access: 'read'},
      {name: 'cellKeys', view: props.cellKeys, type: 'u32', access: 'read_write'},
      {name: 'identity', view: props.identity, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    declarations: getTrackConstants(props.shape),
    body: `identity[identityOffset + index] = index;
  var key = TRACK_COUNT * ZONE_COUNT;
  if (index < state[stateOffset + 1u]) {
    key = sortedTracks[sortedTracksOffset + index] * ZONE_COUNT + sortedZones[sortedZonesOffset + index];
  }
  cellKeys[cellKeysOffset + index] = key;`
  });
}

/**
 * Event types in parallel. Events of one `(track, zone)` cell are a run of the cell-ordered list
 * (a stable sort by cell of the time-ordered events, so each run is in time order), and the `k`-th
 * event of a run toggles the cell's seeded inside state for the `k`-th time: after it the state is
 * `seed ^ ((k + 1) & 1)`, so it is an enter when that is 1 (the sequential walk's alternation, in
 * one step). Writes the type back to the event's `(track, time, edge row)` slot and gathers the
 * event times into cell order for the dwell sum and the per-cell finalize. @internal
 */
export function createZoneCellEventsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    cellOrder: GraphDataView<'uint32'>;
    sortedCellKeys: GraphDataView<'uint32'>;
    cellOffsets: GraphDataView<'uint32'>;
    initialInside: GraphDataView<'uint32'>;
    sortedTimes: GraphDataView<'float32'>;
    eventTypes: GraphDataView<'uint32'>;
    cellTimes: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'cell-events',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'cellOrder', view: props.cellOrder, type: 'u32', access: 'read'},
      {name: 'sortedCellKeys', view: props.sortedCellKeys, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: props.cellOffsets, type: 'u32', access: 'read'},
      {name: 'initialInside', view: props.initialInside, type: 'u32', access: 'read'},
      {name: 'sortedTimes', view: props.sortedTimes, type: 'f32', access: 'read'},
      {name: 'eventTypes', view: props.eventTypes, type: 'u32', access: 'read_write'},
      {name: 'cellTimes', view: props.cellTimes, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    body: `if (index >= state[stateOffset + 1u]) {
    cellTimes[cellTimesOffset + index] = 0.0;
    return;
  }
  let cell = sortedCellKeys[sortedCellKeysOffset + index];
  let rank = index - cellOffsets[cellOffsetsOffset + cell];
  let after = (initialInside[initialInsideOffset + cell] & 1u) ^ ((rank + 1u) & 1u);
  let slot = cellOrder[cellOrderOffset + index];
  eventTypes[eventTypesOffset + slot] = select(1u, 0u, after == 1u);
  cellTimes[cellTimesOffset + index] = sortedTimes[sortedTimesOffset + slot];`
  });
}

/**
 * Module-scope WGSL of the load-balanced dwell sum: the contribution of the cell-ordered event
 * `item` of cell `row` is `time - enterTime` for an exit, where the enter is the previous event of
 * the run, or time 0 when the cell starts inside and the exit is the run's first event, and 0 for
 * an enter. These are the pair differences the sequential walk added. @internal
 */
export const ZONE_DWELL_CONTRIBUTION_WGSL = /* wgsl */ `
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let rank = item - loadBalancedOffsets[loadBalancedOffsetsOffset + row];
  let seed = initialInside[initialInsideOffset + row] & 1u;
  let after = seed ^ ((rank + 1u) & 1u);
  if (after == 1u) {
    return 0.0;
  }
  var enterTime = 0.0;
  if (rank > 0u) {
    enterTime = cellTimes[cellTimesOffset + item - 1u];
  }
  return cellTimes[cellTimesOffset + item] - enterTime;
}`;

/**
 * Per-event rank inside its track: the distance to the first event of the track's run in the
 * `(track, time, edge row)` order, found by binary search (no per-track walk). Slots at and after
 * the event count get `0xffffffff`. @internal
 */
export function createZoneEventRanksNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    sortedTracks: GraphDataView<'uint32'>;
    eventRanks: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'event-ranks',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
      {name: 'eventRanks', view: props.eventRanks, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    body: `let eventTotal = state[stateOffset + 1u];
  if (index >= eventTotal) {
    eventRanks[eventRanksOffset + index] = 0xffffffffu;
    return;
  }
  let track = sortedTracks[sortedTracksOffset + index];
  var low = 0u;
  var high = index;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    if (sortedTracks[sortedTracksOffset + middle] < track) { low = middle + 1u; } else { high = middle; }
  }
  eventRanks[eventRanksOffset + index] = index - low;`
  });
}

/**
 * Per-track event count from two binary searches over the track-sorted events, and the
 * per-track truncation flag (`state[2]`) when a track has more than `trackEventCapacity` events.
 * @internal
 */
export function createZoneTrackCountsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    sortedTracks: GraphDataView<'uint32'>;
    trackEventCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'track-counts',
    bindings: [
      {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
      {name: 'sortedTracks', view: props.sortedTracks, type: 'u32', access: 'read'},
      {name: 'trackEventCounts', view: props.trackEventCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount,
    declarations: `${getTrackConstants(props.shape)}
// First event position whose track is at least (or, with strict, above) the given track.
fn findTrackBound(track: u32, eventTotal: u32, strict: bool) -> u32 {
  var low = 0u;
  var high = eventTotal;
  loop {
    if (low >= high) { break; }
    let middle = (low + high) / 2u;
    let value = sortedTracks[sortedTracksOffset + middle];
    if (value < track || (strict && value == track)) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
    body: `let eventTotal = atomicLoad(&state[stateOffset + 1u]);
  let count = findTrackBound(index, eventTotal, true) - findTrackBound(index, eventTotal, false);
  trackEventCounts[trackEventCountsOffset + index] = count;
  if (count > MAXIMUM_EVENTS_PER_TRACK) { atomicOr(&state[stateOffset + 2u], 1u); }`
  });
}

/**
 * Per-cell finalize in O(1), replacing the sequential walk: a cell whose run of `L` events starts
 * from the seeded inside state `seed` ends in `seed ^ (L & 1)`, counts `seed` plus its enters
 * (`ceil(L / 2)` when it starts outside, `floor(L / 2)` when inside) as visits, and when it ends
 * inside closes the open visit at the track's last sample: `dwell += duration - enterTime`, with
 * the enter time of the run's last event (0 for a cell that is inside throughout). `dwellTimes`
 * already holds the load-balanced sum of the closed visits. @internal
 */
export function createZoneCellFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    isWordMode: boolean;
    timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    trackOffsets: GraphDataView<'uint32'>;
    cellOffsets: GraphDataView<'uint32'>;
    initialInside: GraphDataView<'uint32'>;
    cellTimes: GraphDataView<'float32'>;
    dwellTimes: GraphDataView<'float32'>;
    visitCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.isWordMode ? 'cell-finalize-words' : 'cell-finalize',
    bindings: [
      {
        name: 'timestamps',
        view: props.timestamps,
        type: props.isWordMode ? 'u32' : 'f32',
        access: 'read'
      },
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: props.cellOffsets, type: 'u32', access: 'read'},
      {name: 'initialInside', view: props.initialInside, type: 'u32', access: 'read'},
      {name: 'cellTimes', view: props.cellTimes, type: 'f32', access: 'read'},
      {name: 'dwellTimes', view: props.dwellTimes, type: 'f32', access: 'read_write'},
      {name: 'visitCounts', view: props.visitCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount * props.shape.zoneCount,
    declarations: `${getTrackConstants(props.shape)}
${getTimeSource(props.isWordMode)}`,
    body: `let start = cellOffsets[cellOffsetsOffset + index];
  let eventCount = cellOffsets[cellOffsetsOffset + index + 1u] - start;
  let seed = initialInside[initialInsideOffset + index] & 1u;
  let enters = select((eventCount + 1u) / 2u, eventCount / 2u, seed == 1u);
  visitCounts[visitCountsOffset + index] = seed + enters;
  if ((seed ^ (eventCount & 1u)) == 0u) { return; }
  var enterTime = 0.0;
  if (eventCount > 0u) { enterTime = cellTimes[cellTimesOffset + start + eventCount - 1u]; }
  let track = index / ZONE_COUNT;
  let firstRow = trackOffsets[trackOffsetsOffset + track];
  let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
  let duration = rowTimeDifference(lastRow, firstRow);
  dwellTimes[dwellTimesOffset + index] = dwellTimes[dwellTimesOffset + index] + (duration - enterTime);`
  });
}

/**
 * Per-cell first enter and last exit in O(1) from the cell's run: the first enter time is 0 for a
 * cell that starts inside, the run's first event for one that starts outside, or -1 when never
 * entered; the last exit is the track duration for a visit still open at the end (the close
 * rule), the run's last event when it ends outside, or 0 without events. @internal
 */
export function createZoneCellSpansNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    isWordMode: boolean;
    timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
    trackOffsets: GraphDataView<'uint32'>;
    cellOffsets: GraphDataView<'uint32'>;
    initialInside: GraphDataView<'uint32'>;
    cellTimes: GraphDataView<'float32'>;
    firstEnterTimes: GraphDataView<'float32'>;
    lastExitTimes: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: props.isWordMode ? 'cell-spans-words' : 'cell-spans',
    bindings: [
      {
        name: 'timestamps',
        view: props.timestamps,
        type: props.isWordMode ? 'u32' : 'f32',
        access: 'read'
      },
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: props.cellOffsets, type: 'u32', access: 'read'},
      {name: 'initialInside', view: props.initialInside, type: 'u32', access: 'read'},
      {name: 'cellTimes', view: props.cellTimes, type: 'f32', access: 'read'},
      {name: 'firstEnterTimes', view: props.firstEnterTimes, type: 'f32', access: 'read_write'},
      {name: 'lastExitTimes', view: props.lastExitTimes, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.shape.trackCount * props.shape.zoneCount,
    declarations: `${getTrackConstants(props.shape)}
${getTimeSource(props.isWordMode)}`,
    body: `let start = cellOffsets[cellOffsetsOffset + index];
  let eventCount = cellOffsets[cellOffsetsOffset + index + 1u] - start;
  let seed = initialInside[initialInsideOffset + index] & 1u;
  var firstEnter = -1.0;
  if (seed == 1u) {
    firstEnter = 0.0;
  } else if (eventCount > 0u) {
    firstEnter = cellTimes[cellTimesOffset + start];
  }
  firstEnterTimes[firstEnterTimesOffset + index] = firstEnter;
  var lastExit = 0.0;
  if ((seed ^ (eventCount & 1u)) == 1u) {
    let track = index / ZONE_COUNT;
    let firstRow = trackOffsets[trackOffsetsOffset + track];
    let lastRow = trackOffsets[trackOffsetsOffset + track + 1u] - 1u;
    lastExit = rowTimeDifference(lastRow, firstRow);
  } else if (eventCount > 0u) {
    lastExit = cellTimes[cellTimesOffset + start + eventCount - 1u];
  }
  lastExitTimes[lastExitTimesOffset + index] = lastExit;`
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

/** One column copied from sorted events to compact output slots. @internal */
export type ZoneGatherColumn = {
  source: GraphDataView;
  destination: GraphDataView;
  /** Bit pattern written at and after the kept count: `0xffffffff` for IDs, `0` for floats. */
  sentinel: string;
  /** Words copied per event, 2 for a `float32x2` column. Defaults to 1. */
  stride?: number;
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
    const stride = column.stride ?? 1;
    for (let word = 0; word < stride; word++) {
      lines.push(
        `destination${columnIndex}[destination${columnIndex}Offset + index * ${stride}u + ${word}u] = select(${column.sentinel}, source${columnIndex}[source${columnIndex}Offset + row * ${stride}u + ${word}u], isLive);`
      );
    }
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

/**
 * Interpolated crossing position of every sorted event: `previous + along * (current - previous)`
 * on the segment ending at the event's row. Slots at and after the event count are zero. @internal
 */
export function createZoneSortedPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    shape: ZoneEventsShape;
    state: GraphDataView<'uint32'>;
    order: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    candidateParameters: GraphDataView<'float32'>;
    positions: GraphDataView<'float32x2'>;
    sortedPositions: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'sorted-positions',
    bindings: [
      {name: 'state', view: props.state, type: 'u32', access: 'read'},
      {name: 'order', view: props.order, type: 'u32', access: 'read'},
      {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
      {name: 'candidateParameters', view: props.candidateParameters, type: 'f32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'sortedPositions', view: props.sortedPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.shape.candidateCapacity,
    body: `var crossing = vec2f(0.0, 0.0);
  if (index < state[stateOffset + 1u]) {
    let slot = order[orderOffset + index];
    let row = candidatePairs[candidatePairsOffset + slot * 2u];
    let along = candidateParameters[candidateParametersOffset + slot];
    let previous = vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u]);
    let current = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
    crossing = previous + along * (current - previous);
  }
  sortedPositions[sortedPositionsOffset + index * 2u] = crossing.x;
  sortedPositions[sortedPositionsOffset + index * 2u + 1u] = crossing.y;`
  });
}

/** Flags every `(track, zone)` cell with at least one visit and writes the identity cell ids. @internal */
export function createZoneTableFlagsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    visitCounts: GraphDataView<'uint32'>;
    flags: GraphDataView<'uint32'>;
    cellIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'table-flags',
    bindings: [
      {name: 'visitCounts', view: props.visitCounts, type: 'u32', access: 'read'},
      {name: 'flags', view: props.flags, type: 'u32', access: 'read_write'},
      {name: 'cellIds', view: props.cellIds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `cellIds[cellIdsOffset + index] = index;
  flags[flagsOffset + index] = select(0u, 1u, visitCounts[visitCountsOffset + index] > 0u);`
  });
}

/** One column of the `(track, zone)` table. @internal */
export type ZoneTableColumn = {
  /** `'track'` and `'zone'` derive from the cell index; `'value'` copies `source[cell]`. */
  kind: 'track' | 'zone' | 'value';
  /** Dense per-cell source for `'value'` columns. */
  source?: GraphDataView<'uint32'> | GraphDataView<'float32'>;
  destination: GraphDataView<'uint32'> | GraphDataView<'float32'>;
};

/** Number of storage bindings a table column needs. @internal */
export function getZoneTableColumnBindingCount(column: ZoneTableColumn): number {
  return column.kind === 'value' ? 2 : 1;
}

/**
 * Copies table columns for the compacted cells. Slots at and after the kept count receive
 * sentinels (`0xffffffff` for integers, 0 for floats). @internal
 */
export function createZoneTableGatherNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    zoneCount: number;
    keptTotal: GraphDataView<'uint32'>;
    keptCells: GraphDataView<'uint32'>;
    columns: readonly ZoneTableColumn[];
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'keptTotal', view: props.keptTotal, type: 'u32', access: 'read'},
    {name: 'keptCells', view: props.keptCells, type: 'u32', access: 'read'}
  ];
  const lines: string[] = [];
  for (const [columnIndex, column] of props.columns.entries()) {
    const type = column.destination.format === 'float32' ? 'f32' : 'u32';
    const sentinel = type === 'f32' ? '0.0' : '0xffffffffu';
    bindings.push({
      name: `destination${columnIndex}`,
      view: column.destination,
      type,
      access: 'read_write'
    });
    let value: string;
    if (column.kind === 'value' && column.source) {
      bindings.push({name: `source${columnIndex}`, view: column.source, type, access: 'read'});
      value = `source${columnIndex}[source${columnIndex}Offset + cell]`;
    } else {
      value = column.kind === 'track' ? 'cell / ZONE_COUNT' : 'cell % ZONE_COUNT';
    }
    lines.push(
      `destination${columnIndex}[destination${columnIndex}Offset + index] = select(${sentinel}, ${value}, isLive);`
    );
  }
  // The caller chunks columns so that a kernel never needs more than the 8 storage bindings of a
  // core device, see getZoneTableColumnBindingCount.
  assert(bindings.length <= 8);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'table-gather',
    bindings,
    invocationCount: props.capacity,
    declarations: `const ZONE_COUNT: u32 = ${props.zoneCount}u;`,
    body: `let isLive = index < keptTotal[keptTotalOffset];
  let cell = select(0u, keptCells[keptCellsOffset + index], isLive);
  ${lines.join('\n  ')}`
  });
}
