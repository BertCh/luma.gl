// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUSpatialEdgeModel} from '../../analysis-core/contracts';
import {LON_LAT_RING_WGSL} from '../../analysis-core/lon-lat-ring-wgsl';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

type Words = GraphDataView<'uint32'>;
type Positions = GraphDataView<'float32x2'>;
type Turns = GraphDataView<'sint32'>;
const OPERATION = 'GPULonLatPolygonPreparation';
function read(
  name: string,
  view: GraphDataView,
  type: 'u32' | 'i32' | 'f32' = 'u32'
): WGSLKernelBinding {
  return {name, view, type, access: 'read'};
}
function write(
  name: string,
  view: GraphDataView,
  type: 'u32' | 'i32' | 'f32' | 'atomic<u32>' = 'u32'
): WGSLKernelBinding {
  return {name, view, type, access: 'read_write'};
}

/** Initializes per-encoding counters and the constant clipping rectangle. @internal */
export function createPreparationInitializeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    diagnostics: Words;
    rectangle: GraphDataView<'float32'>;
    stageFlags: Words;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'initialize',
    bindings: [
      write('diagnostics', props.diagnostics),
      write('rectangle', props.rectangle, 'f32'),
      write('stageFlags', props.stageFlags)
    ],
    invocationCount: 1,
    body: `for (var stage = 0u; stage < 6u; stage++) { stageFlags[stageFlagsOffset + stage] = 0u; }
  diagnostics[diagnosticsOffset] = 0u;
  diagnostics[diagnosticsOffset + 1u] = 0u;
  rectangle[rectangleOffset] = -180.0;
  rectangle[rectangleOffset + 1u] = -90.0;
  rectangle[rectangleOffset + 2u] = 180.0;
  rectangle[rectangleOffset + 3u] = 90.0;`
  });
}

/** Classifies original edges and the densified latitude range without storing unwrapped floats. @internal */
export function createPreparationClassifyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: Positions;
    ringOffsets: Words;
    densePositions: Positions;
    denseOffsets: Words;
    parameters: GraphDataView<'float32'>;
    metadata: Words;
    counts: Words;
    diagnostics: Words;
    edgeModel: GPUSpatialEdgeModel;
    maximumPieceCount: number;
    antimeridian: 'split' | 'reject';
    poleRings: 'cap' | 'reject';
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'classify',
    bindings: [
      read('positions', props.positions, 'f32'),
      read('ringOffsets', props.ringOffsets),
      read('densePositions', props.densePositions, 'f32'),
      read('denseOffsets', props.denseOffsets),
      read('parameters', props.parameters, 'f32'),
      write('metadata', props.metadata),
      write('counts', props.counts),
      write('diagnostics', props.diagnostics, 'atomic<u32>')
    ],
    invocationCount: props.ringOffsets.length - 1,
    declarations: `${LON_LAT_RING_WGSL}\n${GEODESIC_WGSL}
fn position(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn finitePosition(value: vec2<f32>) -> bool {
  return all((bitcast<vec2<u32>>(value) & vec2<u32>(0x7f800000u)) != vec2<u32>(0x7f800000u));
}`,
    body: `let begin = ringOffsets[ringOffsetsOffset + index];
  var end = ringOffsets[ringOffsetsOffset + index + 1u];
  var flags = 0u;
  var invalid = false;
  var turn = 0i;
  var winding = 0.0;
  var pole = 0.0;
  var hasSecond = false;
  var hasThird = false;
  let denseBegin = denseOffsets[denseOffsetsOffset + index];
  let denseEnd = denseOffsets[denseOffsetsOffset + index + 1u];
  if (end > begin) {
    let first = position(begin);
    var second = first;
    if (end > begin + 1u && all(first == position(end - 1u))) { end--; }
    var minimum = first.x;
    var maximum = first.x;
    var longitude = first.x;
    for (var row = begin; row < end; row++) {
      let current = position(row);
      let next = position(select(row + 1u, begin, row + 1u >= end));
      if (!finitePosition(current) || abs(current.y) > 90.0) { flags |= 16u; continue; }
      if (any(current != first)) {
        if (!hasSecond) { second = current; hasSecond = true; }
        else if (any(current != second)) { hasThird = true; }
      }
      if (!finitePosition(next)) { continue; }
      if (lonLatIsCrossingEdge(current, next)) { flags |= 1u; }
      if (${props.edgeModel === 'linear' ? 'lonLatIsAmbiguousLinearEdge' : 'lonLatIsAmbiguousGreatCircleEdge'}(current, next)) { flags |= 4u; }
      let step = lonLatRingStep(current, next);
      winding += step;
      if (!lonLatIsPoleEdge(current, next)) { turn -= i32(round((next.x - current.x) / 360.0)); }
      longitude = next.x + 360.0 * f32(turn);
      minimum = min(minimum, longitude);
      maximum = max(maximum, longitude);
      let maximumStep = parameters[parametersOffset];
      let extent = max(abs(lonLatWrapLongitudeStep(next.x - current.x)), geodesicCentralAngle(current, next) * GEODESIC_RADIANS_TO_DEGREES);
      if (maximumStep > 0.0 && ceil(extent / maximumStep) > ${props.maximumPieceCount}.0) { flags |= 32u; }
    }
    if (!hasThird) { flags &= 16u; winding = 0.0; }
    if (hasThird && maximum - minimum > 360.0) { flags |= 8u; }
    invalid = (flags & 28u) != 0u ${props.antimeridian === 'reject' ? '|| (flags & 1u) != 0u' : ''};
    if (lonLatIsPoleWinding(winding)) {
      var minimumLatitude = 90.0;
      var maximumLatitude = -90.0;
      for (var row = denseBegin; row < denseEnd; row++) {
        let latitude = densePositions[densePositionsOffset + 2u * row + 1u];
        minimumLatitude = min(minimumLatitude, latitude);
        maximumLatitude = max(maximumLatitude, latitude);
      }
      pole = sign(minimumLatitude + maximumLatitude) * 90.0;
      invalid = invalid || pole == 0.0 ${props.poleRings === 'reject' ? '|| true' : ''};
      if (!invalid) { flags |= 2u; }
    }
  }
  metadata[metadataOffset + 4u * index] = flags;
  metadata[metadataOffset + 4u * index + 1u] = select(0u, 1u, invalid);
  metadata[metadataOffset + 4u * index + 2u] = bitcast<u32>(pole);
  metadata[metadataOffset + 4u * index + 3u] = bitcast<u32>(turn);
  counts[countsOffset + index] = select(0u, denseEnd - denseBegin + select(0u, 3u, (flags & 2u) != 0u), hasThird && !invalid);
  ${
    props.edgeModel === 'great-circle'
      ? `if (parameters[parametersOffset] == 0.0 || (flags & 32u) != 0u) {
    atomicStore(&diagnostics[diagnosticsOffset + 1u], 1u);
  }`
      : ''
  }`
  });
}

/** Rejects complete features, counting each once, and zeros every affected ring count. @internal */
export function createPreparationFeatureNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    featureOffsets: Words;
    polygonOffsets: Words;
    metadata: Words;
    counts: Words;
    diagnostics: Words;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'feature-validity',
    bindings: [
      read('featureOffsets', props.featureOffsets),
      read('polygonOffsets', props.polygonOffsets),
      read('metadata', props.metadata),
      write('counts', props.counts),
      write('diagnostics', props.diagnostics, 'atomic<u32>')
    ],
    invocationCount: props.featureOffsets.length - 1,
    body: `let begin = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index]];
  let end = polygonOffsets[polygonOffsetsOffset + featureOffsets[featureOffsetsOffset + index + 1u]];
  var invalid = false;
  for (var ring = begin; ring < end; ring++) { invalid = invalid || metadata[metadataOffset + 4u * ring + 1u] != 0u; }
  if (invalid) {
    atomicAdd(&diagnostics[diagnosticsOffset], 1u);
    for (var ring = begin; ring < end; ring++) { counts[countsOffset + ring] = 0u; }
  } else {
    // An empty shell has no filled area; its holes cannot become standalone polygons.
    for (var polygon = featureOffsets[featureOffsetsOffset + index]; polygon < featureOffsets[featureOffsetsOffset + index + 1u]; polygon++) {
      let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
      if (firstRing < lastRing && counts[countsOffset + firstRing] == 0u) {
        for (var ring = firstRing + 1u; ring < lastRing; ring++) { counts[countsOffset + ring] = 0u; }
      }
    }
  }`
  });
}

/** Emits capacity-bounded dense vertices, then the closing endpoint and two pole-cap vertices. @internal */
export function createPreparationCapNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: Positions;
    turns: Turns;
    offsets: Words;
    metadata: Words;
    starts: Words;
    counts: Words;
    outputPositions: Positions;
    outputTurns: Turns;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'cap',
    bindings: [
      read('positions', props.positions, 'f32'),
      read('turns', props.turns, 'i32'),
      read('offsets', props.offsets),
      read('metadata', props.metadata),
      read('starts', props.starts),
      read('counts', props.counts),
      write('outputPositions', props.outputPositions, 'f32'),
      write('outputTurns', props.outputTurns, 'i32')
    ],
    invocationCount: props.counts.length,
    declarations: `fn copyVertex(destination: u32, source: u32, latitude: f32, turn: i32) {
  if (destination >= ${props.outputPositions.length}u) { return; }
  outputPositions[outputPositionsOffset + 2u * destination] = positions[positionsOffset + 2u * source];
  outputPositions[outputPositionsOffset + 2u * destination + 1u] = latitude;
  outputTurns[outputTurnsOffset + destination] = turn;
}`,
    body: `if (counts[countsOffset + index] == 0u) { return; }
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let start = starts[startsOffset + index];
  for (var row = begin; row < end; row++) {
    copyVertex(start + row - begin, row, positions[positionsOffset + 2u * row + 1u], turns[turnsOffset + row]);
  }
  if ((metadata[metadataOffset + 4u * index] & 2u) != 0u) {
    let firstTurn = turns[turnsOffset + begin];
    let closureTurn = firstTurn + bitcast<i32>(metadata[metadataOffset + 4u * index + 3u]);
    let pole = bitcast<f32>(metadata[metadataOffset + 4u * index + 2u]);
    copyVertex(start + end - begin, begin, positions[positionsOffset + 2u * begin + 1u], closureTurn);
    copyVertex(start + end - begin + 1u, begin, pole, closureTurn);
    copyVertex(start + end - begin + 2u, begin, pole, firstTurn);
  }`
  });
}

/** Clamps ring offsets and optionally records the unclamped stage total for overflow reporting. @internal */
export function createPreparationOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    starts: Words;
    counts: Words;
    offsets: Words;
    capacity: number;
    stageFlags?: Words;
    stage?: number;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'ring-offsets',
    bindings: [
      read('starts', props.starts),
      read('counts', props.counts),
      write('offsets', props.offsets),
      ...(props.stageFlags ? [write('stageFlags', props.stageFlags)] : [])
    ],
    invocationCount: props.counts.length + 1,
    body: `var value = 0u;
  if (index < ${props.counts.length}u) { value = starts[startsOffset + index]; }
  else if (index > 0u) { value = starts[startsOffset + index - 1u] + counts[countsOffset + index - 1u]; }
  offsets[offsetsOffset + index] = min(value, ${props.capacity}u);
  ${props.stageFlags ? `if (index == ${props.counts.length}u) { stageFlags[stageFlagsOffset + ${props.stage ?? 0}u] = value; }` : ''}`
  });
}

/** Normalizes shells and aligns holes using whole-turn integer shifts only. @internal */
export function createPreparationNormalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: Positions;
    turns: Turns;
    ringOffsets: Words;
    polygonOffsets: Words;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'normalize',
    bindings: [
      read('positions', props.positions, 'f32'),
      write('turns', props.turns, 'i32'),
      read('ringOffsets', props.ringOffsets),
      read('polygonOffsets', props.polygonOffsets)
    ],
    invocationCount: props.polygonOffsets.length - 1,
    declarations: `fn longitude(row: u32) -> f32 { return positions[positionsOffset + 2u * row] + 360.0 * f32(turns[turnsOffset + row]); }`,
    body: `let shell = polygonOffsets[polygonOffsetsOffset + index];
  let ringEnd = polygonOffsets[polygonOffsetsOffset + index + 1u];
  if (shell == ringEnd) { return; }
  let begin = ringOffsets[ringOffsetsOffset + shell];
  let end = ringOffsets[ringOffsetsOffset + shell + 1u];
  if (begin == end) { return; }
  var minimum = longitude(begin);
  var maximum = minimum;
  for (var row = begin + 1u; row < end; row++) { minimum = min(minimum, longitude(row)); maximum = max(maximum, longitude(row)); }
  let shellTurns = i32(floor((minimum + 180.0) / 360.0));
  let centre = (minimum + maximum) * 0.5;
  for (var ring = shell; ring < ringEnd; ring++) {
    let first = ringOffsets[ringOffsetsOffset + ring];
    let last = ringOffsets[ringOffsetsOffset + ring + 1u];
    if (first == last) { continue; }
    var shift = -shellTurns;
    if (ring != shell) { shift += i32(floor((centre - longitude(first)) / 360.0 + 0.5)); }
    for (var row = first; row < last; row++) { turns[turnsOffset + row] += shift; }
  }`
  });
}

/** Writes feature-contiguous main/east polygon topology and a source-ring/copy map. @internal */
export function createPreparationTopologyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    featureOffsets: Words;
    polygonOffsets: Words;
    outputFeatureOffsets: Words;
    outputPolygonOffsets: Words;
    ringMap: Words;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'topology',
    bindings: [
      read('featureOffsets', props.featureOffsets),
      read('polygonOffsets', props.polygonOffsets),
      write('outputFeatureOffsets', props.outputFeatureOffsets),
      write('outputPolygonOffsets', props.outputPolygonOffsets),
      write('ringMap', props.ringMap)
    ],
    invocationCount: props.featureOffsets.length,
    body: `let first = featureOffsets[featureOffsetsOffset + index];
  outputFeatureOffsets[outputFeatureOffsetsOffset + index] = 2u * first;
  if (index == ${props.featureOffsets.length - 1}u) {
    outputPolygonOffsets[outputPolygonOffsetsOffset + 2u * first] = 2u * polygonOffsets[polygonOffsetsOffset + first];
    return;
  }
  let last = featureOffsets[featureOffsetsOffset + index + 1u];
  let firstRing = polygonOffsets[polygonOffsetsOffset + first];
  let lastRing = polygonOffsets[polygonOffsetsOffset + last];
  for (var copy = 0u; copy < 2u; copy++) {
    for (var polygon = first; polygon < last; polygon++) {
      let targetPolygon = 2u * first + copy * (last - first) + polygon - first;
      let sourceRing = polygonOffsets[polygonOffsetsOffset + polygon];
      let targetRing = 2u * firstRing + copy * (lastRing - firstRing) + sourceRing - firstRing;
      outputPolygonOffsets[outputPolygonOffsetsOffset + targetPolygon] = targetRing;
      for (var ring = sourceRing; ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u]; ring++) {
        ringMap[ringMapOffset + targetRing + ring - sourceRing] = 2u * ring + copy;
      }
    }
  }`
  });
}

const MERGE_WGSL = `
fn getPosition(row: u32) -> vec2<f32> {
  var longitude = positions[positionsOffset + 2u * row];
  let turn = turns[turnsOffset + row];
  if (turn != 0i) { longitude += 360.0 * f32(turn); }
  return vec2<f32>(longitude, positions[positionsOffset + 2u * row + 1u]);
}
fn ringEnd(begin: u32, last: u32) -> u32 {
  var end = last;
  if (end > begin + 1u && all(getPosition(begin) == getPosition(end - 1u))) { end--; }
  return end;
}
fn keepRow(row: u32, begin: u32) -> bool {
  if (row == begin) { return true; }
  return any(getPosition(row) != getPosition(row - 1u));
}`;

/** Counts nonduplicate clipped vertices, dropping seam-only rings with fewer than three positions. @internal */
export function createPreparationMergeCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: Positions;
    turns: Turns;
    offsets: Words;
    ringMap: Words;
    counts: Words;
    copy: number;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `merge-count-${props.copy}`,
    bindings: [
      read('positions', props.positions, 'f32'),
      read('turns', props.turns, 'i32'),
      read('offsets', props.offsets),
      read('ringMap', props.ringMap),
      write('counts', props.counts)
    ],
    invocationCount: props.counts.length,
    declarations: MERGE_WGSL,
    body: `let mapping = ringMap[ringMapOffset + index];
  if (mapping % 2u != ${props.copy}u) { return; }
  let ring = mapping / 2u;
  let begin = offsets[offsetsOffset + ring];
  let end = ringEnd(begin, offsets[offsetsOffset + ring + 1u]);
  var count = 0u;
  var second = vec2<f32>(0.0);
  var hasSecond = false;
  var hasThird = false;
  for (var row = begin; row < end; row++) {
    let current = getPosition(row);
    if (any(current != getPosition(begin))) {
      if (!hasSecond) { second = current; hasSecond = true; }
      else if (any(current != second)) { hasThird = true; }
    }
    if (keepRow(row, begin)) { count++; }
  }
  counts[countsOffset + index] = select(0u, count, hasThird);`
  });
}

/** Merges one copy into scanned, capacity-bounded output positions in deterministic order. @internal */
export function createPreparationMergeEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: Positions;
    turns: Turns;
    offsets: Words;
    ringMap: Words;
    counts: Words;
    starts: Words;
    output: Positions;
    copy: number;
  }
) {
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `merge-emit-${props.copy}`,
    bindings: [
      read('positions', props.positions, 'f32'),
      read('turns', props.turns, 'i32'),
      read('offsets', props.offsets),
      read('ringMap', props.ringMap),
      read('counts', props.counts),
      read('starts', props.starts),
      write('output', props.output, 'f32')
    ],
    invocationCount: props.counts.length,
    declarations: MERGE_WGSL,
    body: `let mapping = ringMap[ringMapOffset + index];
  if (mapping % 2u != ${props.copy}u || counts[countsOffset + index] == 0u) { return; }
  let ring = mapping / 2u;
  let begin = offsets[offsetsOffset + ring];
  let end = ringEnd(begin, offsets[offsetsOffset + ring + 1u]);
  var destination = starts[startsOffset + index];
  for (var row = begin; row < end; row++) {
    if (keepRow(row, begin)) {
      if (destination < ${props.output.length}u) {
        let position = getPosition(row);
        output[outputOffset + 2u * destination] = position.x;
        output[outputOffset + 2u * destination + 1u] = position.y;
      }
      destination++;
    }
  }`
  });
}

/**
 * Aggregates every stage's overflow and writes exact or conservative totals for the publisher.
 * Each longitude boundary doubles at most; two copies need at most 8 * (dense count + 3R).
 * @internal
 */
export function createPreparationStatusNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    counts: Words;
    starts: Words;
    stageFlags: Words;
    capacity: number;
    metadata: Words;
    diagnostics: Words;
    ringFlags?: Words;
    invalidCount?: Words;
    approximation?: Words;
  }
) {
  const bindings = [
    read('counts', props.counts),
    read('starts', props.starts),
    write('stageFlags', props.stageFlags),
    read('metadata', props.metadata),
    read('diagnostics', props.diagnostics)
  ];
  for (const name of ['ringFlags', 'invalidCount', 'approximation'] as const) {
    if (props[name]) {
      bindings.push(write(name, props[name]));
    }
  }
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'status',
    bindings,
    invocationCount: Math.max(1, props.metadata.length / 4),
    body: `${props.ringFlags ? 'ringFlags[ringFlagsOffset + index] = metadata[metadataOffset + 4u * index];' : ''}
  if (index == 0u) {
    let last = ${props.counts.length - 1}u;
    let total = starts[startsOffset + last] + counts[countsOffset + last];
    let denseTotal = stageFlags[stageFlagsOffset];
    var intermediateTotal = 0u;
    for (var stage = 0u; stage < 6u; stage++) {
      intermediateTotal = max(intermediateTotal, stageFlags[stageFlagsOffset + stage]);
    }
    let requiredCount = select(total, 8u * (denseTotal + ${(3 * props.metadata.length) / 4}u), intermediateTotal > ${props.capacity}u);
    stageFlags[stageFlagsOffset] = intermediateTotal;
    stageFlags[stageFlagsOffset + 1u] = 0u;
    stageFlags[stageFlagsOffset + 2u] = 0u;
    stageFlags[stageFlagsOffset + 3u] = total;
    stageFlags[stageFlagsOffset + 4u] = requiredCount;
    ${props.invalidCount ? 'invalidCount[invalidCountOffset] = diagnostics[diagnosticsOffset];' : ''}
    ${props.approximation ? 'approximation[approximationOffset] = diagnostics[diagnosticsOffset + 1u];' : ''}
  }`
  });
}
