// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {FIND_PATH_WGSL} from '../line-segmentize/line-segmentize-kernels';
import {GPU_LINE_NO_SOURCE} from '../line-segmentize/line-segmentize-types';

/** Number of float32 elements in a `GPURectangleClip` parameter buffer. */
export const GPU_RECTANGLE_CLIP_PARAMETER_LENGTH = 4;

/**
 * WGSL helpers for line clipping over the bound `positions`, `pathOffsets`, `parameters` and the
 * constants `PATH_COUNT` and `ROW_COUNT`: `getClip(row)` returns `(t0, t1, kept)` for the segment
 * starting at `row` (Liang-Barsky), and `isContinuation(row)` whether the clipped segment extends
 * the clipped segment before it (same path, previous kept and unclipped at its end, this one
 * unclipped at its start).
 *
 * @internal
 */
const LINE_CLIP_WGSL = /* wgsl */ `
${FIND_PATH_WGSL}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getClip(row: u32) -> vec3<f32> {
  let path = findPath(row);
  if (path == NO_PATH || row + 1u >= min(pathOffsets[pathOffsetsOffset + path + 1u], ROW_COUNT)) {
    return vec3<f32>(0.0);
  }
  let a = getPosition(row);
  let d = getPosition(row + 1u) - a;
  let lower = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let upper = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  var t0 = 0.0;
  var t1 = 1.0;
  var p = array<f32, 4>(-d.x, d.x, -d.y, d.y);
  var q = array<f32, 4>(a.x - lower.x, upper.x - a.x, a.y - lower.y, upper.y - a.y);
  for (var side = 0u; side < 4u; side++) {
    if (p[side] == 0.0) {
      if (q[side] < 0.0) {
        return vec3<f32>(0.0);
      }
    } else {
      let r = q[side] / p[side];
      if (p[side] < 0.0) {
        if (r > t1) {
          return vec3<f32>(0.0);
        }
        t0 = max(t0, r);
      } else {
        if (r < t0) {
          return vec3<f32>(0.0);
        }
        t1 = min(t1, r);
      }
    }
  }
  // A single touching point is not a segment, except a genuinely zero-length input segment.
  if (t1 < t0 || (t1 == t0 && (d.x != 0.0 || d.y != 0.0))) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(t0, t1, 1.0);
}

fn getPathStart(row: u32) -> u32 {
  return pathOffsets[pathOffsetsOffset + findPath(row)];
}

fn isContinuation(row: u32, clip: vec3<f32>) -> bool {
  if (clip.z == 0.0 || row == 0u || row <= getPathStart(row)) {
    return false;
  }
  let previous = getClip(row - 1u);
  return previous.z != 0.0 && previous.y == 1.0 && clip.x == 0.0;
}

fn getClippedStart(row: u32, clip: vec3<f32>) -> vec2<f32> {
  let a = getPosition(row);
  return select(a + (getPosition(row + 1u) - a) * clip.x, a, clip.x == 0.0);
}

fn getClippedEnd(row: u32, clip: vec3<f32>) -> vec2<f32> {
  let a = getPosition(row);
  let b = getPosition(row + 1u);
  return select(a + (b - a) * clip.y, b, clip.y == 1.0);
}
`;

/** Inputs shared by the line clip kernels. @internal */
export type LineClipInputs = {
  operation: string;
  positions: GraphDataView<'float32x2'>;
  pathOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
};

function getLineClipDeclarations(inputs: LineClipInputs): string {
  return `const PATH_COUNT: u32 = ${inputs.pathOffsets.length - 1}u;
const ROW_COUNT: u32 = ${inputs.positions.length}u;
${LINE_CLIP_WGSL}`;
}

function getLineClipBindings(inputs: LineClipInputs): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: inputs.positions, type: 'f32', access: 'read'},
    {name: 'pathOffsets', view: inputs.pathOffsets, type: 'u32', access: 'read'},
    {name: 'parameters', view: inputs.parameters, type: 'f32', access: 'read'}
  ];
}

/** Per-row output vertex count (0, 1 or 2) and path-start flag of the clipped segments. @internal */
export function createLineClipCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LineClipInputs & {
    id: string;
    counts: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'line-count',
    bindings: [
      ...getLineClipBindings(props),
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'},
      {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: getLineClipDeclarations(props),
    body: `let clip = getClip(index);
  let continues = isContinuation(index, clip);
  counts[countsOffset + index] = select(0u, select(2u, 1u, continues), clip.z != 0.0);
  startFlags[startFlagsOffset + index] = select(0u, 1u, clip.z != 0.0 && !continues);`
  });
}

/** Writes clipped vertices, path starts and optional source paths. @internal */
export function createLineClipEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LineClipInputs & {
    id: string;
    starts: GraphDataView<'uint32'>;
    pathIndices: GraphDataView<'uint32'>;
    outputPositions: GraphDataView<'float32x2'>;
    outputPathOffsets: GraphDataView<'uint32'>;
    outputSourcePaths?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const capacity = props.outputPositions.length;
  const pathCapacity = props.outputPathOffsets.length - 1;
  const bindings: WGSLKernelBinding[] = [
    ...getLineClipBindings(props),
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {name: 'pathIndices', view: props.pathIndices, type: 'u32', access: 'read'},
    {name: 'outPositions', view: props.outputPositions, type: 'f32', access: 'read_write'},
    {name: 'outPathOffsets', view: props.outputPathOffsets, type: 'u32', access: 'read_write'}
  ];
  if (props.outputSourcePaths) {
    bindings.push({
      name: 'outSourcePaths',
      view: props.outputSourcePaths,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'line-emit',
    bindings,
    invocationCount: props.positions.length,
    declarations: `${getLineClipDeclarations(props)}
const CAPACITY: u32 = ${capacity}u;
const PATH_CAPACITY: u32 = ${pathCapacity}u;
fn writeVertex(slot: u32, p: vec2<f32>) {
  if (slot < CAPACITY) {
    outPositions[outPositionsOffset + 2u * slot] = p.x;
    outPositions[outPositionsOffset + 2u * slot + 1u] = p.y;
  }
}`,
    body: `let clip = getClip(index);
  if (clip.z == 0.0) {
    return;
  }
  let continues = isContinuation(index, clip);
  var slot = starts[startsOffset + index];
  if (!continues) {
    let pathIndex = pathIndices[pathIndicesOffset + index];
    if (pathIndex < PATH_CAPACITY) {
      outPathOffsets[outPathOffsetsOffset + pathIndex] = min(slot, CAPACITY);
      ${props.outputSourcePaths ? 'outSourcePaths[outSourcePathsOffset + pathIndex] = findPath(index);' : ''}
    }
    writeVertex(slot, getClippedStart(index, clip));
    slot += 1u;
  }
  writeVertex(slot, getClippedEnd(index, clip));`
  });
}

/** Publishes counts and flags and the unclamped totals to `totals = [vertices, paths]`. @internal */
export function createLineClipPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    capacity: number;
    pathCapacity: number;
    starts: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    pathIndices: GraphDataView<'uint32'>;
    startFlags: GraphDataView<'uint32'>;
    totals: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'pathIndices', view: props.pathIndices, type: 'u32', access: 'read'},
    {name: 'startFlags', view: props.startFlags, type: 'u32', access: 'read'},
    {name: 'totals', view: props.totals, type: 'u32', access: 'read_write'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.requiredCount) {
    bindings.push({name: 'totalOut', view: props.requiredCount, type: 'u32', access: 'read_write'});
  }
  const last = props.starts.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'line-publish',
    bindings,
    invocationCount: 1,
    body: `let vertexTotal = starts[startsOffset + ${last}u] + counts[countsOffset + ${last}u];
  let pathTotal = pathIndices[pathIndicesOffset + ${last}u] + startFlags[startFlagsOffset + ${last}u];
  let pathClamped = min(pathTotal, ${props.pathCapacity}u);
  totals[totalsOffset] = min(vertexTotal, ${props.capacity}u);
  totals[totalsOffset + 1u] = pathClamped;
  countOut[countOutOffset] = min(vertexTotal, ${props.capacity}u);
  overflowOut[overflowOutOffset] = select(0u, 1u, vertexTotal > ${props.capacity}u || pathTotal > ${props.pathCapacity}u);
  ${props.requiredCount ? 'totalOut[totalOutOffset] = vertexTotal;' : ''}`
  });
}

/**
 * Fills the unused tail of `pathOffsets` (paths at or after `totals[1]`) with the vertex count,
 * and optional source paths with `GPU_LINE_NO_SOURCE`.
 *
 * @internal
 */
export function createPathTailNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    totals: GraphDataView<'uint32'>;
    pathOffsets: GraphDataView<'uint32'>;
    sourcePaths?: GraphDataView<'uint32'>;
    pathCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'totals', view: props.totals, type: 'u32', access: 'read'},
    {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read_write'}
  ];
  if (props.pathCount) {
    bindings.push({name: 'pathCountOut', view: props.pathCount, type: 'u32', access: 'read_write'});
  }
  if (props.sourcePaths) {
    bindings.push({
      name: 'sourcePaths',
      view: props.sourcePaths,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'path-tail',
    bindings,
    invocationCount: props.pathOffsets.length,
    body: `${props.pathCount ? 'if (index == 0u) {\n    pathCountOut[pathCountOutOffset] = totals[totalsOffset + 1u];\n  }' : ''}
  if (index >= totals[totalsOffset + 1u]) {
    pathOffsets[pathOffsetsOffset + index] = totals[totalsOffset];
    ${props.sourcePaths ? `if (index < ${props.sourcePaths.length}u) {\n      sourcePaths[sourcePathsOffset + index] = ${GPU_LINE_NO_SOURCE}u;\n    }` : ''}
  }`
  });
}

/** Properties of one Sutherland-Hodgman stage. @internal */
export type PolygonStageInputs = {
  id: string;
  operation: string;
  /** Boundary index: 0 left, 1 right, 2 bottom, 3 top. */
  boundary: number;
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
};

const POLYGON_STAGE_WGSL = /* wgsl */ `
${FIND_PATH_WGSL}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

const BOUND_SLOTS = array<u32, 4>(0u, 2u, 1u, 3u);

fn getBound() -> f32 {
  // left x >= minX, right x <= maxX, bottom y >= minY, top y <= maxY
  return parameters[parametersOffset + BOUND_SLOTS[BOUNDARY]];
}

fn isInside(p: vec2<f32>) -> bool {
  let bound = getBound();
  switch (BOUNDARY) {
    case 0u: { return p.x >= bound; }
    case 1u: { return p.x <= bound; }
    case 2u: { return p.y >= bound; }
    default: { return p.y <= bound; }
  }
}

fn getIntersection(s: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
  let bound = getBound();
  if (BOUNDARY < 2u) {
    let t = (bound - s.x) / (e.x - s.x);
    return vec2<f32>(bound, s.y + t * (e.y - s.y));
  }
  let t = (bound - s.y) / (e.y - s.y);
  return vec2<f32>(s.x + t * (e.x - s.x), bound);
}

// Previous vertex of the ring that owns \`row\`, or false when the row is outside every ring.
fn getRingPrevious(row: u32) -> vec3<f32> {
  let ring = findPath(row);
  if (ring == NO_PATH) {
    return vec3<f32>(0.0);
  }
  let start = pathOffsets[pathOffsetsOffset + ring];
  let end = min(pathOffsets[pathOffsetsOffset + ring + 1u], INPUT_ROWS);
  let previous = getPosition(select(row - 1u, end - 1u, row == start));
  return vec3<f32>(previous, 1.0);
}
`;

function getStageDeclarations(props: PolygonStageInputs): string {
  return `const BOUNDARY: u32 = ${props.boundary}u;
const PATH_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const INPUT_ROWS: u32 = ${props.positions.length}u;
${POLYGON_STAGE_WGSL}`;
}

function getStageBindings(props: PolygonStageInputs): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'pathOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
  ];
}

/** Per-vertex output count (0, 1 or 2) of one Sutherland-Hodgman stage. @internal */
export function createPolygonStageCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PolygonStageInputs & {counts: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'polygon-count',
    bindings: [
      ...getStageBindings(props),
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: getStageDeclarations(props),
    body: `let previous = getRingPrevious(index);
  var count = 0u;
  if (previous.z != 0.0) {
    let endInside = isInside(getPosition(index));
    let startInside = isInside(previous.xy);
    count = select(select(0u, 1u, startInside), select(2u, 1u, startInside), endInside);
  }
  counts[countsOffset + index] = count;`
  });
}

/** Emits the vertices of one Sutherland-Hodgman stage. @internal */
export function createPolygonStageEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PolygonStageInputs & {
    starts: GraphDataView<'uint32'>;
    outputPositions: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  const capacity = props.outputPositions.length;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'polygon-emit',
    bindings: [
      ...getStageBindings(props),
      {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
      {name: 'outPositions', view: props.outputPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `${getStageDeclarations(props)}
const CAPACITY: u32 = ${capacity}u;
fn writeVertex(slot: u32, p: vec2<f32>) {
  if (slot < CAPACITY) {
    outPositions[outPositionsOffset + 2u * slot] = p.x;
    outPositions[outPositionsOffset + 2u * slot + 1u] = p.y;
  }
}`,
    body: `let previous = getRingPrevious(index);
  if (previous.z == 0.0) {
    return;
  }
  let end = getPosition(index);
  let endInside = isInside(end);
  let startInside = isInside(previous.xy);
  var slot = starts[startsOffset + index];
  if (endInside) {
    if (!startInside) {
      writeVertex(slot, getIntersection(previous.xy, end));
      slot += 1u;
    }
    writeVertex(slot, end);
  } else if (startInside) {
    writeVertex(slot, getIntersection(previous.xy, end));
  }`
  });
}

/**
 * Maps the input ring offsets of one stage to output ring offsets through the scanned counts and
 * records the stage's overflow flag.
 *
 * @internal
 */
export function createPolygonStageOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    capacity: number;
    inputOffsets: GraphDataView<'uint32'>;
    starts: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    outputOffsets: GraphDataView<'uint32'>;
    stageFlags: GraphDataView<'uint32'>;
    stage: number;
  }
): GPUCommandNode<Parameters> {
  const rows = props.starts.length;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'polygon-offsets',
    bindings: [
      {name: 'inputOffsets', view: props.inputOffsets, type: 'u32', access: 'read'},
      {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
      {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
      {name: 'outputOffsets', view: props.outputOffsets, type: 'u32', access: 'read_write'},
      {name: 'stageFlags', view: props.stageFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.outputOffsets.length,
    declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${props.capacity}u;`,
    body: `let total = starts[startsOffset + ROWS - 1u] + counts[countsOffset + ROWS - 1u];
  let row = inputOffsets[inputOffsetsOffset + index];
  let mapped = select(total, starts[startsOffset + min(row, ROWS - 1u)], row < ROWS);
  outputOffsets[outputOffsetsOffset + index] = min(mapped, CAPACITY);
  if (index == 0u) {
    stageFlags[stageFlagsOffset + ${props.stage}u] = total;
  }`
  });
}

/** Publishes the polygon result: count, overflow, unclamped total and ring count. @internal */
export function createPolygonPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    capacity: number;
    ringCount: number;
    stageFlags: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    pathCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'stageFlags', view: props.stageFlags, type: 'u32', access: 'read'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.requiredCount) {
    bindings.push({name: 'totalOut', view: props.requiredCount, type: 'u32', access: 'read_write'});
  }
  if (props.pathCount) {
    bindings.push({name: 'pathCountOut', view: props.pathCount, type: 'u32', access: 'read_write'});
  }
  // stageFlags holds each stage's unclamped total; any stage above capacity truncated the result.
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'polygon-publish',
    bindings,
    invocationCount: 1,
    body: `let finalTotal = stageFlags[stageFlagsOffset + 3u];
  var overflow = false;
  for (var stage = 0u; stage < 4u; stage++) {
    overflow = overflow || stageFlags[stageFlagsOffset + stage] > ${props.capacity}u;
  }
  countOut[countOutOffset] = min(finalTotal, ${props.capacity}u);
  overflowOut[overflowOutOffset] = select(0u, 1u, overflow);
  ${props.requiredCount ? 'totalOut[totalOutOffset] = finalTotal;' : ''}
  ${props.pathCount ? `pathCountOut[pathCountOutOffset] = ${props.ringCount}u;` : ''}`
  });
}
