// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {FIND_PATH_WGSL} from '../line-segmentize/line-segmentize-kernels';

/** Words per point in the internal `GPULinearReferencing` result column. @internal */
export const LINEAR_REFERENCING_RESULT_STRIDE = 8;

/** Words per event in the internal `GPULineLocate` result column. @internal */
export const LINE_LOCATE_RESULT_STRIDE = 6;

/** `GPULineLocate` per-event status codes. */
export const GPU_LINE_LOCATE_STATUS = {
  /** The measure lies on the path. */
  ok: 0,
  /** The measure was outside `[0, length]` and was clamped to the nearest end. */
  clamped: 1,
  /** The path index is out of range or the path is empty; outputs are NaN. */
  invalid: 2
} as const;

const POSITION_WGSL = /* wgsl */ `
const NAN_BITS: u32 = 0x7fc00000u;
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
`;

/**
 * Writes one segment per vertex row: `[row, row + 1]` when both rows belong to the same path,
 * otherwise a NaN segment, which the nearest-feature join gives empty bounds and never matches.
 *
 * @internal
 */
export function createPathSegmentsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    pathOffsets: GraphDataView<'uint32'>;
    segmentStarts: GraphDataView<'float32x2'>;
    segmentEnds: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'path-segments',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
      {name: 'segmentStarts', view: props.segmentStarts, type: 'f32', access: 'read_write'},
      {name: 'segmentEnds', view: props.segmentEnds, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const PATH_COUNT: u32 = ${props.pathOffsets.length - 1}u;
${FIND_PATH_WGSL}
${POSITION_WGSL}`,
    body: /* wgsl */ `let path = findPath(index);
  // A runtime operand keeps the NaN out of const-expression evaluation, where it is an error.
  var start = vec2<f32>(bitcast<f32>(0x7fc00000u | (index & 0u)));
  var end = start;
  if (path != NO_PATH && index + 1u < pathOffsets[pathOffsetsOffset + path + 1u]) {
    start = getPosition(index);
    end = getPosition(index + 1u);
  }
  segmentStarts[segmentStartsOffset + 2u * index] = start.x;
  segmentStarts[segmentStartsOffset + 2u * index + 1u] = start.y;
  segmentEnds[segmentEndsOffset + 2u * index] = end.x;
  segmentEnds[segmentEndsOffset + 2u * index + 1u] = end.y;`
  });
}

/**
 * Projects each point onto its nearest segment and writes the packed result
 * `[pathIndex, segmentRow, fraction, footX, footY, distance, measure, side]` (indices and side as
 * u32/i32 bits).
 *
 * @internal
 */
export function createPointProjectionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    points: GraphDataView<'float32x2'>;
    positions: GraphDataView<'float32x2'>;
    pathOffsets: GraphDataView<'uint32'>;
    nearestSegments: GraphDataView<'uint32'>;
    vertexMeasures: GraphDataView<'float32'>;
    results: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'project',
    bindings: [
      {name: 'points', view: props.points, type: 'f32', access: 'read'},
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
      {name: 'nearestSegments', view: props.nearestSegments, type: 'u32', access: 'read'},
      {name: 'vertexMeasures', view: props.vertexMeasures, type: 'f32', access: 'read'},
      {name: 'results', view: props.results, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.points.length,
    declarations: `const PATH_COUNT: u32 = ${props.pathOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const STRIDE: u32 = ${LINEAR_REFERENCING_RESULT_STRIDE}u;
${FIND_PATH_WGSL}
${POSITION_WGSL}`,
    body: /* wgsl */ `let base = resultsOffset + index * STRIDE;
  let segment = nearestSegments[nearestSegmentsOffset + index];
  // Unmatched points carry the join's 0xffffffff sentinel; it also fails this bound.
  if (segment >= ROW_COUNT - 1u) {
    results[base] = NO_PATH;
    results[base + 1u] = NO_PATH;
    results[base + 2u] = NAN_BITS;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = bitcast<u32>(-1.0);
    results[base + 6u] = NAN_BITS;
    results[base + 7u] = 0u;
    return;
  }
  let point = vec2<f32>(points[pointsOffset + 2u * index], points[pointsOffset + 2u * index + 1u]);
  let start = getPosition(segment);
  let direction = getPosition(segment + 1u) - start;
  let relative = point - start;
  let lengthSquared = dot(direction, direction);
  var fraction = 0.0;
  if (lengthSquared > 0.0) {
    fraction = clamp(dot(relative, direction) / lengthSquared, 0.0, 1.0);
  }
  let footOffset = direction * fraction;
  let distance = length(relative - footOffset);
  let cross = direction.x * relative.y - direction.y * relative.x;
  var side = 0i;
  if (distance > 0.0 && cross > 0.0) {
    side = 1i;
  } else if (distance > 0.0 && cross < 0.0) {
    side = -1i;
  }
  let path = findPath(segment);
  results[base] = path;
  results[base + 1u] = segment - pathOffsets[pathOffsetsOffset + path];
  results[base + 2u] = bitcast<u32>(fraction);
  results[base + 3u] = bitcast<u32>(start.x + footOffset.x);
  results[base + 4u] = bitcast<u32>(start.y + footOffset.y);
  results[base + 5u] = bitcast<u32>(distance);
  results[base + 6u] = bitcast<u32>(vertexMeasures[vertexMeasuresOffset + segment] + fraction * sqrt(lengthSquared));
  results[base + 7u] = bitcast<u32>(side);`
  });
}

/** Output columns of the projection scatter. @internal */
export type PointProjectionColumns = {
  pathIndices?: GraphDataView<'uint32'>;
  segmentIndices?: GraphDataView<'uint32'>;
  fractions?: GraphDataView<'float32'>;
  footPoints?: GraphDataView<'float32x2'>;
  distances?: GraphDataView<'float32'>;
  measures?: GraphDataView<'float32'>;
  sides?: GraphDataView<'sint32'>;
  signedOffsets?: GraphDataView<'float32'>;
};

type ScatterColumn = {
  name: string;
  view: GraphDataView | undefined;
  type: 'f32' | 'u32' | 'i32';
  statement: string;
};

/**
 * Builds up to two scatter nodes copying a packed result column into requested outputs (at most
 * seven outputs per node, within the eight storage bindings).
 *
 * @internal
 */
export function createPackedScatterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rowCount: number;
    stride: number;
    results: GraphDataView<'uint32'>;
    columns: readonly ScatterColumn[];
  }
): GPUCommandNode<Parameters>[] {
  const columns = props.columns.filter(column => column.view);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let first = 0; first < columns.length; first += 7) {
    const group = columns.slice(first, first + 7);
    const bindings: WGSLKernelBinding[] = [
      {name: 'results', view: props.results, type: 'u32', access: 'read'},
      ...group.map(column => ({
        name: column.name,
        view: column.view as GraphDataView,
        type: column.type,
        access: 'read_write' as const
      }))
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: first === 0 ? `${props.id}` : `${props.id}-${first / 7}`,
        operation: props.operation,
        variant: 'scatter',
        bindings,
        invocationCount: props.rowCount,
        declarations: `const STRIDE: u32 = ${props.stride}u;`,
        body: `let base = resultsOffset + index * STRIDE;
  ${group.map(column => column.statement).join('\n  ')}`
      })
    );
  }
  return nodes;
}

/** Scatter columns of `GPULinearReferencing`. @internal */
export function getPointProjectionScatterColumns(columns: PointProjectionColumns): ScatterColumn[] {
  return [
    {
      name: 'pathIndices',
      view: columns.pathIndices,
      type: 'u32',
      statement: 'pathIndices[pathIndicesOffset + index] = results[base];'
    },
    {
      name: 'segmentIndices',
      view: columns.segmentIndices,
      type: 'u32',
      statement: 'segmentIndices[segmentIndicesOffset + index] = results[base + 1u];'
    },
    {
      name: 'fractions',
      view: columns.fractions,
      type: 'f32',
      statement: 'fractions[fractionsOffset + index] = bitcast<f32>(results[base + 2u]);'
    },
    {
      name: 'footPoints',
      view: columns.footPoints,
      type: 'f32',
      statement: `footPoints[footPointsOffset + 2u * index] = bitcast<f32>(results[base + 3u]);
  footPoints[footPointsOffset + 2u * index + 1u] = bitcast<f32>(results[base + 4u]);`
    },
    {
      name: 'distances',
      view: columns.distances,
      type: 'f32',
      statement: 'distances[distancesOffset + index] = bitcast<f32>(results[base + 5u]);'
    },
    {
      name: 'measures',
      view: columns.measures,
      type: 'f32',
      statement: 'measures[measuresOffset + index] = bitcast<f32>(results[base + 6u]);'
    },
    {
      name: 'sides',
      view: columns.sides,
      type: 'i32',
      statement: 'sides[sidesOffset + index] = bitcast<i32>(results[base + 7u]);'
    },
    {
      name: 'signedOffsets',
      view: columns.signedOffsets,
      type: 'f32',
      // Unmatched points copy the NaN fraction instead of multiplying the -1 distance sentinel.
      statement: `signedOffsets[signedOffsetsOffset + index] = select(
    f32(bitcast<i32>(results[base + 7u])) * bitcast<f32>(results[base + 5u]),
    bitcast<f32>(results[base + 2u]),
    results[base] == 0xffffffffu
  );`
    }
  ];
}

/**
 * Locates each event on its path by measure and writes the packed result
 * `[x, y, segmentIndex, tangentX, tangentY, status]` (index and status as u32 bits).
 *
 * @internal
 */
export function createLineLocateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    pathOffsets: GraphDataView<'uint32'>;
    vertexMeasures: GraphDataView<'float32'>;
    eventPaths: GraphDataView<'uint32'>;
    eventMeasures: GraphDataView<'float32'>;
    eventOffsets?: GraphDataView<'float32'>;
    parameters?: GraphDataView<'float32'>;
    measureMode: 'distance' | 'fraction';
    results: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
    {name: 'vertexMeasures', view: props.vertexMeasures, type: 'f32', access: 'read'},
    {name: 'eventPaths', view: props.eventPaths, type: 'u32', access: 'read'},
    {name: 'eventMeasures', view: props.eventMeasures, type: 'f32', access: 'read'},
    {name: 'results', view: props.results, type: 'u32', access: 'read_write'}
  ];
  if (props.eventOffsets) {
    bindings.push({name: 'eventOffsets', view: props.eventOffsets, type: 'f32', access: 'read'});
  }
  if (props.parameters) {
    bindings.push({name: 'parameters', view: props.parameters, type: 'f32', access: 'read'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'locate',
    bindings,
    invocationCount: props.eventPaths.length,
    declarations: `const PATH_COUNT: u32 = ${props.pathOffsets.length - 1}u;
const ROW_COUNT: u32 = ${props.positions.length}u;
const STRIDE: u32 = ${LINE_LOCATE_RESULT_STRIDE}u;
const IS_FRACTION: bool = ${props.measureMode === 'fraction'};
const STATUS_OK: u32 = ${GPU_LINE_LOCATE_STATUS.ok}u;
const STATUS_CLAMPED: u32 = ${GPU_LINE_LOCATE_STATUS.clamped}u;
const STATUS_INVALID: u32 = ${GPU_LINE_LOCATE_STATUS.invalid}u;
${POSITION_WGSL}

fn getMeasure(row: u32) -> f32 {
  return vertexMeasures[vertexMeasuresOffset + row];
}`,
    body: /* wgsl */ `let base = resultsOffset + index * STRIDE;
  let path = eventPaths[eventPathsOffset + index];
  var pathStart = 0u;
  var pathEnd = 0u;
  if (path < PATH_COUNT) {
    pathStart = min(pathOffsets[pathOffsetsOffset + path], ROW_COUNT);
    pathEnd = min(max(pathOffsets[pathOffsetsOffset + path + 1u], pathStart), ROW_COUNT);
  }
  if (pathEnd <= pathStart) {
    results[base] = NAN_BITS;
    results[base + 1u] = NAN_BITS;
    results[base + 2u] = 0xffffffffu;
    results[base + 3u] = NAN_BITS;
    results[base + 4u] = NAN_BITS;
    results[base + 5u] = STATUS_INVALID;
    return;
  }
  let total = getMeasure(pathEnd - 1u);
  var measure = eventMeasures[eventMeasuresOffset + index];
  ${props.parameters ? 'measure = measure * parameters[parametersOffset] + parameters[parametersOffset + 1u];' : ''}
  if (IS_FRACTION) {
    measure = measure * total;
  }
  var status = STATUS_OK;
  if (measure < 0.0) {
    measure = 0.0;
    status = STATUS_CLAMPED;
  } else if (measure > total) {
    measure = total;
    status = STATUS_CLAMPED;
  }
  var position = getPosition(pathStart);
  var tangent = vec2<f32>(0.0);
  var segment = 0u;
  if (pathEnd - pathStart >= 2u) {
    // Upper bound: the first row in (pathStart, pathEnd) whose measure exceeds the target.
    var low = pathStart + 1u;
    var high = pathEnd;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (getMeasure(middle) <= measure) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let segmentRow = min(low, pathEnd - 1u) - 1u;
    let start = getPosition(segmentRow);
    let direction = getPosition(segmentRow + 1u) - start;
    let segmentMeasure = getMeasure(segmentRow + 1u) - getMeasure(segmentRow);
    var fraction = 0.0;
    if (segmentMeasure > 0.0) {
      fraction = clamp((measure - getMeasure(segmentRow)) / segmentMeasure, 0.0, 1.0);
    }
    position = start + direction * fraction;
    let directionLength = length(direction);
    if (directionLength > 0.0) {
      tangent = direction / directionLength;
    }
    segment = segmentRow - pathStart;
  }
  ${
    props.eventOffsets
      ? 'position = position + vec2<f32>(-tangent.y, tangent.x) * eventOffsets[eventOffsetsOffset + index];'
      : ''
  }
  results[base] = bitcast<u32>(position.x);
  results[base + 1u] = bitcast<u32>(position.y);
  results[base + 2u] = segment;
  results[base + 3u] = bitcast<u32>(tangent.x);
  results[base + 4u] = bitcast<u32>(tangent.y);
  results[base + 5u] = status;`
  });
}
