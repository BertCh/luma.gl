// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import type {GPULineCoordinateSystem} from './line-segmentize-types';

/**
 * WGSL `findPath(row) -> u32`: upper-bound binary search over the bound `pathOffsets`
 * (`PATH_COUNT + 1` rows). Returns the path owning `row`, or `NO_PATH` when the row lies outside
 * every path. Empty paths are skipped naturally.
 *
 * @internal
 */
export const FIND_PATH_WGSL = /* wgsl */ `
const NO_PATH: u32 = 0xffffffffu;

fn findPath(row: u32) -> u32 {
  var low = 0u;
  var high = PATH_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pathOffsets[pathOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > PATH_COUNT) {
    return NO_PATH;
  }
  return low - 1u;
}
`;

/**
 * WGSL segment helpers over the bound `positions` (f32 pairs) for one coordinate system:
 * `getPosition(row)`, `getSegmentLength(a, b)` in output units (planar units or radius units).
 *
 * @internal
 */
export function getLineGeometrySource(
  coordinateSystem: GPULineCoordinateSystem,
  radius: number
): string {
  const lengthSource =
    coordinateSystem === 'spherical'
      ? `return geodesicCentralAngle(a, b) * RADIUS;`
      : `return length(b - a);`;
  return /* wgsl */ `
${coordinateSystem === 'spherical' ? GEODESIC_WGSL : ''}
const RADIUS: f32 = ${getWGSLFloatLiteral(radius)};

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  ${lengthSource}
}
`;
}

/** Properties for {@link createPathPrefixNode}. @internal */
export type PathPrefixNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  pathOffsets: GraphDataView<'uint32'>;
  coordinateSystem: GPULineCoordinateSystem;
  radius: number;
  /** Per-row measure of the row's vertex from its path start. Rows outside paths are untouched. */
  rowMeasures?: GraphDataView<'float32'>;
  /**
   * Per-row longitude shift (a multiple of 360) that makes longitudes continuous along the path.
   * Spherical only.
   */
  rowShifts?: GraphDataView<'float32'>;
};

/**
 * Builds the per-path sequential prefix: cumulative measures (Neumaier-compensated f32 sum in row
 * order, so the result is deterministic) and longitude unwrapping shifts. One invocation per path.
 *
 * @internal
 */
export function createPathPrefixNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PathPrefixNodeProps
): GPUCommandNode<Parameters> {
  const pathCount = props.pathOffsets.length - 1;
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {
      name: 'pathOffsets',
      view: props.pathOffsets,
      type: 'u32',
      access: 'read'
    }
  ];
  if (props.rowMeasures) {
    bindings.push({
      name: 'rowMeasures',
      view: props.rowMeasures,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.rowShifts) {
    bindings.push({
      name: 'rowShifts',
      view: props.rowShifts,
      type: 'f32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'path-prefix',
    bindings,
    invocationCount: pathCount,
    declarations: `const ROW_COUNT: u32 = ${props.positions.length}u;
${getLineGeometrySource(props.coordinateSystem, props.radius)}`,
    body: /* wgsl */ `let pathStart = min(pathOffsets[pathOffsetsOffset + index], ROW_COUNT);
  let pathEnd = min(max(pathOffsets[pathOffsetsOffset + index + 1u], pathStart), ROW_COUNT);
  if (pathEnd > pathStart) {
    var sum = 0.0;
    var compensation = 0.0;
    var shift = 0.0;
    var previous = getPosition(pathStart);
    ${props.rowMeasures ? 'rowMeasures[rowMeasuresOffset + pathStart] = 0.0;' : ''}
    ${props.rowShifts ? 'rowShifts[rowShiftsOffset + pathStart] = 0.0;' : ''}
    for (var row = pathStart + 1u; row < pathEnd; row++) {
      let current = getPosition(row);
      ${
        props.rowMeasures
          ? `let segmentLength = getSegmentLength(previous, current);
      let nextSum = sum + segmentLength;
      if (abs(sum) >= abs(segmentLength)) {
        compensation = compensation + ((sum - nextSum) + segmentLength);
      } else {
        compensation = compensation + ((segmentLength - nextSum) + sum);
      }
      sum = nextSum;
      rowMeasures[rowMeasuresOffset + row] = sum + compensation;`
          : ''
      }
      ${
        props.rowShifts
          ? `shift = shift - 360.0 * round((current.x - previous.x) / 360.0);
      rowShifts[rowShiftsOffset + row] = shift;`
          : ''
      }
      previous = current;
    }
  }`
  });
}

/** Properties for {@link createSegmentizeCountNode}. @internal */
export type SegmentizeCountNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  pathOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  coordinateSystem: GPULineCoordinateSystem;
  radius: number;
  maximumPiecesPerSegment: number;
  /** Per-row output vertex count. */
  counts: GraphDataView<'uint32'>;
};

/**
 * Per-row output vertex count of `GPULineSegmentize`: `0` outside paths, `1` for the last row of a
 * path, otherwise `clamp(ceil(length / maximumSegmentLength), 1, maximumPiecesPerSegment)`.
 *
 * @internal
 */
export function createSegmentizeCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SegmentizeCountNodeProps
): GPUCommandNode<Parameters> {
  const pathCount = props.pathOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'count',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {
        name: 'pathOffsets',
        view: props.pathOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const PATH_COUNT: u32 = ${pathCount}u;
const MAXIMUM_PIECES: u32 = ${props.maximumPiecesPerSegment}u;
${FIND_PATH_WGSL}
${getLineGeometrySource(props.coordinateSystem, props.radius)}

fn getPieceCount(segmentLength: f32, maximumLength: f32) -> u32 {
  if (!(maximumLength > 0.0) || !(segmentLength > 0.0)) {
    return 1u;
  }
  let pieces = min(ceil(segmentLength / maximumLength), f32(MAXIMUM_PIECES));
  return clamp(u32(pieces), 1u, MAXIMUM_PIECES);
}`,
    body: /* wgsl */ `let path = findPath(index);
  var count = 0u;
  if (path != NO_PATH) {
    if (index + 1u >= pathOffsets[pathOffsetsOffset + path + 1u]) {
      count = 1u;
    } else {
      count = getPieceCount(
        getSegmentLength(getPosition(index), getPosition(index + 1u)),
        parameters[parametersOffset]
      );
    }
  }
  counts[countsOffset + index] = count;`
  });
}

/** Properties for {@link createSegmentizeEmitNode}. @internal */
export type SegmentizeEmitNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  coordinateSystem: GPULineCoordinateSystem;
  radius: number;
  counts: GraphDataView<'uint32'>;
  starts: GraphDataView<'uint32'>;
  rowMeasures?: GraphDataView<'float32'>;
  rowShifts?: GraphDataView<'float32'>;
  outputPositions: GraphDataView<'float32x2'>;
  outputSourceRows?: GraphDataView<'uint32'>;
  outputMeasures?: GraphDataView<'float32'>;
};

/**
 * Emits each row's own vertex and its interior points (planar lerp or great-circle slerp, evenly
 * spaced) at the row's scanned start. Writes past the vertex capacity are dropped.
 *
 * @internal
 */
export function createSegmentizeEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SegmentizeEmitNodeProps
): GPUCommandNode<Parameters> {
  const spherical = props.coordinateSystem === 'spherical';
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {
      name: 'outputPositions',
      view: props.outputPositions,
      type: 'f32',
      access: 'read_write'
    }
  ];
  if (spherical && props.rowShifts) {
    bindings.push({
      name: 'rowShifts',
      view: props.rowShifts,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.outputMeasures && props.rowMeasures) {
    bindings.push(
      {
        name: 'rowMeasures',
        view: props.rowMeasures,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'outputMeasures',
        view: props.outputMeasures,
        type: 'f32',
        access: 'read_write'
      }
    );
  }
  if (props.outputSourceRows) {
    bindings.push({
      name: 'outputSourceRows',
      view: props.outputSourceRows,
      type: 'u32',
      access: 'read_write'
    });
  }
  const writeMeasures = Boolean(props.outputMeasures && props.rowMeasures);
  const interpolate = spherical
    ? 'geodesicInterpolate(start, end, angle, fraction)'
    : 'mix(start, end, fraction)';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'emit',
    bindings,
    invocationCount: props.positions.length,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;
${getLineGeometrySource(props.coordinateSystem, props.radius)}

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${writeMeasures ? 'outputMeasures[outputMeasuresOffset + outputRow] = measure;' : ''}
  ${props.outputSourceRows ? 'outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;' : ''}
}`,
    body: /* wgsl */ `let count = counts[countsOffset + index];
  if (count == 0u) {
    return;
  }
  let outputStart = starts[startsOffset + index];
  var start = getPosition(index);
  ${spherical && props.rowShifts ? 'start.x = start.x + rowShifts[rowShiftsOffset + index];' : ''}
  let measure = ${writeMeasures ? 'rowMeasures[rowMeasuresOffset + index]' : '0.0'};
  writeVertex(outputStart, start, measure, index);
  if (count > 1u) {
    let end = getPosition(index + 1u);
    let segmentLength = getSegmentLength(start, end);
    ${spherical ? 'let angle = geodesicCentralAngle(start, end);' : ''}
    for (var piece = 1u; piece < count; piece++) {
      let fraction = f32(piece) / f32(count);
      writeVertex(outputStart + piece, ${interpolate}, measure + segmentLength * fraction, index);
    }
  }`
  });
}

/** Properties for {@link createArcCountNode}. @internal */
export type ArcCountNodeProps = {
  id: string;
  operation: string;
  sources: GraphDataView<'float32x2'>;
  targets: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  radius: number;
  maximumSegments: number;
  /** Per-pair output vertex count (`segments + 1`). */
  counts: GraphDataView<'uint32'>;
};

const ARC_GEOMETRY_WGSL = /* wgsl */ `
fn getSource(row: u32) -> vec2<f32> {
  return vec2<f32>(sources[sourcesOffset + 2u * row], sources[sourcesOffset + 2u * row + 1u]);
}

fn getTarget(row: u32) -> vec2<f32> {
  return vec2<f32>(targets[targetsOffset + 2u * row], targets[targetsOffset + 2u * row + 1u]);
}
`;

/**
 * Per-pair vertex count of `GPUGreatCircleArcs`: `segments + 1` with
 * `segments = clamp(max(ceil(distance / maximumSegmentLength), minimumSegments), 1, maximumSegments)`.
 *
 * @internal
 */
export function createArcCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ArcCountNodeProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'count',
    bindings: [
      {name: 'sources', view: props.sources, type: 'f32', access: 'read'},
      {name: 'targets', view: props.targets, type: 'f32', access: 'read'},
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {name: 'counts', view: props.counts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.sources.length,
    declarations: `${GEODESIC_WGSL}
const RADIUS: f32 = ${getWGSLFloatLiteral(props.radius)};
const MAXIMUM_SEGMENTS: u32 = ${props.maximumSegments}u;
${ARC_GEOMETRY_WGSL}`,
    body: /* wgsl */ `let distance = geodesicCentralAngle(getSource(index), getTarget(index)) * RADIUS;
  let maximumLength = parameters[parametersOffset];
  let minimumSegments = parameters[parametersOffset + 1u];
  var segments = 1.0;
  if (maximumLength > 0.0 && distance > 0.0) {
    segments = ceil(min(distance / maximumLength, f32(MAXIMUM_SEGMENTS)));
  }
  if (minimumSegments > segments) {
    segments = min(round(minimumSegments), f32(MAXIMUM_SEGMENTS));
  }
  counts[countsOffset + index] = clamp(u32(segments), 1u, MAXIMUM_SEGMENTS) + 1u;`
  });
}

/** Properties for {@link createArcEmitNode}. @internal */
export type ArcEmitNodeProps = {
  id: string;
  operation: string;
  sources: GraphDataView<'float32x2'>;
  targets: GraphDataView<'float32x2'>;
  radius: number;
  counts: GraphDataView<'uint32'>;
  starts: GraphDataView<'uint32'>;
  outputPositions: GraphDataView<'float32x2'>;
  outputSourceRows?: GraphDataView<'uint32'>;
  outputMeasures?: GraphDataView<'float32'>;
};

/**
 * Emits each pair's great-circle arc, slerped evenly in angle, with longitudes continuous from the
 * source. The first and last vertices are the source and the unwrapped target exactly.
 *
 * @internal
 */
export function createArcEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ArcEmitNodeProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'sources', view: props.sources, type: 'f32', access: 'read'},
    {name: 'targets', view: props.targets, type: 'f32', access: 'read'},
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {
      name: 'outputPositions',
      view: props.outputPositions,
      type: 'f32',
      access: 'read_write'
    }
  ];
  if (props.outputMeasures) {
    bindings.push({
      name: 'outputMeasures',
      view: props.outputMeasures,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.outputSourceRows) {
    bindings.push({
      name: 'outputSourceRows',
      view: props.outputSourceRows,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'emit',
    bindings,
    invocationCount: props.sources.length,
    declarations: `${GEODESIC_WGSL}
const RADIUS: f32 = ${getWGSLFloatLiteral(props.radius)};
const CAPACITY: u32 = ${props.outputPositions.length}u;
${ARC_GEOMETRY_WGSL}

fn writeVertex(outputRow: u32, vertex: vec2<f32>, measure: f32, sourceRow: u32) {
  if (outputRow >= CAPACITY) {
    return;
  }
  outputPositions[outputPositionsOffset + 2u * outputRow] = vertex.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = vertex.y;
  ${props.outputMeasures ? 'outputMeasures[outputMeasuresOffset + outputRow] = measure;' : ''}
  ${props.outputSourceRows ? 'outputSourceRows[outputSourceRowsOffset + outputRow] = sourceRow;' : ''}
}`,
    body: /* wgsl */ `let count = counts[countsOffset + index];
  let outputStart = starts[startsOffset + index];
  let source = getSource(index);
  var destination = getTarget(index);
  destination.x = source.x + geodesicWrapLongitudeDelta(destination.x - source.x);
  let angle = geodesicCentralAngle(source, destination);
  let segments = count - 1u;
  writeVertex(outputStart, source, 0.0, index);
  for (var vertex = 1u; vertex < segments; vertex++) {
    let fraction = f32(vertex) / f32(segments);
    writeVertex(
      outputStart + vertex,
      geodesicInterpolate(source, destination, angle, fraction),
      angle * RADIUS * fraction,
      index
    );
  }
  writeVertex(outputStart + segments, destination, angle * RADIUS, index);`
  });
}

/** Properties for {@link createPathOffsetsPublishNode}. @internal */
export type PathOffsetsPublishNodeProps = {
  id: string;
  operation: string;
  /**
   * Input path row offsets (`pathCount + 1`). Omit when every path is one input row (pairs), so
   * path `p` starts at row `p`.
   */
  pathOffsets?: GraphDataView<'uint32'>;
  pathCount: number;
  rowCount: number;
  counts: GraphDataView<'uint32'>;
  starts: GraphDataView<'uint32'>;
  outputPathOffsets: GraphDataView<'uint32'>;
  capacity: number;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  totalCount?: GraphDataView<'uint32'>;
  pathCountOutput?: GraphDataView<'uint32'>;
};

/**
 * Writes clamped output path offsets from the scanned row starts, plus the clamped vertex count,
 * the overflow flag, and the optional unclamped total and path count. Every word is rewritten.
 *
 * @internal
 */
export function createPathOffsetsPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PathOffsetsPublishNodeProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [];
  if (props.pathOffsets) {
    bindings.push({
      name: 'pathOffsets',
      view: props.pathOffsets,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'starts', view: props.starts, type: 'u32', access: 'read'},
    {
      name: 'outputPathOffsets',
      view: props.outputPathOffsets,
      type: 'u32',
      access: 'read_write'
    },
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {
      name: 'overflowOut',
      view: props.overflow,
      type: 'u32',
      access: 'read_write'
    }
  );
  if (props.totalCount) {
    bindings.push({
      name: 'totalOut',
      view: props.totalCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.pathCountOutput) {
    bindings.push({
      name: 'pathCountOut',
      view: props.pathCountOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  const pathRow = props.pathOffsets
    ? 'min(pathOffsets[pathOffsetsOffset + index], ROW_COUNT)'
    : 'min(index, ROW_COUNT)';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'publish',
    bindings,
    invocationCount: props.pathCount + 1,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const CAPACITY: u32 = ${props.capacity}u;
const PATH_COUNT: u32 = ${props.pathCount}u;`,
    body: /* wgsl */ `let total = starts[startsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  let row = ${pathRow};
  var start = total;
  if (row < ROW_COUNT) {
    start = starts[startsOffset + row];
  }
  outputPathOffsets[outputPathOffsetsOffset + index] = min(start, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${props.totalCount ? 'totalOut[totalOutOffset] = total;' : ''}
    ${props.pathCountOutput ? 'pathCountOut[pathCountOutOffset] = PATH_COUNT;' : ''}
  }`
  });
}

/**
 * Writes the identity source-path column (`sourcePaths[p] = p`) for contributors that emit one output
 * path per input path.
 *
 * @internal
 */
export function createIdentitySourcePathsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    sourcePaths: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'source-paths',
    bindings: [
      {
        name: 'sourcePaths',
        view: props.sourcePaths,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.sourcePaths.length,
    body: 'sourcePaths[sourcePathsOffset + index] = index;'
  });
}
