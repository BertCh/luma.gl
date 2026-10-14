// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {Device} from '@luma.gl/core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL, IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {EXACT_LATTICE_WGSL, LATTICE_MAXIMUM_COORDINATE} from '../../utils/exact-lattice-wgsl';
import {
  GPU_SEGMENT_INTERSECTION_KIND,
  GPUSegmentIntersection,
  getGPUSegmentIntersectionMaximumVertexCount
} from '../segment-intersection/index';
import {getBoundsReductionNodes} from '../hilbert-keys/bounds-reduction';
import type {GPULineGeometryPort, GPUPackedGeometryPort} from '../../analysis-core/contracts/index';
import {GPU_TOPOLOGY_NONE, type GPUNodedSegmentPort} from './topology-types';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUSnapRoundedNoding';

/** Key bits of a lattice coordinate: coordinates are at most `2^22 + 1 < 2^23`. */
const LATTICE_KEY_BITS = 23;

/** Sort key of an item that is not a hot pixel; larger than every lattice coordinate. */
const LATTICE_KEY_SENTINEL = 2 ** LATTICE_KEY_BITS - 1;

/** Number of `u32` words per crossing: sign and three limbs of the denominator and numerator. */
const CROSSING_TERM_WORDS = 8;

/**
 * Snap-rounding output: the noded pieces of {@link GPUNodedSegmentPort} (`endpoints` in world
 * coordinates, `origin + k * cell` rounded to `float32`) plus the exact lattice coordinates.
 *
 * @internal
 */
export type GPUSnapRoundedSegmentPort = GPUNodedSegmentPort & {
  /**
   * Piece endpoints `(x0, y0, x1, y1)` as integer-valued `float32` lattice coordinates in
   * `[0, 2^22 + 1]`, aligned with `endpoints`. Two pieces coincide exactly when their rows are
   * equal.
   */
  latticeEndpoints: GraphDataView<'float32x4'>;
  /**
   * Four rows `(originX, originY, cell, 0)`: the world position of lattice point `(i, j)` is
   * `(originX + i * cell, originY + j * cell)`. `cell` is a power of two.
   */
  lattice: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSnapRoundedNoding}.
 *
 * @internal
 */
export type GPUSnapRoundedNodingProps = {
  /** Prefix for generated node and transient IDs. */
  id?: string;
  /** Planar only (default: native planar coordinates); a longitude-latitude context throws. */
  spatialContext?: GPUSpatialContext;
  /** Arbitrary finite planar linework, as for `GPULineNoding`. */
  lines: GPUPackedGeometryPort<GPULineGeometryPort> & {
    /** Optional source feature ID per line, written to `output.sourceFeatureIds`. */
    sourceIds?: GraphDataView<'uint32'>;
  };
  /**
   * Snap grid, finite and positive, in coordinate units. The lattice step is the smallest power of
   * two that is at least `gridSize` and at least `2^-22` of the extent of the input, so a
   * power-of-two `gridSize` reproduces GEOS fixed precision (`grid_size`) exactly.
   */
  gridSize: number;
  /**
   * Capacity of the crossing list (pairs of snapped segments that cross at one interior point).
   * The hot-pixel table and the candidate table are sized from it, the output capacity and the
   * vertex count.
   */
  intersectionCapacity: number;
  /** Caller-owned pieces, lattice coordinates and status. */
  output: GPUSnapRoundedSegmentPort;
  /** Compile-time. Morton-sorts segments before building the crossing BVH (default: from 1024 vertices). */
  spatialSort?: boolean;
  /** Power-of-two BVH leaf slots of the crossing search. */
  leafCapacity?: number;
};

/**
 * Smallest power of two that is at least `gridSize` (clamped to the normal `float32` range): the
 * lower bound of the lattice step of {@link GPUSnapRoundedNoding}.
 *
 * @internal
 */
export function getSnapRoundingGridCell(gridSize: number): number {
  let exponent = Math.ceil(Math.log2(gridSize));
  while (2 ** exponent < gridSize) {
    exponent++;
  }
  while (2 ** (exponent - 1) >= gridSize) {
    exponent--;
  }
  return 2 ** Math.min(127, Math.max(-126, exponent));
}

/**
 * Largest `lines.positions` row count of {@link GPUSnapRoundedNoding} with `intersectionCapacity`
 * on `device`: four diagonal vertices per hot pixel (`vertices + intersectionCapacity` pixels)
 * share one segment table, at 32 bytes per diagonal vertex. The hot-pixel limit is therefore
 * `floor(device.limits.maxStorageBufferBindingSize / 128)`: 1048576 with 128 MiB bindings,
 * or 8388608 with 1 GiB bindings. The 23-bit lattice keys encode coordinates, not row IDs,
 * and do not limit the number of hot pixels.
 *
 * @internal
 */
export function getGPUSnapRoundedNodingMaximumVertexCount(
  device: Device,
  intersectionCapacity: number
): number {
  return getStorageBindingElementLimit(device, 128) - intersectionCapacity;
}

/**
 * Snap-rounded noding (Hobby; Guibas and Marimont; GEOS `SnapRoundingNoder`) of linework on a
 * power-of-two lattice, entirely on the GPU.
 *
 * Every vertex is rounded to the lattice (half up), every proper crossing of two rounded segments
 * is computed exactly and rounded to the lattice, and every rounded segment is rerouted through
 * the centre of every hot pixel it meets (a hot pixel is a lattice point holding a vertex or a
 * rounded crossing; a pixel is the half-open square `[k - 1/2, k + 1/2)` on both axes). The pieces
 * between consecutive hot pixels of a segment meet only at shared endpoints, and coincident pieces
 * are bit-identical in `latticeEndpoints`. Rounded segments that collapse to a point produce no
 * pieces, and neither do lines that collapse entirely.
 *
 * Pieces are ordered by source segment ID (the start-vertex row in `lines.positions`) and then by
 * position along the segment, so output is deterministic. `sourceRingIds` is the line index,
 * `sourceFeatureIds` the line's `sourceIds` entry or the line index, `sourceEdgeIds` the segment
 * ID, and the parameters are the projections of the piece ends on the rounded source segment.
 *
 * `requiredCount` is exact when `candidateOverflow` is zero. `overflow` is set by any capacity
 * (output, crossings, candidates), a non-finite vertex or a coordinate range that does not fit
 * the lattice, and a segment pair the exact predicates could not certify.
 *
 * @internal
 */
export class GPUSnapRoundedNoding implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSnapRoundedNodingProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Lower bound of the lattice step: `gridSize` rounded up to a power of two. */
  readonly gridCell: number;

  constructor(props: GPUSnapRoundedNodingProps) {
    this.id = props.id ?? 'snap-rounded-noding';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
    const {lines, output} = props;
    if (lines.kind !== 'lines') {
      throw new Error(`${id} lines must be linestring geometry`);
    }
    validatePackedView(lines.positions, ['float32x2'], `${id} lines.positions`);
    validatePackedUint32View(lines.lineOffsets, `${id} lines.lineOffsets`);
    if (lines.lineOffsets.length < 2 || lines.positions.length < 1) {
      throw new Error(`${id} lines require at least one line and one vertex`);
    }
    if (lines.sourceIds) {
      validatePackedUint32View(lines.sourceIds, `${id} lines.sourceIds`);
    }
    if (!Number.isFinite(props.gridSize) || !(props.gridSize > 0) || props.gridSize > 2 ** 127) {
      throw new Error(`${id} gridSize must be finite and positive`);
    }
    this.gridCell = getSnapRoundingGridCell(props.gridSize);
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    const capacity = output.endpoints.length;
    validatePackedView(output.endpoints, ['float32x4'], `${id} output.endpoints`);
    validatePackedView(output.latticeEndpoints, ['float32x4'], `${id} output.latticeEndpoints`);
    if (output.latticeEndpoints.length !== capacity) {
      throw new Error(`${id} output.latticeEndpoints length must equal output.endpoints length`);
    }
    validatePackedView(output.lattice, ['float32'], `${id} output.lattice`);
    if (output.lattice.length !== 4) {
      throw new Error(`${id} output.lattice must contain four rows`);
    }
    for (const [name, view] of [
      ['sourceFeatureIds', output.sourceFeatureIds],
      ['sourceRingIds', output.sourceRingIds],
      ['sourceEdgeIds', output.sourceEdgeIds]
    ] as const) {
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length !== capacity) {
        throw new Error(`${id} output.${name} length must equal output.endpoints length`);
      }
    }
    for (const [name, view] of [
      ['sourceStartParameters', output.sourceStartParameters],
      ['sourceEndParameters', output.sourceEndParameters]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} output.${name}`);
      if (view.length !== capacity) {
        throw new Error(`${id} output.${name} length must equal output.endpoints length`);
      }
    }
    for (const [name, view] of Object.entries(output.status)) {
      if (view) {
        validatePackedUint32View(view, `${id} output.status.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one row`);
        }
      }
    }
    if (!output.status.requiredCount || !output.status.candidateOverflow) {
      throw new Error(`${id} output.status requires requiredCount and candidateOverflow`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {lines, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      lines.positions,
      lines.lineOffsets,
      lines.sourceIds,
      output.endpoints,
      output.latticeEndpoints,
      output.lattice,
      output.sourceFeatureIds,
      output.sourceRingIds,
      output.sourceEdgeIds,
      output.sourceStartParameters,
      output.sourceEndParameters,
      ...Object.values(output.status)
    ]);
    const vertexCount = lines.positions.length;
    const lineCount = lines.lineOffsets.length - 1;
    const outputCapacity = output.endpoints.length;
    const crossingCapacity = props.intersectionCapacity;
    // Hot pixels: one per vertex and one per crossing. Four diagonal vertices per pixel.
    const hotCapacity = vertexCount + crossingCapacity;
    // Each proper crossing can add two pieces, and each segment has two endpoint incidences.
    // Keep this input-derived budget even when the caller requests a small output prefix.
    // Hot-pixel rerouting can still exceed it; candidateOverflow reports that separately.
    const incidenceCapacity = Math.max(outputCapacity + vertexCount, 2 * hotCapacity);
    const candidateCapacity = 2 * incidenceCapacity;
    validateMaximumCount(
      id,
      'lines.positions rows',
      vertexCount,
      getGPUSegmentIntersectionMaximumVertexCount(graph.device)
    );
    validateMaximumCount(
      id,
      '4 x (lines.positions rows + intersectionCapacity)',
      4 * hotCapacity,
      getGPUSegmentIntersectionMaximumVertexCount(graph.device)
    );

    const transient = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const positionsBinding: WGSLKernelBinding = {
      name: 'positions',
      view: lines.positions,
      type: 'f32',
      access: 'read'
    };

    // State words: [0] invalid input (non-finite vertex or unrepresentable range), [1] number of
    // usable (finite, non-collapsed) rounded segments.
    const state = transient('state', 'uint32', 4);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-state`,
        operation: OPERATION,
        variant: 'clear-state',
        bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
        invocationCount: 4,
        body: 'state[stateOffset + index] = 0u;'
      })
    );

    // 1. Lattice frame from the bounds of the finite vertices.
    const bounds = transient('bounds', 'float32', 4);
    nodes.push(
      ...getBoundsReductionNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        variant: 'bounds',
        bindings: [positionsBinding],
        declarations: `${IS_FINITE_BITS_WGSL}
fn readItem(row: u32) -> vec4f {
  let point = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
  let valid = isFiniteBits(point.x) && isFiniteBits(point.y);
  return vec4f(point, select(0.0, 1.0, valid), 0.0);
}`,
        itemCount: vertexCount,
        output: bounds
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-lattice`,
        operation: OPERATION,
        variant: 'lattice',
        bindings: [
          {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
          {name: 'lattice', view: output.lattice, type: 'f32', access: 'read_write'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${IS_FINITE_BITS_WGSL}
const GRID_CELL: f32 = ${getWGSLFloatLiteral(this.gridCell)};
// Smallest power of two at least value, for a finite value >= 0 (at least 2^-126).
fn ceilPowerOfTwo(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let exponent = bits >> 23u;
  if (exponent == 0u) { return 1.17549435e-38; }
  let isExact = (bits & 0x7fffffu) == 0u;
  return bitcast<f32>(select(exponent + 1u, exponent, isExact) << 23u);
}`,
        body: `let minimum = vec2f(bounds[boundsOffset], bounds[boundsOffset + 1u]);
  let maximum = vec2f(bounds[boundsOffset + 2u], bounds[boundsOffset + 3u]);
  let extent = max(maximum.x - minimum.x, maximum.y - minimum.y);
  var usable = isFiniteBits(extent);
  let extentPower = ceilPowerOfTwo(select(0.0, extent, usable));
  usable = usable && isFiniteBits(extentPower);
  // Both operands are powers of two: so is the larger one (2^-22 of the extent, or the grid).
  let cell = select(1.0, max(GRID_CELL, extentPower * 2.384185791015625e-7), usable);
  let scaled = minimum / cell;
  usable = usable && isFiniteBits(scaled.x) && isFiniteBits(scaled.y);
  // The origin is a multiple of the cell, so the origin (and every lattice offset) is exact.
  let origin = select(vec2f(0.0), floor(scaled) * cell, usable);
  lattice[latticeOffset] = origin.x;
  lattice[latticeOffset + 1u] = origin.y;
  lattice[latticeOffset + 2u] = select(1.0, cell, usable);
  lattice[latticeOffset + 3u] = 0.0;
  if (!usable) { atomicOr(&state[stateOffset], 1u); }`
      })
    );

    // 2. Snapped vertices (same packed layout as the input).
    const latticePositions = transient('lattice-positions', 'float32x2', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-snap-vertices`,
        operation: OPERATION,
        variant: 'snap-vertices',
        bindings: [
          positionsBinding,
          {name: 'lattice', view: output.lattice, type: 'f32', access: 'read'},
          {name: 'latPos', view: latticePositions, type: 'f32', access: 'read_write'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${IS_FINITE_BITS_WGSL}
${GET_NAN_WGSL}
const MAXIMUM_COORDINATE: f32 = ${LATTICE_MAXIMUM_COORDINATE}.0;
// floor((value - origin) / cell + 1/2) without a rounding step: the scaled value, its floor and
// the origin multiple are exact, and only the fraction decides the half.
fn snapAxis(value: f32, originMultiple: f32, cell: f32) -> f32 {
  let scaled = value / cell;
  let base = floor(scaled);
  let fraction = scaled - base;
  return (base - originMultiple) + select(0.0, 1.0, fraction >= 0.5);
}`,
        body: `let cell = lattice[latticeOffset + 2u];
  let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  var snapped = vec2f(getNaN());
  if (isFiniteBits(point.x) && isFiniteBits(point.y)) {
    let x = snapAxis(point.x, lattice[latticeOffset] / cell, cell);
    let y = snapAxis(point.y, lattice[latticeOffset + 1u] / cell, cell);
    if (isFiniteBits(x) && isFiniteBits(y) && x >= 0.0 && y >= 0.0 && x <= MAXIMUM_COORDINATE && y <= MAXIMUM_COORDINATE) {
      snapped = vec2f(x, y);
    } else {
      atomicOr(&state[stateOffset], 1u);
    }
  } else {
    atomicOr(&state[stateOffset], 1u);
  }
  latPos[latPosOffset + index * 2u] = snapped.x;
  latPos[latPosOffset + index * 2u + 1u] = snapped.y;`
      })
    );

    const latticeLines = {
      kind: 'lines' as const,
      positions: latticePositions,
      lineOffsets: lines.lineOffsets
    };
    const lineWGSL = `const LINE_COUNT: u32 = ${lineCount}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;
const LINE_NONE: u32 = 0xffffffffu;
// The line holding the vertex, or LINE_NONE.
fn lineOfVertex(vertex: u32) -> u32 {
  if (lineOffsets[lineOffsetsOffset] > vertex) { return LINE_NONE; }
  var low = 0u;
  var high = LINE_COUNT;
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle; }
  }
  if (vertex >= min(lineOffsets[lineOffsetsOffset + low + 1u], VERTEX_COUNT)) { return LINE_NONE; }
  return low;
}`;
    const latticePointWGSL = `${IS_NAN_BITS_WGSL}
const LATTICE_VERTEX_COUNT: u32 = ${vertexCount}u;
// The vertex is clamped so that a read of the row after the last vertex stays in range even where a
// caller evaluates both sides of an && (the executed specs' interpreter does).
fn latticeFloat(vertex: u32) -> vec2f {
  let row = min(vertex, LATTICE_VERTEX_COUNT - 1u);
  return vec2f(latPos[latPosOffset + row * 2u], latPos[latPosOffset + row * 2u + 1u]);
}
fn latticePoint(vertex: u32) -> vec2i {
  let point = latticeFloat(vertex);
  return vec2i(i32(point.x), i32(point.y));
}
// Both ends finite and distinct: the segment starting at vertex survives snapping.
fn segmentUsable(vertex: u32) -> bool {
  let start = latticeFloat(vertex);
  let end = latticeFloat(vertex + 1u);
  return !isNanBits(start.x) && !isNanBits(start.y) && !isNanBits(end.x) && !isNanBits(end.y) &&
    (start.x != end.x || start.y != end.y);
}`;

    // 3. Proper crossings of the snapped segments.
    const keysX = transient('keys-x', 'uint32', hotCapacity);
    const keysY = transient('keys-y', 'uint32', hotCapacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mark-vertices`,
        operation: OPERATION,
        variant: 'mark-vertices',
        bindings: [
          {name: 'latPos', view: latticePositions, type: 'f32', access: 'read'},
          {name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read_write'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `${lineWGSL}
${latticePointWGSL}
const KEY_SENTINEL: u32 = ${LATTICE_KEY_SENTINEL}u;`,
        // A vertex is a hot pixel when a usable segment of its line starts or ends there. A line
        // that collapses to one lattice point (no usable segment) adds no pixel, as in GEOS.
        body: `let line = lineOfVertex(index);
  var hot = false;
  if (line != LINE_NONE) {
    let lineStart = lineOffsets[lineOffsetsOffset + line];
    let lineEnd = min(lineOffsets[lineOffsetsOffset + line + 1u], VERTEX_COUNT);
    let hasNext = index + 1u < lineEnd && segmentUsable(index);
    let hasPrevious = index > lineStart && segmentUsable(index - 1u);
    hot = hasNext || hasPrevious;
    if (hasNext) { atomicAdd(&state[stateOffset + 1u], 1u); }
  }
  let point = latticePoint(index);
  keysX[keysXOffset + index] = select(KEY_SENTINEL, u32(point.x), hot);
  keysY[keysYOffset + index] = select(KEY_SENTINEL, u32(point.y), hot);`
      })
    );

    const crossingLeft = transient('crossing-left', 'uint32', crossingCapacity);
    const crossingRight = transient('crossing-right', 'uint32', crossingCapacity);
    const crossingCount = transient('crossing-count', 'uint32', 1);
    const crossingOverflow = transient('crossing-overflow', 'uint32', 1);
    const crossingRequired = transient('crossing-required', 'uint32', 1);
    const crossingKinds = transient('crossing-kinds', 'uint32', crossingCapacity);
    const uncertainCount =
      output.status.uncertainCount ?? transient('uncertain-count', 'uint32', 1);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-crossings`,
        left: latticeLines,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        output: {
          pairs: {
            leftIds: crossingLeft,
            rightIds: crossingRight,
            status: {
              count: crossingCount,
              overflow: crossingOverflow,
              requiredCount: crossingRequired
            }
          },
          kinds: crossingKinds,
          status: {uncertainCount}
        }
      }).getCommandNodes(graph)
    );

    // 4. Exact rounded crossings become hot pixels. Two kernels so each has one multiplication.
    const crossingTerms = transient(
      'crossing-terms',
      'uint32',
      crossingCapacity * CROSSING_TERM_WORDS
    );
    const crossingBindings: WGSLKernelBinding[] = [
      {name: 'pairLeft', view: crossingLeft, type: 'u32', access: 'read'},
      {name: 'pairRight', view: crossingRight, type: 'u32', access: 'read'},
      {name: 'pairCount', view: crossingCount, type: 'u32', access: 'read'},
      {name: 'kinds', view: crossingKinds, type: 'u32', access: 'read'},
      {name: 'latPos', view: latticePositions, type: 'f32', access: 'read'}
    ];
    const crossingDeclarations = `${EXACT_LATTICE_WGSL}
${latticePointWGSL.replace(/fn segmentUsable[\s\S]*$/, '')}
const KIND_PROPER: u32 = ${GPU_SEGMENT_INTERSECTION_KIND.proper}u;
fn isProperPair(row: u32) -> bool {
  return row < min(pairCount[pairCountOffset], ${crossingCapacity}u) && kinds[kindsOffset + row] == KIND_PROPER;
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-crossing-terms`,
        operation: OPERATION,
        variant: 'crossing-terms',
        bindings: [
          ...crossingBindings,
          {name: 'termWords', view: crossingTerms, type: 'u32', access: 'read_write'}
        ],
        invocationCount: crossingCapacity,
        declarations: crossingDeclarations,
        body: `var found = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  if (isProperPair(index)) {
    let left = pairLeft[pairLeftOffset + index];
    let right = pairRight[pairRightOffset + index];
    found = latticeCrossingTerms(latticePoint(left), latticePoint(left + 1u), latticePoint(right), latticePoint(right + 1u));
  }
  // Sign and three limbs (below 2^48) of the denominator, then of the numerator.
  let base = termWordsOffset + index * ${CROSSING_TERM_WORDS}u;
  for (var slot = 0u; slot < 4u; slot++) {
    termWords[base + slot] = found[slot];
    termWords[base + 4u + slot] = found[6u + slot];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-round-crossings`,
        operation: OPERATION,
        variant: 'round-crossings',
        bindings: [
          ...crossingBindings,
          {name: 'termWords', view: crossingTerms, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read_write'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read_write'}
        ],
        invocationCount: crossingCapacity,
        declarations: `${crossingDeclarations}
const KEY_SENTINEL: u32 = ${LATTICE_KEY_SENTINEL}u;
fn readTerms(index: u32) -> array<u32, 12> {
  let base = termWordsOffset + index * ${CROSSING_TERM_WORDS}u;
  var terms = array<u32, 12>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  for (var slot = 0u; slot < 4u; slot++) {
    terms[slot] = termWords[base + slot];
    terms[6u + slot] = termWords[base + 4u + slot];
  }
  return terms;
}`,
        body: `var keyX = KEY_SENTINEL;
  var keyY = KEY_SENTINEL;
  if (isProperPair(index)) {
    let left = pairLeft[pairLeftOffset + index];
    let rounded = latticeRoundedCrossing(latticePoint(left), latticePoint(left + 1u), readTerms(index));
    keyX = u32(rounded.x);
    keyY = u32(rounded.y);
  }
  keysX[keysXOffset + ${vertexCount}u + index] = keyX;
  keysY[keysYOffset + ${vertexCount}u + index] = keyY;`
      })
    );

    // 5. Hot pixels: stable lexicographic sort by (x, y), unique, compact.
    const rowIds = transient('row-ids', 'uint32', hotCapacity);
    const sortedKeysY = transient('sorted-keys-y', 'uint32', hotCapacity);
    const orderByY = transient('order-by-y', 'uint32', hotCapacity);
    const keysXByY = transient('keys-x-by-y', 'uint32', hotCapacity);
    const sortedKeysX = transient('sorted-keys-x', 'uint32', hotCapacity);
    const hotOrder = transient('hot-order', 'uint32', hotCapacity);
    const hotFlags = transient('hot-flags', 'uint32', hotCapacity);
    const hotScan = transient('hot-scan', 'uint32', hotCapacity);
    const hot = transient('hot', 'uint32x2', hotCapacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-ids`,
        operation: OPERATION,
        variant: 'row-ids',
        bindings: [{name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}],
        invocationCount: hotCapacity,
        body: 'rowIds[rowIdsOffset + index] = index;'
      }),
      ...new GPUSort({
        id: `${id}-sort-y`,
        keys: keysY,
        values: rowIds,
        outputKeys: sortedKeysY,
        outputValues: orderByY,
        algorithm: 'radix',
        keyBits: LATTICE_KEY_BITS
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather-x`,
        operation: OPERATION,
        variant: 'gather-x',
        bindings: [
          {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
          {name: 'orderByY', view: orderByY, type: 'u32', access: 'read'},
          {name: 'keysXByY', view: keysXByY, type: 'u32', access: 'read_write'}
        ],
        invocationCount: hotCapacity,
        body: 'keysXByY[keysXByYOffset + index] = keysX[keysXOffset + orderByY[orderByYOffset + index]];'
      }),
      ...new GPUSort({
        id: `${id}-sort-x`,
        keys: keysXByY,
        values: orderByY,
        outputKeys: sortedKeysX,
        outputValues: hotOrder,
        algorithm: 'radix',
        keyBits: LATTICE_KEY_BITS
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unique-hot`,
        operation: OPERATION,
        variant: 'unique-hot',
        bindings: [
          {name: 'hotOrder', view: hotOrder, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read'},
          {name: 'hotFlags', view: hotFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: hotCapacity,
        declarations: `const KEY_SENTINEL: u32 = ${LATTICE_KEY_SENTINEL}u;`,
        body: `let row = hotOrder[hotOrderOffset + index];
  var unique = keysX[keysXOffset + row] != KEY_SENTINEL;
  if (unique && index > 0u) {
    let previous = hotOrder[hotOrderOffset + index - 1u];
    unique = keysX[keysXOffset + previous] != keysX[keysXOffset + row] ||
      keysY[keysYOffset + previous] != keysY[keysYOffset + row];
  }
  hotFlags[hotFlagsOffset + index] = select(0u, 1u, unique);`
      }),
      ...new GPUScan({
        id: `${id}-hot-scan`,
        input: hotFlags,
        output: hotScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact-hot`,
        operation: OPERATION,
        variant: 'compact-hot',
        bindings: [
          {name: 'hotOrder', view: hotOrder, type: 'u32', access: 'read'},
          {name: 'keysX', view: keysX, type: 'u32', access: 'read'},
          {name: 'keysY', view: keysY, type: 'u32', access: 'read'},
          {name: 'hotFlags', view: hotFlags, type: 'u32', access: 'read'},
          {name: 'hotScan', view: hotScan, type: 'u32', access: 'read'},
          {name: 'hot', view: hot, type: 'u32', access: 'read_write'}
        ],
        invocationCount: hotCapacity,
        body: `if (hotFlags[hotFlagsOffset + index] == 0u) { return; }
  let row = hotOrder[hotOrderOffset + index];
  let pixel = hotScan[hotScanOffset + index] - 1u;
  hot[hotOffset + pixel * 2u] = keysX[keysXOffset + row];
  hot[hotOffset + pixel * 2u + 1u] = keysY[keysYOffset + row];`
      })
    );

    // 6. The two diagonals of every hot pixel as 2-vertex lines (NaN for unused slots).
    const diagonalPositions = transient('diagonal-positions', 'float32x2', 4 * hotCapacity);
    const diagonalOffsets = transient('diagonal-offsets', 'uint32', 2 * hotCapacity + 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-build-diagonals`,
        operation: OPERATION,
        variant: 'build-diagonals',
        bindings: [
          {name: 'hot', view: hot, type: 'u32', access: 'read'},
          {name: 'hotScan', view: hotScan, type: 'u32', access: 'read'},
          {name: 'diagonalPositions', view: diagonalPositions, type: 'f32', access: 'read_write'},
          {name: 'diagonalOffsets', view: diagonalOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: hotCapacity,
        declarations: GET_NAN_WGSL,
        body: `diagonalOffsets[diagonalOffsetsOffset + index * 2u] = index * 4u;
  diagonalOffsets[diagonalOffsetsOffset + index * 2u + 1u] = index * 4u + 2u;
  if (index + 1u == ${hotCapacity}u) { diagonalOffsets[diagonalOffsetsOffset + index * 2u + 2u] = ${4 * hotCapacity}u; }
  let isPixel = index < hotScan[hotScanOffset + ${hotCapacity - 1}u];
  let nan = getNaN();
  let center = vec2f(f32(hot[hotOffset + index * 2u]), f32(hot[hotOffset + index * 2u + 1u]));
  // Diagonals bottom-left to top-right and top-left to bottom-right. Half-integers are exact.
  var corners = array<vec2f, 4>(
    center + vec2f(-0.5, -0.5), center + vec2f(0.5, 0.5),
    center + vec2f(-0.5, 0.5), center + vec2f(0.5, -0.5));
  for (var corner = 0u; corner < 4u; corner++) {
    let point = select(vec2f(nan), corners[corner], isPixel);
    diagonalPositions[diagonalPositionsOffset + (index * 4u + corner) * 2u] = point.x;
    diagonalPositions[diagonalPositionsOffset + (index * 4u + corner) * 2u + 1u] = point.y;
  }`
      })
    );

    // 7. Candidate incidences: segments against the diagonals, exact half-open filter, dedup.
    const candidateLeft = transient('candidate-left', 'uint32', candidateCapacity);
    const candidateRight = transient('candidate-right', 'uint32', candidateCapacity);
    const candidateCount = transient('candidate-count', 'uint32', 1);
    const candidateOverflow = transient('candidate-overflow', 'uint32', 1);
    const candidateRequired = transient('candidate-required', 'uint32', 1);
    nodes.push(
      ...new GPUSegmentIntersection({
        id: `${id}-incidence`,
        left: latticeLines,
        right: {kind: 'lines', positions: diagonalPositions, lineOffsets: diagonalOffsets},
        output: {
          pairs: {
            leftIds: candidateLeft,
            rightIds: candidateRight,
            status: {
              count: candidateCount,
              overflow: candidateOverflow,
              requiredCount: candidateRequired
            }
          }
        }
      }).getCommandNodes(graph)
    );
    const candidateFlags = transient('candidate-flags', 'uint32', candidateCapacity);
    const candidateScan = transient('candidate-scan', 'uint32', candidateCapacity);
    const incidenceSegments = transient('incidence-segments', 'uint32', incidenceCapacity);
    const incidencePixels = transient('incidence-pixels', 'uint32', incidenceCapacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-filter-candidates`,
        operation: OPERATION,
        variant: 'filter-candidates',
        bindings: [
          {name: 'pairLeft', view: candidateLeft, type: 'u32', access: 'read'},
          {name: 'pairRight', view: candidateRight, type: 'u32', access: 'read'},
          {name: 'pairCount', view: candidateCount, type: 'u32', access: 'read'},
          {name: 'latPos', view: latticePositions, type: 'f32', access: 'read'},
          {name: 'hot', view: hot, type: 'u32', access: 'read'},
          {name: 'candidateFlags', view: candidateFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `${EXACT_LATTICE_WGSL}
${latticePointWGSL.replace(/fn segmentUsable[\s\S]*$/, '')}`,
        // Two diagonals of one pixel hit one segment: keep the first of each (segment, pixel) run.
        body: `var keep = false;
  if (index < min(pairCount[pairCountOffset], ${candidateCapacity}u)) {
    let segment = pairLeft[pairLeftOffset + index];
    let pixel = pairRight[pairRightOffset + index] >> 2u;
    var duplicate = false;
    if (index > 0u) {
      duplicate = pairLeft[pairLeftOffset + index - 1u] == segment &&
        (pairRight[pairRightOffset + index - 1u] >> 2u) == pixel;
    }
    if (!duplicate) {
      let center = vec2i(i32(hot[hotOffset + pixel * 2u]), i32(hot[hotOffset + pixel * 2u + 1u]));
      keep = latticeSegmentMeetsPixel(latticePoint(segment), latticePoint(segment + 1u), center);
    }
  }
  candidateFlags[candidateFlagsOffset + index] = select(0u, 1u, keep);`
      }),
      ...new GPUScan({
        id: `${id}-candidate-scan`,
        input: candidateFlags,
        output: candidateScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter-incidences`,
        operation: OPERATION,
        variant: 'scatter-incidences',
        bindings: [
          {name: 'pairLeft', view: candidateLeft, type: 'u32', access: 'read'},
          {name: 'pairRight', view: candidateRight, type: 'u32', access: 'read'},
          {name: 'candidateScan', view: candidateScan, type: 'u32', access: 'read'},
          {name: 'incidenceSegments', view: incidenceSegments, type: 'u32', access: 'read_write'},
          {name: 'incidencePixels', view: incidencePixels, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        body: `let slot = candidateScan[candidateScanOffset + index];
  var before = 0u;
  if (index > 0u) { before = candidateScan[candidateScanOffset + index - 1u]; }
  if (slot == before || slot > ${incidenceCapacity}u) { return; }
  incidenceSegments[incidenceSegmentsOffset + slot - 1u] = pairLeft[pairLeftOffset + index];
  incidencePixels[incidencePixelsOffset + slot - 1u] = pairRight[pairRightOffset + index] >> 2u;`
      })
    );

    // 8. Order each segment's pixels along the segment. Incidences are grouped by segment with
    // pixels in lexicographic (x, y) order, so the order along a monotone segment follows by
    // reversing runs: x runs when it goes toward smaller x, and y within a run when toward smaller y.
    const orderedSegments = transient('ordered-segments', 'uint32', incidenceCapacity);
    const orderedPixels = transient('ordered-pixels', 'uint32', incidenceCapacity);
    const countWGSL = `fn incidenceTotal() -> u32 { return min(candidateScan[candidateScanOffset + ${candidateCapacity - 1}u], ${incidenceCapacity}u); }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-incidences`,
        operation: OPERATION,
        variant: 'order-incidences',
        bindings: [
          {name: 'candidateScan', view: candidateScan, type: 'u32', access: 'read'},
          {name: 'incidenceSegments', view: incidenceSegments, type: 'u32', access: 'read'},
          {name: 'incidencePixels', view: incidencePixels, type: 'u32', access: 'read'},
          {name: 'hot', view: hot, type: 'u32', access: 'read'},
          {name: 'latPos', view: latticePositions, type: 'f32', access: 'read'},
          {name: 'orderedSegments', view: orderedSegments, type: 'u32', access: 'read_write'},
          {name: 'orderedPixels', view: orderedPixels, type: 'u32', access: 'read_write'}
        ],
        invocationCount: incidenceCapacity,
        declarations: `${latticePointWGSL.replace(/fn segmentUsable[\s\S]*$/, '')}
${countWGSL}
// Key of an incidence row for the binary searches: the segment (mode 0, 1) or the pixel x (2, 3).
fn rowKey(row: u32, mode: u32) -> u32 {
  if (mode < 2u) { return incidenceSegments[incidenceSegmentsOffset + row]; }
  return hot[hotOffset + incidencePixels[incidencePixelsOffset + row] * 2u];
}
// First row in [first, last) whose key is at least searchKey (even mode) or above it (odd mode).
fn searchRows(first: u32, last: u32, mode: u32, searchKey: u32) -> u32 {
  var low = first;
  var high = last;
  while (low < high) {
    let middle = (low + high) / 2u;
    let key = rowKey(middle, mode);
    let before = select(key < searchKey, key <= searchKey, (mode & 1u) == 1u);
    if (before) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `let total = incidenceTotal();
  if (index >= total) { return; }
  let segment = incidenceSegments[incidenceSegmentsOffset + index];
  let groupStart = searchRows(0u, total, 0u, segment);
  let groupEnd = searchRows(groupStart, total, 1u, segment);
  let x = rowKey(index, 2u);
  let runStart = searchRows(groupStart, groupEnd, 2u, x);
  let runEnd = searchRows(runStart, groupEnd, 3u, x);
  let start = latticePoint(segment);
  let end = latticePoint(segment + 1u);
  var before = groupEnd - runEnd;
  if (end.x >= start.x) { before = runStart - groupStart; }
  var offset = runEnd - 1u - index;
  if (end.y >= start.y) { offset = index - runStart; }
  let targetRow = groupStart + before + offset;
  orderedSegments[orderedSegmentsOffset + targetRow] = segment;
  orderedPixels[orderedPixelsOffset + targetRow] = incidencePixels[incidencePixelsOffset + index];`
      })
    );

    // 9. Pieces between consecutive pixels of one segment.
    const pieceFlags = transient('piece-flags', 'uint32', incidenceCapacity);
    const pieceScan = transient('piece-scan', 'uint32', incidenceCapacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-flag-pieces`,
        operation: OPERATION,
        variant: 'flag-pieces',
        bindings: [
          {name: 'candidateScan', view: candidateScan, type: 'u32', access: 'read'},
          {name: 'orderedSegments', view: orderedSegments, type: 'u32', access: 'read'},
          {name: 'pieceFlags', view: pieceFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: incidenceCapacity,
        declarations: countWGSL,
        body: `let total = incidenceTotal();
  var isPiece = false;
  if (index + 1u < total) {
    isPiece = orderedSegments[orderedSegmentsOffset + index] == orderedSegments[orderedSegmentsOffset + index + 1u];
  }
  pieceFlags[pieceFlagsOffset + index] = select(0u, 1u, isPiece);`
      }),
      ...new GPUScan({
        id: `${id}-piece-scan`,
        input: pieceFlags,
        output: pieceScan,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    const scanRowWGSL = `// Output row of piece i, or NONE when it is not a piece or beyond the capacity.
const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
const ROW_NONE: u32 = 0xffffffffu;
fn pieceRow(index: u32) -> u32 {
  let scan = pieceScan[pieceScanOffset + index];
  var before = 0u;
  if (index > 0u) { before = pieceScan[pieceScanOffset + index - 1u]; }
  if (scan == before || scan > OUTPUT_CAPACITY) { return ROW_NONE; }
  return scan - 1u;
}`;
    const pieceScanBinding: WGSLKernelBinding = {
      name: 'pieceScan',
      view: pieceScan,
      type: 'u32',
      access: 'read'
    };
    const sourceIdBinding: WGSLKernelBinding[] = lines.sourceIds
      ? [{name: 'inputSourceIds', view: lines.sourceIds, type: 'u32', access: 'read'}]
      : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {name: 'endpoints', view: output.endpoints, type: 'f32', access: 'read_write'},
          {
            name: 'latticeEndpoints',
            view: output.latticeEndpoints,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'sourceFeatureIds',
            view: output.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read_write'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read_write'},
          {
            name: 'startParameters',
            view: output.sourceStartParameters,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'endParameters',
            view: output.sourceEndParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: outputCapacity,
        body: `let endpoint = index * 4u;
  for (var component = 0u; component < 4u; component++) {
    endpoints[endpointsOffset + endpoint + component] = 0.0;
    latticeEndpoints[latticeEndpointsOffset + endpoint + component] = 0.0;
  }
  sourceFeatureIds[sourceFeatureIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  sourceRingIds[sourceRingIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  sourceEdgeIds[sourceEdgeIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  startParameters[startParametersOffset + index] = 0.0;
  endParameters[endParametersOffset + index] = 0.0;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-geometry`,
        operation: OPERATION,
        variant: 'write-geometry',
        bindings: [
          pieceScanBinding,
          {name: 'orderedPixels', view: orderedPixels, type: 'u32', access: 'read'},
          {name: 'hot', view: hot, type: 'u32', access: 'read'},
          {name: 'lattice', view: output.lattice, type: 'f32', access: 'read'},
          {
            name: 'latticeEndpoints',
            view: output.latticeEndpoints,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'endpoints', view: output.endpoints, type: 'f32', access: 'read_write'}
        ],
        invocationCount: incidenceCapacity,
        declarations: scanRowWGSL,
        body: `let row = pieceRow(index);
  if (row == ROW_NONE) { return; }
  let first = orderedPixels[orderedPixelsOffset + index];
  let second = orderedPixels[orderedPixelsOffset + index + 1u];
  let firstCenter = vec2f(f32(hot[hotOffset + first * 2u]), f32(hot[hotOffset + first * 2u + 1u]));
  let secondCenter = vec2f(f32(hot[hotOffset + second * 2u]), f32(hot[hotOffset + second * 2u + 1u]));
  let origin = vec2f(lattice[latticeOffset], lattice[latticeOffset + 1u]);
  let cell = lattice[latticeOffset + 2u];
  // World position origin + k * cell: k * cell is exact (power-of-two scaling), the sum rounds once.
  let world = array<vec2f, 2>(origin + firstCenter * cell, origin + secondCenter * cell);
  let base = row * 4u;
  latticeEndpoints[latticeEndpointsOffset + base] = firstCenter.x;
  latticeEndpoints[latticeEndpointsOffset + base + 1u] = firstCenter.y;
  latticeEndpoints[latticeEndpointsOffset + base + 2u] = secondCenter.x;
  latticeEndpoints[latticeEndpointsOffset + base + 3u] = secondCenter.y;
  endpoints[endpointsOffset + base] = world[0].x;
  endpoints[endpointsOffset + base + 1u] = world[0].y;
  endpoints[endpointsOffset + base + 2u] = world[1].x;
  endpoints[endpointsOffset + base + 3u] = world[1].y;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-provenance`,
        operation: OPERATION,
        variant: 'write-provenance',
        bindings: [
          pieceScanBinding,
          {name: 'orderedSegments', view: orderedSegments, type: 'u32', access: 'read'},
          {name: 'lineOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          ...sourceIdBinding,
          {
            name: 'sourceFeatureIds',
            view: output.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read_write'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: incidenceCapacity,
        declarations: `${scanRowWGSL}
${lineWGSL}`,
        body: `let row = pieceRow(index);
  if (row == ROW_NONE) { return; }
  let segment = orderedSegments[orderedSegmentsOffset + index];
  let line = lineOfVertex(segment);
  sourceFeatureIds[sourceFeatureIdsOffset + row] = ${lines.sourceIds ? 'inputSourceIds[inputSourceIdsOffset + line]' : 'line'};
  sourceRingIds[sourceRingIdsOffset + row] = line;
  sourceEdgeIds[sourceEdgeIdsOffset + row] = segment;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-parameters`,
        operation: OPERATION,
        variant: 'write-parameters',
        bindings: [
          pieceScanBinding,
          {name: 'latPos', view: latticePositions, type: 'f32', access: 'read'},
          {
            name: 'latticeEndpoints',
            view: output.latticeEndpoints,
            type: 'f32',
            access: 'read'
          },
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read'},
          {
            name: 'startParameters',
            view: output.sourceStartParameters,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'endParameters',
            view: output.sourceEndParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: incidenceCapacity,
        declarations: `${scanRowWGSL}
${latticePointWGSL.replace(/fn segmentUsable[\s\S]*$/, '')}`,
        body: `let row = pieceRow(index);
  if (row == ROW_NONE) { return; }
  let segment = sourceEdgeIds[sourceEdgeIdsOffset + row];
  let start = latticeFloat(segment);
  let delta = latticeFloat(segment + 1u) - start;
  let lengthSquared = dot(delta, delta);
  let base = row * 4u;
  let first = vec2f(latticeEndpoints[latticeEndpointsOffset + base], latticeEndpoints[latticeEndpointsOffset + base + 1u]);
  let second = vec2f(latticeEndpoints[latticeEndpointsOffset + base + 2u], latticeEndpoints[latticeEndpointsOffset + base + 3u]);
  startParameters[startParametersOffset + row] = clamp(dot(first - start, delta) / lengthSquared, 0.0, 1.0);
  endParameters[endParametersOffset + row] = clamp(dot(second - start, delta) / lengthSquared, 0.0, 1.0);`
      })
    );

    // 10. Status.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status-candidates`,
        operation: OPERATION,
        variant: 'status-candidates',
        bindings: [
          {name: 'crossingOverflow', view: crossingOverflow, type: 'u32', access: 'read'},
          {name: 'incidenceOverflow', view: candidateOverflow, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `candidateOverflow[candidateOverflowOffset] =
    select(0u, 1u, crossingOverflow[crossingOverflowOffset] != 0u || incidenceOverflow[incidenceOverflowOffset] != 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation: OPERATION,
        variant: 'status',
        bindings: [
          {name: 'candidateScan', view: candidateScan, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'uncertainCount', view: uncertainCount, type: 'u32', access: 'read'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {
            name: 'requiredCount',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
const INCIDENCE_CAPACITY: u32 = ${incidenceCapacity}u;`,
        // Every usable segment has an incidence at each end, so pieces = incidences - segments.
        body: `let incidences = candidateScan[candidateScanOffset + ${candidateCapacity - 1}u];
  let segments = min(state[stateOffset + 1u], incidences);
  let required = incidences - segments;
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, OUTPUT_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u,
    required > OUTPUT_CAPACITY || incidences > INCIDENCE_CAPACITY || state[stateOffset] != 0u ||
    candidateOverflow[candidateOverflowOffset] != 0u || uncertainCount[uncertainCountOffset] != 0u);`
      })
    );
    return nodes;
  }
}
