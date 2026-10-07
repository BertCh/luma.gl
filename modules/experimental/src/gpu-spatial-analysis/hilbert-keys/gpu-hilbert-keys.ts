// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {getBoundsReductionNodes} from './bounds-reduction';

const OPERATION = 'GPUHilbertKeys';

/** Largest Hilbert curve order (bits per axis): the key is a `uint32` of `2 * order` bits. */
export const GPU_HILBERT_MAXIMUM_ORDER = 16;

/**
 * Key written for an invalid item (non-finite coordinates or an empty box); invalid items sort
 * last. It is `4^order` (one past the last curve cell) for orders below 16 and `0xffffffff` at
 * order 16, where the 32 key bits are all curve index and the last cell shares the sentinel.
 */
export function getGPUHilbertInvalidKey(order: number): number {
  return order >= GPU_HILBERT_MAXIMUM_ORDER ? 0xffffffff : 4 ** order;
}

/** Number of float32 elements of a {@link GPUHilbertKeys} bounds view. */
export const GPU_HILBERT_BOUNDS_LENGTH = 4;

/** Caller-owned outputs of {@link GPUHilbertKeys}. */
export type GPUHilbertKeysOutput = {
  /** One Hilbert index per item, in `[0, 4^order)`, or {@link getGPUHilbertInvalidKey}. */
  keys: GraphDataView<'uint32'>;
  /**
   * Optional curve-ordered item rows (a stable ascending sort permutation of `keys`). Requires
   * `sortedKeys` or a transient is used for it. Invalid items come last.
   */
  sortedRows?: GraphDataView<'uint32'>;
  /** Optional ascending keys matching `sortedRows`. */
  sortedKeys?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUHilbertKeys}.
 *
 * Per-frame (no recompile): the contents of `points` or `minima`/`maxima` and `bounds`.
 * Compile-time: `order`, the item count and which outputs are present.
 */
export type GPUHilbertKeysProps = {
  /** Prefix for generated node IDs. Defaults to `'hilbert-keys'`. */
  id?: string;
  /** Curve order (bits per axis, 1 to {@link GPU_HILBERT_MAXIMUM_ORDER}). Defaults to 16. */
  order?: number;
  /** Points to key. Provide this or `minima` and `maxima`. */
  points?: GraphDataView<'float32x2'>;
  /** Feature box minima; keys are of the box centers. Requires `maxima`. */
  minima?: GraphDataView<'float32x2'>;
  /** Feature box maxima with the same length as `minima`. */
  maxima?: GraphDataView<'float32x2'>;
  /**
   * Per-frame `float32` view `[minX, minY, maxX, maxY]` mapped onto the curve. When omitted, the
   * bounds of the valid items are reduced on the GPU each encoding. Items outside the bounds are
   * clamped to its border cells.
   */
  bounds?: GraphDataView<'float32'>;
  /** Output keys and optional curve-order permutation. */
  output: GPUHilbertKeysOutput;
};

/**
 * Computes 2D Hilbert curve indices for points or feature-box centers within bounds ("Hilbert
 * keys next to `GPUBVH`") and, optionally, the curve-ordered permutation through the gpgpu
 * radix sort.
 *
 * Definition. The bounds are divided into `2^order x 2^order` cells; an item falls into cell
 * `(floor(u * 2^order), floor(v * 2^order))` clamped to the grid, where `(u, v)` are its
 * normalized coordinates, and its key is the position of that cell on the Hilbert curve (the
 * classic `xy2d` rotation algorithm: the curve starts in the lower-left cell, first moves along
 * +y and ends in the lower-right cell). Consecutive keys are edge-adjacent cells, which makes the order
 * better at keeping neighbours together than the Morton (Z) order.
 *
 * Keys are integer exact (u32 arithmetic) once the cell is chosen. The cell comes from f32
 * arithmetic `(value - minimum) / extent`, so values within an f32 rounding of a cell border may
 * land in either neighbouring cell. Degenerate bounds (zero extent) put every item in cell 0 on
 * that axis. Invalid items (non-finite coordinates, or boxes with `minimum > maximum`) get
 * {@link getGPUHilbertInvalidKey}; at order 16 that value is also the key of the last curve cell,
 * so an invalid item can tie with it (ties keep ascending row order).
 *
 * Cost: one pass over the items (plus one single-workgroup bounds reduction without `bounds`) and,
 * with `sortedRows`, a radix sort over `2 * order + 1` key bits. No atomics; deterministic.
 */
export class GPUHilbertKeys implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHilbertKeysProps;
  /** Curve order. */
  readonly order: number;
  /** Number of items. */
  readonly itemCount: number;

  constructor(props: GPUHilbertKeysProps) {
    const id = props.id ?? 'hilbert-keys';
    this.id = id;
    this.props = props;
    this.order = props.order ?? GPU_HILBERT_MAXIMUM_ORDER;
    if (
      !Number.isSafeInteger(this.order) ||
      this.order < 1 ||
      this.order > GPU_HILBERT_MAXIMUM_ORDER
    ) {
      throw new Error(`${id} order must be an integer in [1, ${GPU_HILBERT_MAXIMUM_ORDER}]`);
    }
    for (const [name, view] of Object.entries({...props, ...props.output})) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const {points, minima, maxima, bounds} = props;
    if (points) {
      if (minima || maxima) {
        throw new Error(`${id} takes points or minima and maxima, not both`);
      }
      validatePackedView(points, ['float32x2'], `${id} points`);
      this.itemCount = points.length;
    } else {
      if (!minima || !maxima) {
        throw new Error(`${id} needs points, or minima and maxima`);
      }
      validatePackedView(minima, ['float32x2'], `${id} minima`);
      validatePackedView(maxima, ['float32x2'], `${id} maxima`);
      if (minima.length !== maxima.length) {
        throw new Error(`${id} minima and maxima must have the same length`);
      }
      this.itemCount = minima.length;
    }
    if (this.itemCount < 1) {
      throw new Error(`${id} needs at least one item`);
    }
    if (bounds) {
      validatePackedView(bounds, ['float32'], `${id} bounds`);
      if (bounds.length < GPU_HILBERT_BOUNDS_LENGTH) {
        throw new Error(`${id} bounds must hold ${GPU_HILBERT_BOUNDS_LENGTH} float32 values`);
      }
    }
    const {keys, sortedRows, sortedKeys} = props.output;
    for (const [name, view] of [
      ['keys', keys],
      ['sortedRows', sortedRows],
      ['sortedKeys', sortedKeys]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== this.itemCount) {
          throw new Error(`${id} output.${name} must have one row per item`);
        }
      }
    }
    if (sortedKeys && !sortedRows) {
      throw new Error(`${id} output.sortedKeys needs output.sortedRows`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [keys, sortedRows, sortedKeys],
      [points, minima, maxima, bounds]
    );
  }

  /** Returns the optional bounds reduction, the key kernel and the optional sort nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, order, itemCount} = this;
    const {points, minima, maxima} = props;
    const {keys, sortedRows, sortedKeys} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      points,
      minima,
      maxima,
      props.bounds,
      keys,
      sortedRows,
      sortedKeys
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const itemBindings: WGSLKernelBinding[] = points
      ? [{name: 'points', view: points, type: 'f32', access: 'read'}]
      : [
          {name: 'minima', view: minima!, type: 'f32', access: 'read'},
          {name: 'maxima', view: maxima!, type: 'f32', access: 'read'}
        ];
    const declarations = `fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
// Returns (x, y, valid, 0) for an item: the point, or the box center (halved before the sum to
// avoid overflow near FLT_MAX).
fn readItem(row: u32) -> vec4f {
${
  points
    ? `  let point = vec2f(points[pointsOffset + row * 2u], points[pointsOffset + row * 2u + 1u]);
  let valid = isFiniteValue(point.x) && isFiniteValue(point.y);
  return vec4f(point, select(0.0, 1.0, valid), 0.0);`
    : `  let minimum = vec2f(minima[minimaOffset + row * 2u], minima[minimaOffset + row * 2u + 1u]);
  let maximum = vec2f(maxima[maximaOffset + row * 2u], maxima[maximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  return vec4f(minimum * 0.5 + maximum * 0.5, select(0.0, 1.0, valid), 0.0);`
}
}`;

    let bounds = props.bounds;
    if (!bounds) {
      bounds = createTransientView(graph, `${id}-bounds`, 'float32', GPU_HILBERT_BOUNDS_LENGTH);
      nodes.push(
        ...getBoundsReductionNodes<Parameters>(graph, {
          id,
          operation: OPERATION,
          variant: points ? 'bounds-points' : 'bounds-boxes',
          bindings: itemBindings,
          declarations,
          itemCount,
          output: bounds
        })
      );
    }

    const rows = sortedRows
      ? createTransientView(graph, `${id}-rows`, 'uint32', itemCount)
      : undefined;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: `${points ? 'points' : 'boxes'}-order-${order}${rows ? '-rows' : ''}`,
        bindings: [
          ...itemBindings,
          {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
          {name: 'keysOut', view: keys, type: 'u32', access: 'read_write'},
          ...(rows
            ? [{name: 'rowsOut', view: rows, type: 'u32' as const, access: 'read_write' as const}]
            : [])
        ],
        invocationCount: itemCount,
        declarations: `${declarations}
const ORDER: u32 = ${order}u;
const CELLS: u32 = ${2 ** order}u;

fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || !isFiniteValue(extent)) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(floor(normalized * f32(CELLS))), CELLS - 1u);
}

// Spreads the low 16 bits of a value to the even bit positions.
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

// The Hilbert curve index of cell (x, y) on a CELLS x CELLS grid, equal to the classic xy2d
// rotation loop. The loop carries a rotate/flip state from the top bit down, so its 16 steps
// are serial and branch on data. The state transform of each bit level is a small affine map over
// GF(2), so the same result comes from a parallel prefix over the levels (Hacker's Delight 16-5,
// as in rawrunprotected's "Hilbert curves in O(log n)"): four branch-free rounds on 16-bit lanes,
// then one interleave.
fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  let x = cellX << (16u - ORDER);
  let y = cellY << (16u - ORDER);
  let a0 = x ^ y;
  let b0 = 0xffffu ^ a0;
  let c0 = 0xffffu ^ (x | y);
  let d0 = x & (y ^ 0xffffu);
  var transformA = a0 | (b0 >> 1u);
  var transformB = (a0 >> 1u) ^ a0;
  var transformC = ((c0 >> 1u) ^ (b0 & (d0 >> 1u))) ^ c0;
  var transformD = ((a0 & (c0 >> 1u)) ^ (d0 >> 1u)) ^ d0;
  for (var shift = 2u; shift <= 4u; shift = shift << 1u) {
    let a = transformA;
    let b = transformB;
    let c = transformC;
    let d = transformD;
    transformA = (a & (a >> shift)) ^ (b & (b >> shift));
    transformB = (a & (b >> shift)) ^ (b & ((a ^ b) >> shift));
    transformC = transformC ^ ((a & (c >> shift)) ^ (b & (d >> shift)));
    transformD = transformD ^ ((b & (c >> shift)) ^ ((a ^ b) & (d >> shift)));
  }
  let a = transformA;
  let b = transformB;
  let c = transformC;
  let d = transformD;
  transformC = c ^ ((a & (c >> 8u)) ^ (b & (d >> 8u)));
  transformD = d ^ ((b & (c >> 8u)) ^ ((a ^ b) & (d >> 8u)));
  let low = transformC ^ (transformC >> 1u);
  let high = transformD ^ (transformD >> 1u);
  let indexLow = x ^ y;
  let indexHigh = high | (0xffffu ^ (indexLow | low));
  return ((spreadBits(indexHigh) << 1u) | spreadBits(indexLow)) >> (32u - 2u * ORDER);
}`,
        body: `${rows ? 'rowsOut[rowsOutOffset + index] = index;' : ''}
  let item = readItem(index);
  var key = ${getGPUHilbertInvalidKey(order)}u;
  if (item.z > 0.5) {
    let cellX = quantizeAxis(item.x, bounds[boundsOffset], bounds[boundsOffset + 2u]);
    let cellY = quantizeAxis(item.y, bounds[boundsOffset + 1u], bounds[boundsOffset + 3u]);
    key = getHilbertIndex(cellX, cellY);
  }
  keysOut[keysOutOffset + index] = key;`
      })
    );

    if (sortedRows && rows) {
      nodes.push(
        ...new GPUSort({
          id: `${id}-sort`,
          keys,
          values: rows,
          outputKeys:
            sortedKeys ?? createTransientView(graph, `${id}-sorted-keys`, 'uint32', itemCount),
          outputValues: sortedRows,
          algorithm: 'radix',
          keyBits: Math.min(32, 2 * order + 1)
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
