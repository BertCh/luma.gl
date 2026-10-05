// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPULatticeWeights';

/** Largest supported neighborhood `radius` in cells. */
export const GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS = 32;

/** Lattice neighborhood shape. `'rook'`: Manhattan distance; `'queen'`: Chebyshev distance. */
export type GPULatticeCriterion = 'rook' | 'queen';

/**
 * Properties for {@link GPULatticeWeights}.
 *
 * Compile-time: every prop except the contents of `mask`.
 */
export type GPULatticeWeightsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'lattice-weights'`. */
  id?: string;
  /** Lattice columns. Row-major ID of cell `(x, y)` is `y * width + x`. */
  width: number;
  /** Lattice rows. The weights have `width * height` rows. */
  height: number;
  /**
   * `'rook'`: cells with `|dx| + |dy| <= radius` (von Neumann; 4 neighbors at radius 1).
   * `'queen'`: cells with `max(|dx|, |dy|) <= radius` (Moore; 8 neighbors at radius 1).
   */
  criterion: GPULatticeCriterion;
  /** Neighborhood radius in cells, an integer in `[1, 32]`. Defaults to 1, which matches PySAL `lat2W`. */
  radius?: number;
  /** Optional per-cell selection (`width * height` rows): zero removes the cell as a row and as a neighbor. */
  mask?: GraphDataView<'uint32'>;
  /** Cell width and height used for `weights.distances`. Defaults to `[1, 1]`. */
  cellSize?: readonly [number, number];
  /**
   * Caller-owned output CSR with `width * height + 1` offsets. Binary weights (1); `distances`
   * (optional) hold the Euclidean distance between cell centers.
   */
  weights: GPUSpatialWeights;
  /** Caller-owned one-row flag: 1 when the neighbors did not fit the capacity, else 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional caller-owned one-row unclamped total neighbor count. */
  totalNeighbors?: GraphDataView<'uint32'>;
};

/**
 * Regular-grid contiguity weights (the PySAL `lat2W` equivalent) written as a
 * {@link GPUSpatialWeights} CSR.
 *
 * Cells are numbered row-major (`id = y * width + x`). A cell's neighbors are the other in-bounds,
 * unmasked cells inside its `criterion` neighborhood of `radius` cells, listed by ascending ID.
 * Masked cells have empty rows and never appear as neighbors. Weights are binary; distances are
 * `sqrt((dx * cellWidth)^2 + (dy * cellHeight)^2)`. Capacity handling matches
 * {@link GPUNeighborSearch}: offsets are clamped, slots past the capacity are dropped and
 * `overflow` is set. Output is deterministic and matches the CPU result exactly (distances within
 * f32 rounding).
 */
export class GPULatticeWeights implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULatticeWeightsProps;

  constructor(props: GPULatticeWeightsProps) {
    const id = props.id ?? 'lattice-weights';
    this.id = id;
    this.props = props;
    if (props.criterion !== 'rook' && props.criterion !== 'queen') {
      throw new Error(`${id} criterion must be 'rook' or 'queen'`);
    }
    for (const [name, value] of [
      ['width', props.width],
      ['height', props.height]
    ] as const) {
      if (!Number.isInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    if (props.width * props.height >= 2 ** 31) {
      throw new Error(`${id} width * height must be below 2^31`);
    }
    const radius = props.radius ?? 1;
    if (!Number.isInteger(radius) || radius < 1 || radius > GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS) {
      throw new Error(
        `${id} radius must be an integer in [1, ${GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS}]`
      );
    }
    if (
      props.cellSize &&
      !(
        props.cellSize.length === 2 &&
        props.cellSize.every(size => Number.isFinite(size) && size > 0)
      )
    ) {
      throw new Error(`${id} cellSize must be two positive finite numbers`);
    }
    const rows = props.width * props.height;
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal width * height`);
      }
    }
    if (validateGPUSpatialWeights(id, props.weights) !== rows) {
      throw new Error(`${id} weights.offsets length must equal width * height + 1`);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must hold one uint32`);
    }
    if (props.totalNeighbors) {
      validatePackedUint32View(props.totalNeighbors, `${id} totalNeighbors`);
      if (props.totalNeighbors.length < 1) {
        throw new Error(`${id} totalNeighbors must hold one uint32`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.distances,
        props.overflow,
        props.totalNeighbors
      ],
      [props.mask]
    );
  }

  /** Returns the lattice nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, weights, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      mask,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      props.overflow,
      props.totalNeighbors
    ]);
    const rows = width * height;
    const capacity = weights.neighbors.length;
    const radius = props.radius ?? 1;
    const [cellWidth, cellHeight] = props.cellSize ?? [1, 1];
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rows);
    const maskBinding = mask
      ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
      : [];
    const declarations = `const WIDTH: i32 = ${width};
const HEIGHT: i32 = ${height};
const RADIUS: i32 = ${radius};
const QUEEN: bool = ${props.criterion === 'queen'};
const CELL_WIDTH: f32 = ${getWGSLFloatLiteral(cellWidth)};
const CELL_HEIGHT: f32 = ${getWGSLFloatLiteral(cellHeight)};`;
    // Visits the in-bounds, unmasked neighbors of cell `index` by ascending ID.
    const visitWGSL = (onNeighbor: string) => `let cellX = i32(index) % WIDTH;
  let cellY = i32(index) / WIDTH;
  if (${mask ? 'mask[maskOffset + index] != 0u' : 'true'}) {
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
      let y = cellY + dy;
      if (y < 0 || y >= HEIGHT) {
        continue;
      }
      for (var dx = -RADIUS; dx <= RADIUS; dx++) {
        let x = cellX + dx;
        if (x < 0 || x >= WIDTH || (dx == 0 && dy == 0)) {
          continue;
        }
        if (!QUEEN && abs(dx) + abs(dy) > RADIUS) {
          continue;
        }
        let neighbor = u32(y * WIDTH + x);
        if (${mask ? 'mask[maskOffset + neighbor] != 0u' : 'true'}) {
          ${onNeighbor}
        }
      }
    }
  }`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          ...maskBinding,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `var count = 0u;
  ${visitWGSL('count++;')}
  counts[countsOffset + index] = count;`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
          ...(props.totalNeighbors
            ? [
                {
                  name: 'totalNeighbors',
                  view: props.totalNeighbors,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows + 1,
        declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `if (index < ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = starts[startsOffset + ROWS - 1u] + counts[countsOffset + ROWS - 1u];
    offsets[offsetsOffset + ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${props.totalNeighbors ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        variant: 'emit',
        bindings: [
          ...maskBinding,
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'},
          ...(weights.distances
            ? [
                {
                  name: 'distances',
                  view: weights.distances,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows,
        declarations,
        body: `var next = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${visitWGSL(`if (next < end) {
            neighbors[neighborsOffset + next] = neighbor;
            weights[weightsOffset + next] = 1.0;
            ${
              weights.distances
                ? `distances[distancesOffset + next] = sqrt(f32(dx * dx) * CELL_WIDTH * CELL_WIDTH + f32(dy * dy) * CELL_HEIGHT * CELL_HEIGHT);`
                : ''
            }
            next++;
          }`)}`
      })
    ];
  }
}
