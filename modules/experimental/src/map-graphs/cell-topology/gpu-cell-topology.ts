// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {CELL_KEY_WGSL, getCellKeyLayout, type GPUCellFamily} from '../cell-aggregation/cell-keys';
import type {GPUCellWordOrder} from '../cell-aggregation/gpu-cell-aggregation';
import {H3_NEIGHBOR_WGSL} from './h3-neighbor-wgsl';
import {H3_TOPOLOGY_WGSL, QUADBIN_TOPOLOGY_WGSL} from './cell-topology-wgsl';

const OPERATION = 'GPUCellTopology';

/** Largest `k` of a disk or ring (stride `3k(k+1)+1` for H3, `(2k+1)^2` for Quadbin). */
export const GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS = 8;

/** Largest per-row output stride of a `children` operation. */
export const GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE = 4096;

/**
 * One per-cell topology operation.
 *
 * - `disk`: cells within grid distance `k` (`gridDisk` / `gridDiskDistances`), sorted by
 *   (distance, key) ascending. Stride `3k(k+1)+1` (H3) or `(2k+1)^2` (Quadbin).
 * - `ring`: cells at grid distance exactly `k`, sorted by key. Stride `6k` (H3), `8k` (Quadbin), 1 for `k = 0`.
 * - `parent`: the ancestor at `resolution`; stride 1. Input cells coarser than `resolution` give zero.
 * - `children`: all descendants at `resolution` of cells at exactly `inputResolution`, ascending
 *   key order (`cellToChildren`). Stride `4^d` (Quadbin) or `7^d` (H3) for `d = resolution - inputResolution`,
 *   at most {@link GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE}. Pentagons have fewer children
 *   (`(5 * 7^d + 1) / 6`), padded with zeros. Input cells at another resolution give zero rows.
 *
 * Quadbin distance is the Chebyshev distance of tile columns and rows. Columns wrap across the
 * antimeridian, rows do not (tiles outside `[0, 2^z)` are dropped), so a neighborhood near a pole
 * is clipped. When the wrap makes the square overlap itself at tiny zoom, every tile appears once
 * at its smallest circular distance. H3 distance is the grid distance of `gridDiskDistances`,
 * exact across faces and pentagons.
 */
export type GPUCellTopologyOperation =
  | {type: 'disk'; k: number}
  | {type: 'ring'; k: number}
  | {type: 'parent'; resolution: number}
  | {type: 'children'; resolution: number; inputResolution: number};

/** Output columns of {@link GPUCellTopology}. */
export type GPUCellTopologyOutput = {
  /** `uint32x2` little-endian `(low, high)` cells, `rows * stride` entries, row-major, zero padded. */
  cells: GraphDataView<'uint32x2'>;
  /** Optional `uint32` grid distances aligned with `cells` (`disk` only), zero padded. */
  distances?: GraphDataView<'uint32'>;
  /** Optional `uint32` number of valid entries per input row (0 for invalid or masked rows). */
  counts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellTopology}.
 *
 * Per-frame (no recompile): the contents of `cells` and `mask`. Topology: family, operation, the
 * view layouts.
 */
export type GPUCellTopologyProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-topology'`. */
  id?: string;
  /** Cell family. */
  family: GPUCellFamily;
  /** Operation, fixing the output stride. */
  operation: GPUCellTopologyOperation;
  /** Input `uint32x2` cell keys. Zero or invalid keys produce zero rows. */
  cells: GraphDataView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'`. */
  wordOrder?: GPUCellWordOrder;
  /** Optional per-row mask; zero produces a zero row with count 0. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCellTopologyOutput;
};

/** Number of H3 cells within grid distance `k` of a hexagon. */
function getH3DiskSize(k: number): number {
  return 3 * k * (k + 1) + 1;
}

/**
 * Returns the fixed output stride (entries per input row) of an operation.
 *
 * @throws if `k`, resolutions or the stride are out of range.
 */
export function getCellTopologyStride(
  family: GPUCellFamily,
  operation: GPUCellTopologyOperation
): number {
  if (operation.type === 'disk' || operation.type === 'ring') {
    const {k} = operation;
    if (!Number.isInteger(k) || k < 0 || k > GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS) {
      throw new Error(
        `${operation.type} k must be an integer in [0, ${GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS}]`
      );
    }
    if (operation.type === 'disk') {
      return family === 'h3' ? getH3DiskSize(k) : (2 * k + 1) ** 2;
    }
    return k === 0 ? 1 : (family === 'h3' ? 6 : 8) * k;
  }
  getCellKeyLayout(family, operation.resolution);
  if (operation.type === 'parent') {
    return 1;
  }
  getCellKeyLayout(family, operation.inputResolution);
  const depth = operation.resolution - operation.inputResolution;
  if (depth < 0) {
    throw new Error('children resolution must not be below inputResolution');
  }
  const stride = (family === 'h3' ? 7 : 4) ** depth;
  if (stride > GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE) {
    throw new Error(
      `children stride ${stride} exceeds ${GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE}; reduce the resolution depth`
    );
  }
  return stride;
}

/**
 * Per-cell fixed fan-out topology of Quadbin and H3 cells: `gridDisk`, `gridDiskDistances`,
 * `gridRing`, `cellToParent` and `cellToChildren`.
 *
 * One kernel; each output row `i` occupies entries `[i * stride, (i + 1) * stride)` of the output
 * columns, filled in ascending (distance, key) order and zero padded. Invalid, zero and masked
 * input cells produce all-zero rows with count 0. H3 disks and rings run a bounded breadth-first
 * search over {@link H3_NEIGHBOR_WGSL} in private arrays, so they are exact across icosahedron
 * faces and pentagons; use `k <= 8`. Everything is integer arithmetic, so results are identical
 * on every device and equal the h3-js and BigInt Quadbin references.
 */
export class GPUCellTopology implements GPUMapGraphRecipe {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'cell-topology';
  /** Validated properties. */
  readonly props: GPUCellTopologyProps;
  /** Output entries per input row. */
  readonly stride: number;

  constructor(props: GPUCellTopologyProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    if (props.family !== 'quadbin' && props.family !== 'h3') {
      throw new Error(`${id} family must be 'quadbin' or 'h3'`);
    }
    try {
      this.stride = getCellTopologyStride(props.family, props.operation);
    } catch (error) {
      throw new Error(`${id} ${(error as Error).message}`);
    }
    for (const [name, view] of [
      ['cells', props.cells],
      ['mask', props.mask],
      ['output.cells', props.output.cells],
      ['output.distances', props.output.distances],
      ['output.counts', props.output.counts]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rows = props.cells.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    validatePackedView(props.cells, ['uint32x2'], `${id} cells`);
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    const {output} = props;
    validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    if (output.cells.length !== rows * this.stride) {
      throw new Error(`${id} output.cells length must be rows * stride = ${rows * this.stride}`);
    }
    if (output.distances) {
      if (props.operation.type !== 'disk') {
        throw new Error(`${id} output.distances is only available for the disk operation`);
      }
      validatePackedUint32View(output.distances, `${id} output.distances`);
      if (output.distances.length !== rows * this.stride) {
        throw new Error(`${id} output.distances length must equal output.cells length`);
      }
    }
    if (output.counts) {
      validatePackedUint32View(output.counts, `${id} output.counts`);
      if (output.counts.length !== rows) {
        throw new Error(`${id} output.counts length must equal the row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.cells, output.distances, output.counts],
      [props.cells, props.mask]
    );
  }

  /** Returns the single topology node `${id}-topology`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, stride} = this;
    const {family, operation, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, props.mask, ...Object.values(output)]);
    const rows = props.cells.length;
    const isH3 = family === 'h3';
    const bindings: MapGraphKernelBinding[] = [
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
      ...(props.mask
        ? [{name: 'rowMask', view: props.mask, type: 'u32', access: 'read'} as const]
        : []),
      {name: 'outCells', view: output.cells, type: 'u32', access: 'read_write'},
      ...(output.distances
        ? [
            {
              name: 'outDistances',
              view: output.distances,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : []),
      ...(output.counts
        ? [{name: 'outCounts', view: output.counts, type: 'u32', access: 'read_write'} as const]
        : [])
    ];

    const isNeighborhood = operation.type === 'disk' || operation.type === 'ring';
    const radius = isNeighborhood ? operation.k : 0;
    const capacity = isNeighborhood
      ? isH3
        ? getH3DiskSize(radius)
        : getCellTopologyStride(family, {type: 'disk', k: radius})
      : 1;
    // The source resolution a row must have (children), or the minimum (parent).
    const targetResolution =
      operation.type === 'parent' || operation.type === 'children' ? operation.resolution : 0;
    const inputResolution = operation.type === 'children' ? operation.inputResolution : 0;
    const depth = operation.type === 'children' ? operation.resolution - inputResolution : 0;

    const resolutionOf = isH3 ? 'dggs_h3_get_resolution(key)' : '((key.x >> 20u) & 0x1fu)';
    const validity = isH3 ? 'dggs_h3_is_valid_cell_id(key)' : 'cellIsValidQuadbin(key)';
    let resolutionCheck = 'true';
    if (operation.type === 'parent') {
      resolutionCheck = `${resolutionOf} >= TARGET_RESOLUTION`;
    } else if (operation.type === 'children') {
      resolutionCheck = `${resolutionOf} == INPUT_RESOLUTION`;
    }
    const readRow = (rowExpression: string) => `
  let words = vec2u(cells[cellsOffset + 2u * ${rowExpression}], cells[cellsOffset + 2u * ${rowExpression} + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  var isValid = ${validity} && ${resolutionCheck};
  ${props.mask ? `isValid = isValid && rowMask[rowMaskOffset + ${rowExpression}] != 0u;` : ''}`;
    const writeCell = (slot: string, cell: string, distance?: string) => `
  outCells[outCellsOffset + 2u * ${slot}] = ${cell}.y;
  outCells[outCellsOffset + 2u * ${slot} + 1u] = ${cell}.x;
  ${output.distances ? `outDistances[outDistancesOffset + ${slot}] = ${distance ?? '0u'};` : ''}`;

    let body: string;
    if (isNeighborhood) {
      body = `${readRow('index')}
  var written = 0u;
  if (isValid) {
    written = ${isH3 ? 'cellTopologyH3Neighborhood' : 'cellTopologyQuadbinNeighborhood'}(key, ${operation.type === 'ring'});
  }
  for (var j = 0u; j < STRIDE; j++) {
    let slot = index * STRIDE + j;
    if (j < written) {${writeCell('slot', 'cellTopologyCells[j]', 'cellTopologyDistances[j]')}
    } else {${writeCell('slot', 'vec2u(0u)')}
    }
  }
  ${output.counts ? 'outCounts[outCountsOffset + index] = written;' : ''}`;
    } else if (operation.type === 'parent') {
      const parent = isH3
        ? 'dggs_h3_get_parent(key, TARGET_RESOLUTION)'
        : 'cellTopologyQuadbinGetParent(key, TARGET_RESOLUTION)';
      body = `${readRow('index')}
  var parent = vec2u(0u);
  if (isValid) {
    parent = ${parent};
  }${writeCell('index', 'parent')}
  ${output.counts ? 'outCounts[outCountsOffset + index] = select(0u, 1u, isValid);' : ''}`;
    } else {
      const child = isH3
        ? 'cellTopologyH3GetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)'
        : 'cellTopologyQuadbinGetChild(key, TARGET_RESOLUTION, DEPTH, slotInRow)';
      const count = isH3 ? 'cellTopologyH3GetChildCount(key, DEPTH)' : 'STRIDE';
      body = `let row = index / STRIDE;
  let slotInRow = index % STRIDE;${readRow('row')}
  var child = vec2u(0u);
  if (isValid) {
    child = ${child};
  }${writeCell('index', 'child')}
  ${output.counts ? `if (slotInRow == 0u) {\n    outCounts[outCountsOffset + row] = select(0u, ${count}, isValid);\n  }` : ''}`;
    }

    return [
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-topology`,
        operation: OPERATION,
        variant: `${family}-${operation.type}`,
        bindings,
        invocationCount: operation.type === 'children' ? rows * stride : rows,
        declarations: `const STRIDE: u32 = ${stride}u;
const CELL_TOPOLOGY_K: u32 = ${radius}u;
const CELL_TOPOLOGY_CAPACITY: u32 = ${capacity}u;
const TARGET_RESOLUTION: u32 = ${targetResolution}u;
const INPUT_RESOLUTION: u32 = ${inputResolution}u;
const DEPTH: u32 = ${depth}u;
${isH3 ? dggs.source : ''}
${CELL_KEY_WGSL}
${isH3 ? H3_NEIGHBOR_WGSL : ''}
${QUADBIN_TOPOLOGY_WGSL}
${isH3 ? H3_TOPOLOGY_WGSL : ''}`,
        body
      })
    ];
  }
}
