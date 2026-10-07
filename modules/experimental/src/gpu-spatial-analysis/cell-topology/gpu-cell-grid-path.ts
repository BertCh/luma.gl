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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import type {GPUCellWordOrder} from '../cell-aggregation/gpu-cell-aggregation';
import {H3_NEIGHBOR_WGSL} from './h3-neighbor-wgsl';
import {H3_LOCAL_IJ_WGSL} from './h3-local-ij-wgsl';

const OPERATION = 'GPUCellGridPath';

/** Value of `output.distances` for pairs whose grid distance is undefined (H3 `gridDistance` fails). */
export const GPU_CELL_GRID_DISTANCE_UNDEFINED = 0xffffffff;

/** Largest `maximumPathLength` (cells stored per pair). */
export const GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH = 256;

/** Caller-owned outputs of {@link GPUCellGridPath}, all row-aligned with the input pairs. */
export type GPUCellGridPathOutput = {
  /**
   * `uint32` grid distance of each pair (H3 `gridDistance`), or
   * {@link GPU_CELL_GRID_DISTANCE_UNDEFINED} for invalid, masked and unreachable pairs: different
   * resolutions, invalid cells, base cells that are not adjacent, or cells separated by the deleted
   * sector of a pentagon.
   */
  distances?: GraphDataView<'uint32'>;
  /**
   * `uint32x2` little-endian `(low, high)` path cells, `rows * maximumPathLength` entries,
   * row-major: the cells of H3 `gridPathCells`, origin first, destination last, zero padded. Paths
   * longer than `maximumPathLength` keep their first `maximumPathLength` cells.
   */
  cells?: GraphDataView<'uint32x2'>;
  /**
   * `uint32` number of cells of each complete path (`distance + 1`), also when it exceeds
   * `maximumPathLength` (then `counts[row] > maximumPathLength` means the row overflowed and
   * `cells` holds only the first `maximumPathLength` cells). Zero for pairs without a path.
   */
  counts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellGridPath}.
 *
 * Per-frame (no recompile): the contents of `origins`, `destinations` and `mask`. Topology: the
 * view layouts and `maximumPathLength`.
 */
export type GPUCellGridPathProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-grid-path'`. */
  id?: string;
  /** Cell family. Only `'h3'` has a grid-path definition. */
  family: 'h3';
  /** Input `uint32x2` origin cells, one per pair. */
  origins: GraphDataView<'uint32x2'>;
  /** Input `uint32x2` destination cells, one per pair, same length as `origins`. */
  destinations: GraphDataView<'uint32x2'>;
  /** Word order of `origins`, `destinations` and `output.cells`. Defaults to `'little-endian'`. */
  wordOrder?: GPUCellWordOrder;
  /** Optional per-pair mask; zero produces an undefined distance and an empty path. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Cells stored per pair when `output.cells` is present, at most
   * {@link GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH}. Required then, ignored otherwise.
   */
  maximumPathLength?: number;
  /** Caller-owned outputs; at least one of `distances` and `cells`. */
  output: GPUCellGridPathOutput;
};

/**
 * H3 grid distance and grid path cells for arbitrary cell pairs (`gridDistance`, `gridPathCells`,
 * `are_neighbor_cells` as `distance == 1`).
 *
 * One kernel; each thread converts both cells to H3 local IJK coordinates anchored at the origin
 * (`cellToLocalIjk`, including the pentagon rotations) and takes the IJK distance, which equals
 * H3 `gridDistance` for every pair H3 accepts and is undefined exactly where H3 fails (checked
 * against h3-js on pentagon neighborhoods and random pairs at resolutions 1-15). Distances of any
 * length are exact integers.
 *
 * Path cells follow the H3 algorithm: the cube coordinates of the endpoints are interpolated
 * linearly, rounded with `cubeRound` and mapped back to a cell. The cell is found as the unique
 * neighbor of the previous path cell whose local IJK equals the rounded coordinate, which is what
 * H3's `localIjkToCell` returns for every path that stays off the 12 pentagon base cells. Measured
 * against h3-js on random and pentagon-centered pairs of resolutions 3-15, two deviations remain:
 *
 * - Cube rounding is evaluated in exact integer arithmetic, H3 in doubles, so at the exact `x.5`
 *   ties of an interpolation step H3's result depends on rounding noise. About 0.2% of random
 *   pairs contain such a tie where H3's cell is the other adjacent one; the path stays a valid
 *   chain of the same length with the same endpoints.
 * - Where a path crosses a pentagon base cell the rounded coordinate may match no neighbor
 *   uniquely. Such pairs (about 6% of the paths that touch a pentagon base cell, under 1% of random
 *   pairs) report `counts = 0` and an empty path row, keeping their distance; about 1% of the
 *   others differ from H3 by one or two cells.
 *
 * Output rows are bounded by `maximumPathLength`; `counts` reports the true length so overflow is
 * `counts[row] > maximumPathLength`. Everything is integer arithmetic, deterministic on every device.
 */
export class GPUCellGridPath implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellGridPathProps;
  /** Cells stored per pair (0 without `output.cells`). */
  readonly maximumPathLength: number;

  constructor(props: GPUCellGridPathProps) {
    this.id = props.id ?? 'cell-grid-path';
    this.props = props;
    const id = this.id;
    if (props.family !== 'h3') {
      throw new Error(`${id} family must be 'h3'`);
    }
    if (props.wordOrder && props.wordOrder !== 'little-endian' && props.wordOrder !== 'high-low') {
      throw new Error(`${id} wordOrder must be 'little-endian' or 'high-low'`);
    }
    const {output} = props;
    for (const [name, view] of [
      ['origins', props.origins],
      ['destinations', props.destinations],
      ['mask', props.mask],
      ['output.distances', output.distances],
      ['output.cells', output.cells],
      ['output.counts', output.counts]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rows = props.origins.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    validatePackedView(props.origins, ['uint32x2'], `${id} origins`);
    validatePackedView(props.destinations, ['uint32x2'], `${id} destinations`);
    if (props.destinations.length !== rows) {
      throw new Error(`${id} destinations length must equal the origins length`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    if (!output.distances && !output.cells) {
      throw new Error(`${id} needs output.distances or output.cells`);
    }
    if (output.counts && !output.cells) {
      throw new Error(`${id} output.counts needs output.cells`);
    }
    if (output.distances) {
      validatePackedUint32View(output.distances, `${id} output.distances`);
      if (output.distances.length !== rows) {
        throw new Error(`${id} output.distances length must equal the row count`);
      }
    }
    this.maximumPathLength = 0;
    if (output.cells) {
      const length = props.maximumPathLength;
      if (
        length === undefined ||
        !Number.isInteger(length) ||
        length < 1 ||
        length > GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH
      ) {
        throw new Error(
          `${id} maximumPathLength must be an integer in [1, ${GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH}] with output.cells`
        );
      }
      this.maximumPathLength = length;
      validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
      if (output.cells.length !== rows * length) {
        throw new Error(
          `${id} output.cells length must be rows * maximumPathLength = ${rows * length}`
        );
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
      [output.distances, output.cells, output.counts],
      [props.origins, props.destinations, props.mask]
    );
  }

  /** Returns the single grid-path node `${id}-path`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumPathLength} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.origins,
      props.destinations,
      props.mask,
      ...Object.values(output)
    ]);
    const rows = props.origins.length;
    const bindings: WGSLKernelBinding[] = [
      {name: 'origins', view: props.origins, type: 'u32', access: 'read'},
      {name: 'destinations', view: props.destinations, type: 'u32', access: 'read'},
      ...(props.mask
        ? [{name: 'rowMask', view: props.mask, type: 'u32', access: 'read'} as const]
        : []),
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
      ...(output.cells
        ? [{name: 'outCells', view: output.cells, type: 'u32', access: 'read_write'} as const]
        : []),
      ...(output.counts
        ? [{name: 'outCounts', view: output.counts, type: 'u32', access: 'read_write'} as const]
        : [])
    ];
    const isHighLow = props.wordOrder === 'high-low';
    const readCell = (
      name: string
    ) => `let ${name}Words = vec2u(${name}s[${name}sOffset + 2u * index], ${name}s[${name}sOffset + 2u * index + 1u]);
  let ${name} = ${isHighLow ? `${name}Words` : `${name}Words.yx`};`;
    const writeCell = (slot: string, cell: string) => `
    outCells[outCellsOffset + 2u * ${slot}] = ${cell}.${isHighLow ? 'x' : 'y'};
    outCells[outCellsOffset + 2u * ${slot} + 1u] = ${cell}.${isHighLow ? 'y' : 'x'};`;

    const pathBody = output.cells
      ? `
  var pathLength = 0u;
  var isPathValid = false;
  if (pathDistance != GRID_DISTANCE_UNDEFINED) {
    isPathValid = true;
    let distanceValue = i32(pathDistance);
    let startCube = cellTopologyH3GetCube(originLocal.ijk);
    let delta = cellTopologyH3GetCube(destinationLocal.ijk) - startCube;
    // floor((delta * step) / distance), advanced one step at a time without overflow.
    var quotientStep = vec3i(0);
    var remainderStep = vec3i(0);
    if (distanceValue > 0) {
      quotientStep = delta / vec3i(distanceValue);
      remainderStep = delta - quotientStep * distanceValue;
      for (var axis = 0; axis < 3; axis++) {
        if (remainderStep[axis] < 0) {
          remainderStep[axis] += distanceValue;
          quotientStep[axis] -= 1;
        }
      }
    }
    cellGridPath[0] = origin;
    pathLength = 1u;
    var quotient = vec3i(0);
    var remainder = vec3i(0);
    var previous = origin;
    let storedLength = min(pathDistance + 1u, PATH_STRIDE);
    for (var pathStep = 1u; pathStep < storedLength; pathStep++) {
      quotient += quotientStep;
      remainder += remainderStep;
      for (var axis = 0; axis < 3; axis++) {
        if (remainder[axis] >= distanceValue) {
          remainder[axis] -= distanceValue;
          quotient[axis] += 1;
        }
      }
      let roundedTarget = cellTopologyH3RoundCubeStep(startCube, quotient, remainder, distanceValue);
      var matchCount = 0u;
      var matchedNeighbor = vec2u(0u);
      for (var direction = 1u; direction <= 6u; direction++) {
        let neighbor = cellTopologyH3Neighbor(previous, direction);
        if (neighbor.x == 0u && neighbor.y == 0u) {
          continue;
        }
        let neighborLocal = cellTopologyH3GetLocalIjk(origin, neighbor);
        if (
          neighborLocal.valid != 0u &&
          neighborLocal.ijk.x - neighborLocal.ijk.z == roundedTarget.x &&
          neighborLocal.ijk.y - neighborLocal.ijk.z == roundedTarget.y
        ) {
          matchCount++;
          matchedNeighbor = neighbor;
        }
      }
      if (matchCount != 1u) {
        isPathValid = false;
        break;
      }
      cellGridPath[pathStep] = matchedNeighbor;
      previous = matchedNeighbor;
      pathLength = pathStep + 1u;
    }
  }
  for (var slot = 0u; slot < PATH_STRIDE; slot++) {
    let row = index * PATH_STRIDE + slot;
    if (isPathValid && slot < pathLength) {${writeCell('row', 'cellGridPath[slot]')}
    } else {${writeCell('row', 'vec2u(0u)')}
    }
  }
  ${output.counts ? 'outCounts[outCountsOffset + index] = select(0u, pathDistance + 1u, isPathValid);' : ''}`
      : '';

    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-path`,
        operation: OPERATION,
        variant: `h3-${output.cells ? 'path' : 'distance'}`,
        bindings,
        invocationCount: rows,
        declarations: `const GRID_DISTANCE_UNDEFINED: u32 = 0xffffffffu;
const PATH_STRIDE: u32 = ${Math.max(maximumPathLength, 1)}u;
var<private> cellGridPath: array<vec2u, PATH_STRIDE>;
${dggs.source}
${CELL_KEY_WGSL}
${H3_NEIGHBOR_WGSL}
${H3_LOCAL_IJ_WGSL}`,
        body: `${readCell('origin')}
  ${readCell('destination')}
  var isValid = true;
  ${props.mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  var pathDistance = GRID_DISTANCE_UNDEFINED;
  var originLocal = CellTopologyH3LocalIjk(vec3i(0), 0u);
  var destinationLocal = CellTopologyH3LocalIjk(vec3i(0), 0u);
  if (isValid) {
    originLocal = cellTopologyH3GetLocalIjk(origin, origin);
    destinationLocal = cellTopologyH3GetLocalIjk(origin, destination);
    if (originLocal.valid != 0u && destinationLocal.valid != 0u) {
      pathDistance = cellTopologyH3GetGridDistance(originLocal.ijk, destinationLocal.ijk);
    }
  }
  ${output.distances ? 'outDistances[outDistancesOffset + index] = pathDistance;' : ''}
  ${pathBody}`
      })
    ];
  }
}
