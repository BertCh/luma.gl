// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {GPUSpatialWeightsTranspose} from '../spatial-weights/gpu-spatial-weights-transpose';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';

/** Lanes of the per-column standardization workgroup. */
const STANDARDIZE_LANES = 256;

/**
 * Number of `uint32` words of the seed-only per-frame parameters shared by `GPUAZPRegions`,
 * `GPUWardRegions` and `GPUMaxPRegions`: `[seed]`.
 *
 * @internal
 */
export const REGIONALIZATION_SEED_PARAMETER_LENGTH = 1;

/**
 * Packs the `[seed]` parameter word of a seeded regionalization contributor.
 *
 * @param operation Contributor class name, for the message.
 * @param seed Tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0.
 * @internal
 */
export function getRegionalizationSeedParameterValues(
  operation: string,
  seed: number = 0
): Uint32Array {
  validateCapacity(operation, 'seed', seed, 0);
  return Uint32Array.of(seed);
}

/**
 * Largest `weights.neighbors` slot count of the AZP and Ward regionalizations on `device`: the
 * undirected adjacency they build holds up to two uint32 neighbors (8 bytes) per input slot in one
 * storage binding, so 16777216 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPURegionalizationMaximumSlotCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Validates that a regionalization contributor's weights fit `maximum` slots, the value of
 * {@link getGPURegionalizationMaximumSlotCount} unless the contributor needs fewer.
 *
 * @internal
 */
export function validateRegionalizationSlotCount(
  id: string,
  weights: GPUSpatialWeights,
  maximum: number
): void {
  validateMaximumCount(id, 'weights.neighbors rows', weights.neighbors.length, maximum);
}

/**
 * Validates the per-frame `uint32` parameter view of a regionalization contributor.
 *
 * @internal
 */
export function validateRegionalizationParameters(
  id: string,
  parameters: GraphDataView,
  length: number
): void {
  validatePackedView(parameters, ['uint32'], `${id} parameters`);
  if (parameters.length < length) {
    throw new Error(`${id} parameters must hold ${length} words`);
  }
}

/**
 * Standardizes every column of a row-major attribute table to mean 0 and population standard
 * deviation 1 (divisor `n`, the `sklearn.preprocessing.StandardScaler` convention, not the sample
 * divisor `n - 1`). Statistics use the finite entries of the column only, so one missing value
 * does not turn the whole column into NaN; non-finite entries stay non-finite in the output. A
 * constant column becomes all zero.
 *
 * One workgroup per column: each lane sums a strided share of the rows and a fixed binary tree
 * merges the lanes (deterministic order); the variance is a second pass about the mean.
 *
 * @internal
 */
export function createStandardizeColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView<'float32'>;
    scaled: GraphDataView<'float32'>;
    rowCount: number;
    columnCount: number;
  }
): GPUCommandNode<Parameters> {
  const {rowCount, columnCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'standardize',
    bindings: [
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'scaled', view: props.scaled, type: 'f32', access: 'read_write'}
    ],
    workgroupSize: STANDARDIZE_LANES,
    invocationCount: columnCount * STANDARDIZE_LANES,
    guardIndex: false,
    declarations: `const ROWS: u32 = ${rowCount}u;
const COLUMNS: u32 = ${columnCount}u;
const LANES: u32 = ${STANDARDIZE_LANES}u;
${IS_FINITE_FLOAT_WGSL}
var<workgroup> partials: array<f32, ${STANDARDIZE_LANES}>;

// Sum of one value per lane over the workgroup, available to every lane.
fn reduceLanes(lane: u32, value: f32) -> f32 {
  partials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partials[lane] = partials[lane] + partials[lane + stride];
    }
    workgroupBarrier();
  }
  let total = partials[0];
  workgroupBarrier();
  return total;
}`,
    body: `let column = index / LANES;
  let lane = localInvocationIndex;
  var sum = 0.0;
  var count = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(value)) { sum += value; count += 1.0; }
  }
  let finiteCount = max(reduceLanes(lane, count), 1.0);
  let mean = reduceLanes(lane, sum) / finiteCount;
  var squares = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(value)) { squares += (value - mean) * (value - mean); }
  }
  let deviation = sqrt(reduceLanes(lane, squares) / finiteCount);
  let inverse = select(0.0, 1.0 / deviation, deviation > 1e-20);
  for (var row = lane; row < ROWS; row += LANES) {
    scaled[scaledOffset + row * COLUMNS + column] =
      (values[valuesOffset + row * COLUMNS + column] - mean) * inverse;
  }`
  });
}

/**
 * Builds the undirected union of `W` and `W'` as a CSR (each row lists the `W` neighbors followed
 * by the transposed neighbors; a link present in both directions appears twice, which every
 * consumer tolerates because they only ask whether some neighbor satisfies a condition). Neighbor
 * IDs outside `[0, rowCount)` are replaced by the row itself (a self link is never a candidate),
 * and the caller reports them separately as invalid input from the original weights.
 *
 * When `weights.information?.symmetric === true` the input is returned unchanged and no node is added.
 *
 * @internal
 */
export function createSymmetricAdjacency<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; weights: GPUSpatialWeights; rowCount: number}
): {weights: GPUSpatialWeights; nodes: GPUCommandNode<Parameters>[]} {
  const {id, operation, weights, rowCount} = props;
  if (weights.information?.symmetric === true) {
    return {weights, nodes: []};
  }
  const capacity = weights.neighbors.length;
  const transposeOffsets = createTransientView(
    graph,
    `${id}-adjacency-t-offsets`,
    'uint32',
    rowCount + 1
  );
  const transposeNeighbors = createTransientView(
    graph,
    `${id}-adjacency-t-neighbors`,
    'uint32',
    capacity
  );
  const transposeWeights = createTransientView(
    graph,
    `${id}-adjacency-t-weights`,
    'float32',
    capacity
  );
  const adjacencyOffsets = createTransientView(
    graph,
    `${id}-adjacency-offsets`,
    'uint32',
    rowCount + 1
  );
  const adjacencyNeighbors = createTransientView(
    graph,
    `${id}-adjacency-neighbors`,
    'uint32',
    2 * capacity
  );
  const adjacencyWeights = createTransientView(
    graph,
    `${id}-adjacency-weights`,
    'float32',
    2 * capacity
  );
  const declarations = `const ROWS: u32 = ${rowCount}u;
const CAPACITY: u32 = ${2 * capacity}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...new GPUSpatialWeightsTranspose({
      id: `${id}-adjacency-transpose`,
      // Only the pattern matters: drop information, status and distances.
      weights: {offsets: weights.offsets, neighbors: weights.neighbors, weights: weights.weights},
      output: {
        offsets: transposeOffsets,
        neighbors: transposeNeighbors,
        weights: transposeWeights
      }
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-adjacency-offsets`,
      operation,
      variant: 'adjacency-offsets',
      bindings: [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'transposeOffsets', view: transposeOffsets, type: 'u32', access: 'read'},
        {name: 'adjacencyOffsets', view: adjacencyOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount + 1,
      body: 'adjacencyOffsets[adjacencyOffsetsOffset + index] = offsets[offsetsOffset + index] + transposeOffsets[transposeOffsetsOffset + index];'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-adjacency-neighbors`,
      operation,
      variant: 'adjacency-neighbors',
      bindings: [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'transposeOffsets', view: transposeOffsets, type: 'u32', access: 'read'},
        {name: 'transposeNeighbors', view: transposeNeighbors, type: 'u32', access: 'read'},
        {name: 'adjacencyOffsets', view: adjacencyOffsets, type: 'u32', access: 'read'},
        {name: 'adjacencyNeighbors', view: adjacencyNeighbors, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 2 * capacity,
      declarations: `${declarations}
// Row that owns adjacency slot \`slot\`: the first row whose range ends after it.
fn findAdjacencyRow(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (adjacencyOffsets[adjacencyOffsetsOffset + middle + 1u] > slot) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}`,
      body: `var neighbor = 0u;
  if (index < adjacencyOffsets[adjacencyOffsetsOffset + ROWS]) {
    let row = findAdjacencyRow(index);
    let local = index - adjacencyOffsets[adjacencyOffsetsOffset + row];
    let ownCount = offsets[offsetsOffset + row + 1u] - offsets[offsetsOffset + row];
    if (local < ownCount) {
      neighbor = neighbors[neighborsOffset + offsets[offsetsOffset + row] + local];
      if (neighbor >= ROWS) { neighbor = row; }
    } else {
      neighbor = transposeNeighbors[transposeNeighborsOffset + transposeOffsets[transposeOffsetsOffset + row] + local - ownCount];
    }
  }
  adjacencyNeighbors[adjacencyNeighborsOffset + index] = neighbor;`
    })
  ];
  return {
    weights: describeGPUSpatialWeights(
      {offsets: adjacencyOffsets, neighbors: adjacencyNeighbors, weights: adjacencyWeights},
      {
        information: {
          rowCount,
          transform: 'unknown',
          symmetric: true
        }
      }
    ),
    nodes
  };
}
