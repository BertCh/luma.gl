// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSaturatedScanTotalNodes} from '../../utils/scan-total';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validateGPUStatusPort} from '../../analysis-core/contracts/index';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights
} from '../../gpu-spatial-analysis/spatial-weights/spatial-weights';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUNetworkCostWeights';

/** Workgroup size of the per-row count and fill kernels. Fixed, so slot order is fixed. */
const ROW_WORKGROUP_SIZE = 256;

/** Number of `float32` words in the per-frame parameter view of {@link GPUNetworkCostWeights}. */
export const GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH = 4;

/**
 * Distance-decay function `f(c)` of {@link GPUNetworkCostWeights}, applied to every cost `c` within
 * `costLimit`:
 * - `'none'`: 1, the binary catchment of classic 2SFCA (Luo and Wang 2003).
 * - `'exponential'`: `exp(-beta * c)`.
 * - `'power'`: `max(c, minimumCost) ^ -beta`.
 * - `'gaussian'`: `(exp(-c^2 / (2 L^2)) - exp(-1/2)) / (1 - exp(-1/2))` with `L = costLimit`, the
 *   catchment kernel of Dai (2010) used by enhanced 2SFCA and 3SFCA (Wan, Zou and Sternberg 2012).
 *   It is 1 at `c = 0` and 0 at `c = L`.
 */
export type GPUNetworkCostWeightsDecay = 'none' | 'exponential' | 'power' | 'gaussian';

/** Per-frame settings of {@link GPUNetworkCostWeights}, packed by {@link getGPUNetworkCostWeightsParameterValues}. */
export type GPUNetworkCostWeightsSettings = {
  /**
   * Catchment cost: an entry becomes a neighbor when its cost is finite, non-negative and
   * `<= costLimit`. `Infinity` keeps every reached entry (not allowed with `'gaussian'`).
   */
  costLimit: number;
  /** Decay that turns a cost into a weight. Defaults to `'none'`. */
  decay?: GPUNetworkCostWeightsDecay;
  /** Rate of the `'exponential'` and `'power'` decays, finite and non-negative. Defaults to 0. */
  beta?: number;
  /** Floor of the `'power'` decay so a zero cost stays finite, positive and finite. Defaults to 1. */
  minimumCost?: number;
};

const DECAY_CODES: Record<GPUNetworkCostWeightsDecay, number> = {
  none: 0,
  exponential: 1,
  power: 2,
  gaussian: 3
};

/**
 * Packs {@link GPUNetworkCostWeightsSettings} into the `GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH`
 * `float32` words read by {@link GPUNetworkCostWeights}: cost limit, decay code, beta, minimum cost.
 *
 * @throws If `costLimit` is NaN or negative, `'gaussian'` has no finite positive `costLimit`, the
 * decay is unknown, `beta` is negative or not finite, or `minimumCost` is not positive and finite.
 */
export function getGPUNetworkCostWeightsParameterValues(
  settings: GPUNetworkCostWeightsSettings
): Float32Array {
  const decay = settings.decay ?? 'none';
  const beta = settings.beta ?? 0;
  const minimumCost = settings.minimumCost ?? 1;
  const {costLimit} = settings;
  if (!(costLimit >= 0)) {
    throw new Error(`${OPERATION} costLimit must be a non-negative number`);
  }
  if (!(decay in DECAY_CODES)) {
    throw new Error(`${OPERATION} decay must be 'none', 'exponential', 'power' or 'gaussian'`);
  }
  if (decay === 'gaussian' && !(Number.isFinite(costLimit) && costLimit > 0)) {
    throw new Error(`${OPERATION} 'gaussian' decay needs a finite positive costLimit`);
  }
  if (!Number.isFinite(beta) || beta < 0) {
    throw new Error(`${OPERATION} beta must be a non-negative finite number`);
  }
  if (!Number.isFinite(minimumCost) || !(minimumCost > 0)) {
    throw new Error(`${OPERATION} minimumCost must be a positive finite number`);
  }
  return Float32Array.from([costLimit, DECAY_CODES[decay], beta, minimumCost]);
}

/**
 * Caller-owned outputs of {@link GPUNetworkCostWeights}: the views of a {@link GPUSpatialWeights}
 * plus its capacity status.
 */
export type GPUNetworkCostWeightsOutput = {
  /** `rowCount + 1` exclusive row offsets, clamped to the slot capacity. */
  offsets: GraphDataView<'uint32'>;
  /** Neighbor ID per slot, ascending within each row. Its length is the slot capacity. */
  neighbors: GraphDataView<'uint32'>;
  /** Decay weight `f(c)` per slot, aligned with `neighbors`. */
  weights: GraphDataView<'float32'>;
  /** Optional network cost `c` per slot, aligned with `neighbors`. */
  distances?: GraphDataView<'float32'>;
  /** Optional GPU-written status words. */
  status?: {
    /** Optional one-row flag: 1 when the in-range entries exceed the slot capacity. */
    overflow?: GraphDataView<'uint32'>;
    /** Optional one-row exact (unclamped) number of in-range entries. */
    requiredCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkCostWeights}.
 *
 * Compile-time: the matrix shape, `columns` length, orientation and capacity. Per-frame: the costs,
 * the column map contents and `parameters` (cost limit, decay, beta, minimum cost).
 */
export type GPUNetworkCostWeightsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-cost-weights'`. */
  id?: string;
  /**
   * Row-major `[rowCount x columnCount]` network costs, `+Infinity` (or any non-finite or negative
   * value) where unreached: the `output.costs` of `GPUNetworkCostMatrix` (one row per start row),
   * or of `GPUNetworkReachability` (one row per lane).
   */
  costs: GraphDataView<'float32'>;
  /** Number of matrix rows; the column count is `costs.length / rowCount`. Defaults to 1. */
  rowCount?: number;
  /**
   * Optional matrix column per location, for example the network node each demand location is
   * snapped to. Location `k` reads column `columns[k]` (out-of-range columns are unreached), so the
   * neighbor IDs of the weights index locations instead of nodes. Defaults to every column, where
   * location `k` is column `k`.
   */
  columns?: GraphDataView<'uint32'>;
  /**
   * Which side becomes the weights rows. Defaults to `'rows'`.
   * - `'rows'`: row `r` of the weights is matrix row `r`, its neighbors are locations.
   * - `'columns'`: the transpose, row `k` is location `k`, its neighbors are matrix rows.
   */
  orientation?: 'rows' | 'columns';
  /**
   * Per-frame parameters: `GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH` `float32` words written with
   * {@link getGPUNetworkCostWeightsParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned weights views and capacity status. */
  output: GPUNetworkCostWeightsOutput;
};

/**
 * Turns a network cost matrix or reachability result into {@link GPUSpatialWeights} with a
 * distance decay, so any weights consumer runs on network costs. With facilities as the matrix rows,
 * `orientation: 'rows'` gives the facility weights and `'columns'` their exact transpose, the demand
 * weights, of `GPUCatchmentAccessibility` (2SFCA and 3SFCA on network costs).
 *
 * Every entry whose cost is finite, non-negative and within `costLimit` becomes one slot, with the
 * decay value as its weight and the cost as its distance. Rows are laid out in row order and slots
 * in ascending neighbor order, so the weights are deterministic. A zero cost is kept (a facility at
 * its own demand location), and a decay may give a slot weight 0 (`'gaussian'` at `costLimit`),
 * which weights consumers skip.
 *
 * Capacity: `output.neighbors.length` slots. `rowCount x columnCount` (or locations) slots always
 * suffice. Slots past the capacity are dropped and the offsets are clamped, so a reader always sees
 * a consistent (possibly truncated) CSR; `output.status.overflow` and `output.status.requiredCount` report it, for the grow loop
 * of `getGPUSpatialWeightsCapacity`.
 *
 * Passes: a per-row count (one workgroup per row), a `GPUScan` into `offsets`, a per-row ordered
 * fill and a finalize pass that clamps the offsets and writes the status.
 */
export class GPUNetworkCostWeights implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkCostWeightsProps;
  /** Resolved orientation. */
  readonly orientation: 'rows' | 'columns';
  /** Number of matrix rows. */
  readonly rowCount: number;
  /** Number of matrix columns, `costs.length / rowCount`. */
  readonly columnCount: number;
  /** Number of locations: `columns.length`, or `columnCount` without a column map. */
  readonly locationCount: number;
  /** Number of weights rows: `rowCount` for `'rows'`, `locationCount` for `'columns'`. */
  readonly weightsRowCount: number;
  /** Slot capacity, `output.neighbors.length`. */
  readonly capacity: number;
  /** The output weights with `information` and `status` attached, for consumers. */
  readonly outputWeights: GPUSpatialWeights;

  /**
   * Validates the matrix shape and output layout.
   *
   * @throws If a view has the wrong format or length, `rowCount` does not divide `costs.length`, the
   * orientation is unknown, or an output shares a buffer with an input.
   */
  constructor(props: GPUNetworkCostWeightsProps) {
    const id = props.id ?? 'network-cost-weights';
    this.id = id;
    this.props = props;
    this.orientation = props.orientation ?? 'rows';
    if (this.orientation !== 'rows' && this.orientation !== 'columns') {
      throw new Error(`${id} orientation must be 'rows' or 'columns'`);
    }
    const {output} = props;
    validatePackedView(props.costs, ['float32'], `${id} costs`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length !== GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH} rows`
      );
    }
    this.rowCount = props.rowCount ?? 1;
    validateCapacity(id, 'rowCount', this.rowCount, 1, props.costs.length);
    if (props.costs.length % this.rowCount !== 0) {
      throw new Error(`${id} costs length must be a multiple of rowCount`);
    }
    this.columnCount = props.costs.length / this.rowCount;
    if (props.columns) {
      validatePackedView(props.columns, ['uint32'], `${id} columns`);
    }
    this.locationCount = props.columns?.length ?? this.columnCount;
    validateCapacity(id, 'columns length', this.locationCount);
    this.weightsRowCount = this.orientation === 'rows' ? this.rowCount : this.locationCount;

    validatePackedView(output.offsets, ['uint32'], `${id} output.offsets`);
    if (output.offsets.length !== this.weightsRowCount + 1) {
      throw new Error(`${id} output.offsets must contain one row per weights row plus one`);
    }
    validatePackedView(output.neighbors, ['uint32'], `${id} output.neighbors`);
    this.capacity = output.neighbors.length;
    validateCapacity(id, 'output.neighbors length', this.capacity);
    for (const [name, view] of [
      ['output.weights', output.weights],
      ['output.distances', output.distances]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== this.capacity) {
          throw new Error(`${id} ${name} length must equal output.neighbors length`);
        }
      }
    }
    validateGPUStatusPort(id, {
      overflow: output.status?.overflow,
      requiredCount: output.status?.requiredCount
    });
    validateGraphOutputsDisjointFromInputs(id, getOutputs(props), getInputs(props));

    this.outputWeights = describeGPUSpatialWeights(
      {
        offsets: output.offsets,
        neighbors: output.neighbors,
        weights: output.weights,
        distances: output.distances
      },
      {
        information: {
          rowCount: this.weightsRowCount,
          transform: 'kernel',
          symmetric: 'unknown',
          provenance: {source: OPERATION, criterion: 'network-cost'}
        },
        status: {overflow: output.status?.overflow, requiredCount: output.status?.requiredCount}
      }
    );
  }

  /** Returns the count, scan, fill and finalize nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const counts = createTransientView(
      graph,
      `${id}-row-counts`,
      'uint32',
      this.weightsRowCount + 1
    );
    const scanTotal = createTransientView(graph, `${id}-scan-total`, 'uint32', 1);
    const declarations = this.getDeclarations();
    const inputBindings: WGSLKernelBinding[] = [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.columns) {
      inputBindings.push({name: 'columns', view: props.columns, type: 'u32', access: 'read'});
    }
    const size = ROW_WORKGROUP_SIZE;

    const countNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-count`,
      operation: OPERATION,
      variant: 'count',
      bindings: [
        ...inputBindings,
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      workgroupSize: size,
      // One workgroup per weights row, plus one that writes the trailing zero the scan turns into
      // the total.
      invocationCount: (this.weightsRowCount + 1) * size,
      guardIndex: false,
      declarations: `${declarations}
var<workgroup> partials: array<u32, ${size}>;`,
      // No early return: every invocation of a workgroup must reach the barriers.
      body: `let row = index / ${size}u;
  let costLimit = parameters[parametersOffset];
  var count = 0u;
  if (row < WEIGHTS_ROWS) {
    for (var neighbor = localInvocationIndex; neighbor < NEIGHBOR_SPACE; neighbor += ${size}u) {
      if (isWithinCostLimit(getEntryCost(row, neighbor), costLimit)) {
        count += 1u;
      }
    }
  }
  partials[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${size / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && row <= WEIGHTS_ROWS) {
    counts[countsOffset + row] = partials[0];
  }`
    });

    const scanNodes = new GPUScan({
      id: `${id}-offsets`,
      input: counts,
      output: output.offsets,
      mode: 'exclusive'
    }).getCommandNodes(graph);

    const fillBindings: WGSLKernelBinding[] = [
      ...inputBindings,
      {name: 'offsets', view: output.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: output.neighbors, type: 'u32', access: 'read_write'},
      {name: 'weights', view: output.weights, type: 'f32', access: 'read_write'}
    ];
    if (output.distances) {
      fillBindings.push({
        name: 'distances',
        view: output.distances,
        type: 'f32',
        access: 'read_write'
      });
    }
    const fillNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-fill`,
      operation: OPERATION,
      variant: 'fill',
      bindings: fillBindings,
      workgroupSize: size,
      invocationCount: this.weightsRowCount * size,
      guardIndex: false,
      declarations: `${declarations}
var<workgroup> prefix: array<u32, ${size}>;
fn getDecayWeight(cost: f32, costLimit: f32, decay: u32, beta: f32, minimumCost: f32) -> f32 {
  if (decay == 1u) {
    return exp(-beta * cost);
  }
  if (decay == 2u) {
    return pow(max(cost, minimumCost), -beta);
  }
  if (decay == 3u) {
    let ratio = cost / costLimit;
    let edge = exp(-0.5);
    return max((exp(-0.5 * ratio * ratio) - edge) / (1.0 - edge), 0.0);
  }
  return 1.0;
}`,
      // A chunk of 256 neighbors at a time: an inclusive Hillis-Steele scan of the in-range flags
      // gives each in-range entry its slot, so slots follow neighbor order. No early return: every
      // invocation must reach the barriers.
      body: `let row = index / ${size}u;
  let isActiveRow = row < WEIGHTS_ROWS;
  let costLimit = parameters[parametersOffset];
  let decay = u32(max(parameters[parametersOffset + 1u], 0.0));
  let beta = parameters[parametersOffset + 2u];
  let minimumCost = parameters[parametersOffset + 3u];
  var base = 0u;
  if (isActiveRow) {
    base = offsets[offsetsOffset + row];
  }
  for (var chunk = 0u; chunk < NEIGHBOR_SPACE; chunk += ${size}u) {
    let neighbor = chunk + localInvocationIndex;
    var cost = -1.0;
    if (isActiveRow && neighbor < NEIGHBOR_SPACE) {
      cost = getEntryCost(row, neighbor);
    }
    let isIncluded = isWithinCostLimit(cost, costLimit);
    prefix[localInvocationIndex] = select(0u, 1u, isIncluded);
    workgroupBarrier();
    for (var stride = 1u; stride < ${size}u; stride = stride * 2u) {
      var addend = 0u;
      if (localInvocationIndex >= stride) {
        addend = prefix[localInvocationIndex - stride];
      }
      workgroupBarrier();
      prefix[localInvocationIndex] += addend;
      workgroupBarrier();
    }
    let slot = base + prefix[localInvocationIndex] - 1u;
    if (isIncluded && slot < CAPACITY) {
      neighbors[neighborsOffset + slot] = neighbor;
      weights[weightsOffset + slot] = getDecayWeight(cost, costLimit, decay, beta, minimumCost);
      ${output.distances ? 'distances[distancesOffset + slot] = cost;' : ''}
    }
    base += prefix[${size - 1}u];
    workgroupBarrier();
  }`
    });

    const finalizeBindings: WGSLKernelBinding[] = [
      {name: 'scanTotal', view: scanTotal, type: 'u32', access: 'read'},
      {name: 'offsets', view: output.offsets, type: 'u32', access: 'read_write'}
    ];
    if (output.status?.overflow) {
      finalizeBindings.push({
        name: 'overflow',
        view: output.status.overflow,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (output.status?.requiredCount) {
      finalizeBindings.push({
        name: 'requiredCount',
        view: output.status.requiredCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    // Each invocation touches only its own offset, so clamping cannot race with reading the total.
    const finalizeNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-finalize`,
      operation: OPERATION,
      variant: 'finalize',
      bindings: finalizeBindings,
      invocationCount: this.weightsRowCount + 1,
      declarations: `const WEIGHTS_ROWS: u32 = ${this.weightsRowCount}u;
const CAPACITY: u32 = ${this.capacity}u;`,
      body: `let value = offsets[offsetsOffset + index];
  if (index == WEIGHTS_ROWS) {
    let total = scanTotal[scanTotalOffset];
    ${output.status?.overflow ? 'overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);' : ''}
    ${output.status?.requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''}
  }
  offsets[offsetsOffset + index] = min(value, CAPACITY);`
    });

    return [
      countNode,
      ...scanNodes,
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: id,
        operation: OPERATION,
        counts: counts,
        offsets: output.offsets,
        rowCount: this.weightsRowCount,
        total: scanTotal
      }),
      fillNode,
      finalizeNode
    ];
  }

  /** Shape constants and the entry-cost and in-range helpers shared by the count and fill kernels. */
  private getDeclarations(): string {
    const isRows = this.orientation === 'rows';
    const neighborSpace = isRows ? this.locationCount : this.rowCount;
    // Location `k` of `'rows'` (row = matrix row) or `'columns'` (row = location) reads its column.
    const location = isRows ? 'neighbor' : 'row';
    const matrixRow = isRows ? 'row' : 'neighbor';
    const column = this.props.columns ? `columns[columnsOffset + ${location}]` : location;
    return `const WEIGHTS_ROWS: u32 = ${this.weightsRowCount}u;
const NEIGHBOR_SPACE: u32 = ${neighborSpace}u;
const MATRIX_COLUMNS: u32 = ${this.columnCount}u;
const CAPACITY: u32 = ${this.capacity}u;
${IS_FINITE_BITS_WGSL}
fn getEntryCost(row: u32, neighbor: u32) -> f32 {
  let column = ${column};
  if (column >= MATRIX_COLUMNS) {
    return -1.0;
  }
  return costs[costsOffset + ${matrixRow} * MATRIX_COLUMNS + column];
}
fn isWithinCostLimit(cost: f32, costLimit: f32) -> bool {
  // Unreached entries are +Infinity; an infinite limit must still reject them.
  return isFiniteBits(cost) && cost >= 0.0 && cost <= costLimit;
}`;
  }
}

/** Returns every read-only view of a cost-weights contributor. */
function getInputs(props: GPUNetworkCostWeightsProps): (GraphDataView | undefined)[] {
  return [props.costs, props.parameters, props.columns];
}

/** Returns every written view of a cost-weights contributor. */
function getOutputs(props: GPUNetworkCostWeightsProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.offsets,
    output.neighbors,
    output.weights,
    output.distances,
    output.status?.overflow,
    output.status?.requiredCount
  ];
}
