// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getWorkgroupArgminWGSL} from './regionalization-wgsl';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {PHILOX_WGSL} from '../../utils/philox-random';
import {GPUAZPRegions, GPU_AZP_MAXIMUM_ITERATIONS, GPU_AZP_MAXIMUM_ROWS} from './gpu-azp-regions';
import {createStandardizeColumnsNode, createSymmetricAdjacency} from './regionalization-inputs';

export const GPU_MAX_P_MAXIMUM_COLUMNS = 8;
/** Number of independent constructions run when `startCount` is not given. */
export const GPU_MAX_P_DEFAULT_START_COUNT = 16;
/** Largest `startCount`: each start owns a scratch label row and region table of the input size. */
export const GPU_MAX_P_MAXIMUM_START_COUNT = 64;
const MAX_P_WORKGROUP_SIZE = 256;
const START_STATE_STRIDE = 4;
/** Fallback for the largest storage binding when the device does not report one (128 MiB). */
const DEFAULT_MAXIMUM_BINDING_BYTES = 128 * 1024 * 1024;

export const GPU_MAX_P_STATUS = {
  /** Number of regions of the selected start (including under-threshold leftovers). */
  regionCount: 0,
  /** 1 when every region reaches the threshold and the input is valid. */
  feasible: 1,
  /** Regions of the selected start that stay below the threshold. */
  infeasibleRegionCount: 2,
  invalidCount: 3,
  /** Index of the start whose construction was returned. */
  selectedStart: 4,
  /** Moves made by the local-improvement phase (0 when `improvementIterations` is 0). */
  improvementMoveCount: 5,
  length: 6
} as const;

export type GPUMaxPRegionsOutput = {
  labels: GraphDataView<'uint32'>;
  /**
   * Total within-region sum of squared deviations of the returned partition (standardized
   * units unless `standardize` is false), the same quantity as `withinSsd` of
   * `GPURegionPartitionEvaluation` on the same table.
   */
  objective: GraphDataView<'float32'>;
  /** Six words indexed by {@link GPU_MAX_P_STATUS}. */
  status: GraphDataView<'uint32'>;
};

export type GPUMaxPRegionsProps = {
  id?: string;
  /**
   * Spatial weights used for contiguous region growth. Two rows are adjacent when either lists
   * the other, so directed (for example k-nearest-neighbor) weights are symmetrized inside the
   * contributor; weights that declare `info.symmetric === true` are used as they are.
   */
  weights: GPUSpatialWeights;
  /** Row-major feature values used to choose the next region member. */
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /**
   * Standardize every column to mean 0 and population standard deviation 1 (divisor `n`, the
   * `StandardScaler` convention, over finite entries) before growth, so columns in different
   * units weigh equally. Defaults to true; pass false when `values` is already on a common scale.
   */
  standardize?: boolean;
  /** Non-negative extensive value per row. */
  extensiveValues: GraphDataView<'float32'>;
  /** One per-frame minimum extensive value for a feasible region. */
  threshold: GraphDataView<'float32'>;
  /**
   * Compile-time number of independent constructions (default
   * {@link GPU_MAX_P_DEFAULT_START_COUNT}, at most {@link GPU_MAX_P_MAXIMUM_START_COUNT}).
   */
  startCount?: number;
  /**
   * Compile-time maximum moves of the optional local-improvement phase (0, the default, skips it;
   * at most 256). It runs `GPUAZPRegions` on the selected construction with the extensive
   * threshold as a move constraint, so the region count never changes, and it needs at most 65,536
   * rows.
   */
  improvementIterations?: number;
  /** Reproducible seed of the random seed-row order of every start. */
  seed: number;
  output: GPUMaxPRegionsOutput;
};

/**
 * Randomized multi-start greedy Max-P regionalization.
 *
 * Each of `startCount` independent constructions grows one contiguous region at a time until its
 * extensive sum reaches the threshold, selecting the adjacent row nearest to the region mean. The
 * next seed row is the unassigned row with the smallest Philox 4x32-10 key of `(row; seed,
 * start)`, a genuinely random seed order per start. An under-threshold remainder is merged into an
 * adjacent completed region when possible. All starts run in one dispatch, one workgroup each,
 * with their own scratch labels and region table, so the starts cost wall-clock time of about one
 * construction when the GPU has that many free compute units.
 *
 * **Start selection.** The returned start has the fewest under-threshold leftover regions (a
 * partition whose regions all reach the threshold beats one that does not), then the most
 * regions (the Max-P objective), then the smallest within-region sum of squares, then the lowest
 * start index. The result is deterministic for a given `seed` and `startCount`.
 *
 * **Local improvement.** With `improvementIterations > 0` the selected construction is refined by
 * greedy steepest-descent AZP moves under the extensive-threshold constraint (a region never
 * drops below the threshold), which lowers the within-region sum of squares at the same region
 * count. spopt's `MaxPHeuristic` instead anneals; this is the greedy variant, off by default
 * because it uses the AZP candidate search (an `O(rowCount)` connectivity check per improving
 * candidate, at most 65,536 rows).
 *
 * `output.objective` is the within-region sum of squared deviations of the returned labels (in
 * standardized units unless `standardize` is false).
 *
 * Default change: columns are standardized (`standardize`, default true); the previous raw-value
 * behavior is `standardize: false`. The previous single hash-ordered construction is `startCount: 1`
 * up to the change of seed order (now Philox).
 *
 * Scale limit: each construction is sequential and runs as one 256-lane workgroup. Every growth
 * step rescans all rows and their neighbor lists, so one start takes about
 * `rowCount * (rowCount + neighborCount) / 256` steps per lane, and the starts run side by side,
 * not faster. That is fine for thousands of rows, but from roughly 50k-100k rows one dispatch can
 * run for seconds and trip the operating system's GPU watchdog (device loss). Partition larger
 * inputs first, or use SKATER. Scratch memory is `startCount * rowCount * (columnCount + 4) * 4`
 * bytes and must fit one storage binding.
 */
export class GPUMaxPRegions implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMaxPRegionsProps;
  readonly rowCount: number;
  readonly columnCount: number;
  /** Number of independent constructions. */
  readonly startCount: number;
  /** Maximum moves of the local-improvement phase. */
  readonly improvementIterations: number;

  constructor(props: GPUMaxPRegionsProps) {
    this.id = props.id ?? 'max-p-regions';
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(this.id, props.weights);
    this.columnCount = props.columnCount ?? 1;
    this.startCount = props.startCount ?? GPU_MAX_P_DEFAULT_START_COUNT;
    this.improvementIterations = props.improvementIterations ?? 0;
    const {id, rowCount, columnCount, startCount, improvementIterations} = this;
    const {output} = props;
    if (
      !Number.isSafeInteger(columnCount) ||
      columnCount < 1 ||
      columnCount > GPU_MAX_P_MAXIMUM_COLUMNS
    ) {
      throw new Error(`${id} columnCount must be an integer in [1, ${GPU_MAX_P_MAXIMUM_COLUMNS}]`);
    }
    if (
      !Number.isSafeInteger(startCount) ||
      startCount < 1 ||
      startCount > GPU_MAX_P_MAXIMUM_START_COUNT
    ) {
      throw new Error(
        `${id} startCount must be an integer in [1, ${GPU_MAX_P_MAXIMUM_START_COUNT}]`
      );
    }
    if (
      !Number.isSafeInteger(improvementIterations) ||
      improvementIterations < 0 ||
      improvementIterations > GPU_AZP_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} improvementIterations must be an integer in [0, ${GPU_AZP_MAXIMUM_ITERATIONS}]`
      );
    }
    if (improvementIterations > 0 && (rowCount < 2 || rowCount > GPU_AZP_MAXIMUM_ROWS)) {
      throw new Error(
        `${id} improvementIterations needs between 2 and ${GPU_AZP_MAXIMUM_ROWS} rows`
      );
    }
    if (!Number.isSafeInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff) {
      throw new Error(`${id} seed must be a uint32`);
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.extensiveValues, ['float32'], `${id} extensiveValues`);
    validatePackedView(props.threshold, ['float32'], `${id} threshold`);
    if (props.values.length !== rowCount * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    if (props.extensiveValues.length !== rowCount) {
      throw new Error(`${id} extensiveValues length must equal the weights row count`);
    }
    if (props.threshold.length < 1) {
      throw new Error(`${id} threshold must hold one value`);
    }
    validatePackedUint32View(output.labels, `${id} output.labels`);
    validatePackedUint32View(output.status, `${id} output.status`);
    validatePackedView(output.objective, ['float32'], `${id} output.objective`);
    if (output.labels.length !== rowCount) {
      throw new Error(`${id} output.labels length must equal the weights row count`);
    }
    if (output.objective.length < 1) {
      throw new Error(`${id} output.objective must hold one value`);
    }
    if (output.status.length < GPU_MAX_P_STATUS.length) {
      throw new Error(`${id} output.status must hold ${GPU_MAX_P_STATUS.length} words`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.objective, output.status],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.extensiveValues,
        props.threshold
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, columnCount, startCount, improvementIterations} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.values,
      props.extensiveValues,
      props.threshold,
      output.labels,
      output.objective,
      output.status
    ]);
    // Region record: row count, extensive sum, then one sum per column.
    const regionStride = columnCount + 2;
    const maximumBindingBytes =
      graph.device.limits.maxStorageBufferBindingSize ?? DEFAULT_MAXIMUM_BINDING_BYTES;
    if (startCount * rowCount * regionStride * 4 > maximumBindingBytes) {
      throw new Error(`${id} scratch exceeds one storage binding; reduce startCount or rowCount`);
    }
    const standardize = props.standardize !== false;
    const table = standardize
      ? createTransientView(graph, `${id}-table`, 'float32', rowCount * columnCount)
      : props.values;
    const adjacency = createSymmetricAdjacency(graph, {
      id,
      operation: 'GPUMaxPRegions',
      weights: props.weights,
      rowCount
    });
    // Row extensive values followed by the threshold.
    const packedExtensive = createTransientView(graph, `${id}-extensive`, 'float32', rowCount + 1);
    // Per start: `rowCount` labels followed by `rowCount` random seed-order keys.
    const startLabels = createTransientView(
      graph,
      `${id}-start-labels`,
      'uint32',
      startCount * 2 * rowCount
    );
    const startRegions = createTransientView(
      graph,
      `${id}-start-regions`,
      'float32',
      startCount * rowCount * regionStride
    );
    // Per start: region count, infeasible count, within-region sum of squares (float bits), spare.
    const startState = createTransientView(
      graph,
      `${id}-start-state`,
      'uint32',
      startCount * START_STATE_STRIDE
    );
    // Words: the invalid-input copy and the selected start.
    const state = createTransientView(graph, `${id}-state`, 'uint32', 2);
    const improving = improvementIterations > 0;
    const bestLabels = improving
      ? createTransientView(graph, `${id}-best-labels`, 'uint32', rowCount)
      : output.labels;
    const improvementObjective = improving
      ? createTransientView(graph, `${id}-improvement-objective`, 'float32', 1)
      : null;
    const improvementStatus = improving
      ? createTransientView(graph, `${id}-improvement-status`, 'uint32', 4)
      : null;
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const START_COUNT: u32 = ${startCount}u;
const SEED: u32 = ${props.seed}u;
const NONE: u32 = 0xffffffffu;
const LARGE_DISTANCE: f32 = 3.402823466e+38;
const LABEL_STRIDE: u32 = ${2 * rowCount}u;
const REGION_BLOCK: u32 = ${rowCount * regionStride}u;
const START_STATE_STRIDE: u32 = ${START_STATE_STRIDE}u;
${IS_FINITE_FLOAT_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...adjacency.nodes,
      ...(standardize
        ? [
            createStandardizeColumnsNode<Parameters>(graph, {
              id: `${id}-standardize`,
              operation: 'GPUMaxPRegions',
              values: props.values,
              scaled: table,
              rowCount,
              columnCount
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: 'GPUMaxPRegions',
        variant: 'initialize',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  objective[objectiveOffset] = 0.0;
  for (var word = 0u; word < ${GPU_MAX_P_STATUS.length}u; word++) { status[statusOffset + word] = 0u; }`
      }),
      // Row extensive values and the threshold in one buffer (one binding for the construction).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pack-extensive`,
        operation: 'GPUMaxPRegions',
        variant: 'pack-extensive',
        invocationCount: rowCount + 1,
        bindings: [
          {name: 'extensive', view: props.extensiveValues, type: 'f32', access: 'read'},
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'packedValues', view: packedExtensive, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `var value = threshold[thresholdOffset];
  if (index < ROW_COUNT) { value = extensive[extensiveOffset + index]; }
  if (!isFiniteFloat(value) || value < 0.0) { atomicAdd(&status[statusOffset + 3u], 1u); }
  packedValues[packedValuesOffset + index] = value;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate-rows`,
        operation: 'GPUMaxPRegions',
        variant: 'parallel-validation',
        invocationCount: rowCount,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `var invalid = 0u;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    invalid += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMN_COUNT + column]));
  }
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalid += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROW_COUNT);
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset + 3u], invalid); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-capture-invalid`,
        operation: 'GPUMaxPRegions',
        variant: 'capture-invalid',
        invocationCount: 1,
        bindings: [
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        body: 'state[stateOffset] = status[statusOffset + 3u];'
      }),
      // Unassigned labels and the per-start random seed-order keys: Philox 4x32-10 of the row
      // under the key (seed, start), so every start draws its own independent order.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-starts`,
        operation: 'GPUMaxPRegions',
        variant: 'initialize-starts',
        invocationCount: startCount * rowCount,
        bindings: [{name: 'labels', view: startLabels, type: 'u32', access: 'read_write'}],
        declarations: `${declarations}
${PHILOX_WGSL}`,
        body: `let start = index / ROW_COUNT;
  let row = index - start * ROW_COUNT;
  let base = labelsOffset + start * LABEL_STRIDE;
  labels[base + row] = NONE;
  labels[base + ROW_COUNT + row] =
    philox4x32(vec4<u32>(row, 0u, 0u, 0u), vec2<u32>(SEED, start)).x;`
      }),
      // Each construction is inherently sequential: every growth step depends on the previous
      // one. One persistent workgroup per start runs the whole seed / grow / merge loop with
      // parallel strided candidate scans and shared-memory argmin reductions, instead of
      // unrolling O(rowCount^2) graph nodes of mostly empty full-size dispatches.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-construct`,
        operation: 'GPUMaxPRegions',
        variant: 'multi-start-persistent-workgroup-construction',
        invocationCount: startCount * MAX_P_WORKGROUP_SIZE,
        workgroupSize: MAX_P_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: table, type: 'f32', access: 'read'},
          {name: 'extensive', view: packedExtensive, type: 'f32', access: 'read'},
          {name: 'labels', view: startLabels, type: 'u32', access: 'read_write'},
          {name: 'regions', view: startRegions, type: 'f32', access: 'read_write'},
          {name: 'startState', view: startState, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read'}
        ],
        declarations: `${declarations}
${getWorkgroupArgminWGSL(MAX_P_WORKGROUP_SIZE)}
// Running count, extensive sum and column sums of the region under construction.
var<workgroup> currentStats: array<f32, REGION_STRIDE>;
var<workgroup> broadcastWord: u32;
var<workgroup> sumScratch: array<f32, ${MAX_P_WORKGROUP_SIZE}>;

// Sum of one value per lane over the workgroup, available to every lane.
fn reduceSum(lane: u32, value: f32) -> f32 {
  sumScratch[lane] = value;
  workgroupBarrier();
  for (var stride = ${MAX_P_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { sumScratch[lane] = sumScratch[lane] + sumScratch[lane + stride]; }
    workgroupBarrier();
  }
  let total = sumScratch[0];
  workgroupBarrier();
  return total;
}`,
        body: `let lane = localInvocationIndex;
  // One workgroup per start, so \`start\` is uniform across the workgroup.
  let start = index / ${MAX_P_WORKGROUP_SIZE}u;
  let labelBase = labelsOffset + start * LABEL_STRIDE;
  let keyBase = labelBase + ROW_COUNT;
  let regionRoot = regionsOffset + start * REGION_BLOCK;
  if (lane == 0u) { broadcastWord = state[stateOffset]; }
  let invalid = workgroupUniformLoad(&broadcastWord);
  var regionCount = 0u;
  var infeasibleCount = 0u;
  if (invalid == 0u) {
    let thresholdValue = extensive[extensiveOffset + ROW_COUNT];
    for (var construction = 0u; construction < ROW_COUNT; construction++) {
      var bestScore = LARGE_DISTANCE;
      var bestKey = NONE;
      var bestValue = NONE;
      for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
        let key = labels[keyBase + row];
        if (labels[labelBase + row] == NONE && isArgminBetter(0.0, key, row, bestScore, bestKey, bestValue)) {
          bestScore = 0.0; bestKey = key; bestValue = row;
        }
      }
      let seed = reduceArgmin(lane, bestScore, bestKey, bestValue);
      if (seed == NONE) { break; }
      let region = regionCount;
      if (lane == 0u) {
        labels[labelBase + seed] = region;
        currentStats[0] = 1.0;
        currentStats[1] = extensive[extensiveOffset + seed];
        for (var column = 0u; column < COLUMN_COUNT; column++) {
          currentStats[2u + column] = values[valuesOffset + seed * COLUMN_COUNT + column];
        }
        broadcastWord = select(1u, 2u, currentStats[1] >= thresholdValue);
      }
      var phase = workgroupUniformLoad(&broadcastWord);
      storageBarrier();
      // Grow while under threshold: the unassigned row adjacent to the region nearest its mean.
      loop {
        if (phase != 1u) { break; }
        bestScore = LARGE_DISTANCE; bestKey = NONE; bestValue = NONE;
        for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
          if (labels[labelBase + row] != NONE) { continue; }
          var adjacent = false;
          for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
            if (labels[labelBase + neighbors[neighborsOffset + slot]] == region) { adjacent = true; break; }
          }
          if (!adjacent) { continue; }
          var distance = 0.0;
          for (var column = 0u; column < COLUMN_COUNT; column++) {
            let difference = values[valuesOffset + row * COLUMN_COUNT + column] - currentStats[2u + column] / currentStats[0];
            distance += difference * difference;
          }
          let key = labels[keyBase + row];
          if (isArgminBetter(distance, key, row, bestScore, bestKey, bestValue)) {
            bestScore = distance; bestKey = key; bestValue = row;
          }
        }
        let grown = reduceArgmin(lane, bestScore, bestKey, bestValue);
        if (grown == NONE) { break; }
        if (lane == 0u) {
          labels[labelBase + grown] = region;
          currentStats[0] += 1.0;
          currentStats[1] += extensive[extensiveOffset + grown];
          for (var column = 0u; column < COLUMN_COUNT; column++) {
            currentStats[2u + column] += values[valuesOffset + grown * COLUMN_COUNT + column];
          }
          broadcastWord = select(1u, 2u, currentStats[1] >= thresholdValue);
        }
        phase = workgroupUniformLoad(&broadcastWord);
        storageBarrier();
      }
      if (lane == 0u) {
        let base = regionRoot + region * REGION_STRIDE;
        for (var slot = 0u; slot < REGION_STRIDE; slot++) { regions[base + slot] = currentStats[slot]; }
      }
      // Nearest adjacent completed region by mean distance, ties to the lowest region ID.
      bestScore = LARGE_DISTANCE; bestKey = NONE; bestValue = NONE;
      for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
        if (labels[labelBase + row] != region) { continue; }
        for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
          let candidate = labels[labelBase + neighbors[neighborsOffset + slot]];
          if (candidate == NONE || candidate == region) { continue; }
          let candidateBase = regionRoot + candidate * REGION_STRIDE;
          var distance = 0.0;
          for (var column = 0u; column < COLUMN_COUNT; column++) {
            let difference = currentStats[2u + column] / currentStats[0] -
              regions[candidateBase + 2u + column] / regions[candidateBase];
            distance += difference * difference;
          }
          if (isArgminBetter(distance, candidate, candidate, bestScore, bestKey, bestValue)) {
            bestScore = distance; bestKey = candidate; bestValue = candidate;
          }
        }
      }
      let mergeRegion = reduceArgmin(lane, bestScore, bestKey, bestValue);
      if (phase != 2u && mergeRegion != NONE) {
        // Enclave: fold the under-threshold remainder into its nearest neighbor region and
        // reuse this region ID for the next seed.
        if (lane == 0u) {
          let mergeBase = regionRoot + mergeRegion * REGION_STRIDE;
          for (var slot = 0u; slot < REGION_STRIDE; slot++) { regions[mergeBase + slot] += currentStats[slot]; }
        }
        for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
          if (labels[labelBase + row] == region) { labels[labelBase + row] = mergeRegion; }
        }
      } else {
        regionCount += 1u;
        if (phase != 2u) { infeasibleCount += 1u; }
      }
      storageBarrier();
    }
  }
  storageBarrier();
  workgroupBarrier();
  // Within-region sum of squares of this start, for the start selection.
  var squares = 0.0;
  if (invalid == 0u) {
    for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
      let label = labels[labelBase + row];
      if (label == NONE) { continue; }
      let base = regionRoot + label * REGION_STRIDE;
      for (var column = 0u; column < COLUMN_COUNT; column++) {
        let difference = values[valuesOffset + row * COLUMN_COUNT + column] - regions[base + 2u + column] / regions[base];
        squares += difference * difference;
      }
    }
  }
  let withinSquares = reduceSum(lane, squares);
  if (lane == 0u) {
    let base = startStateOffset + start * START_STATE_STRIDE;
    startState[base] = regionCount;
    startState[base + 1u] = infeasibleCount;
    startState[base + 2u] = bitcast<u32>(withinSquares);
    startState[base + 3u] = 0u;
  }`
      }),
      // Fewest leftover regions, then most regions, then smallest within-region sum of squares,
      // then lowest start (strict comparisons keep the earlier start on a full tie).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select-start`,
        operation: 'GPUMaxPRegions',
        variant: 'select-start',
        invocationCount: 1,
        bindings: [
          {name: 'startState', view: startState, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var best = 0u;
  for (var start = 1u; start < START_COUNT; start++) {
    let base = startStateOffset + start * START_STATE_STRIDE;
    let bestBase = startStateOffset + best * START_STATE_STRIDE;
    let infeasible = startState[base + 1u];
    let bestInfeasible = startState[bestBase + 1u];
    let regionCount = startState[base];
    let bestRegionCount = startState[bestBase];
    let within = bitcast<f32>(startState[base + 2u]);
    let bestWithin = bitcast<f32>(startState[bestBase + 2u]);
    if (infeasible < bestInfeasible ||
        (infeasible == bestInfeasible && (regionCount > bestRegionCount ||
        (regionCount == bestRegionCount && within < bestWithin)))) {
      best = start;
    }
  }
  state[stateOffset + 1u] = best;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-copy-best`,
        operation: 'GPUMaxPRegions',
        variant: 'copy-best',
        invocationCount: rowCount,
        bindings: [
          {name: 'startLabels', view: startLabels, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'labelsOut', view: bestLabels, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: 'labelsOut[labelsOutOffset + index] = startLabels[startLabelsOffset + state[stateOffset + 1u] * LABEL_STRIDE + index];'
      })
    ];

    if (improving) {
      // The selected construction is refined by greedy AZP moves that keep every region at or
      // above the extensive threshold. The adjacency is already symmetric (`info.symmetric`).
      nodes.push(
        ...new GPUAZPRegions({
          id: `${id}-improvement`,
          weights: adjacency.weights,
          values: table,
          columnCount,
          standardize: false,
          extensiveValues: props.extensiveValues,
          extensiveThreshold: props.threshold,
          initialLabels: bestLabels,
          regionCapacity: rowCount,
          maximumIterations: improvementIterations,
          seed: props.seed,
          labels: output.labels,
          optimization: {
            objective: improvementObjective!,
            status: improvementStatus!,
            seed: props.seed
          }
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUMaxPRegions',
        variant: 'publish',
        invocationCount: 1,
        bindings: [
          {name: 'startState', view: startState, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          ...(improving
            ? [
                {
                  name: 'improvementObjective',
                  view: improvementObjective!,
                  type: 'f32' as const,
                  access: 'read' as const
                },
                {
                  name: 'improvementStatus',
                  view: improvementStatus!,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations,
        body: `let best = state[stateOffset + 1u];
  let base = startStateOffset + best * START_STATE_STRIDE;
  var invalid = state[stateOffset];
  ${
    improving
      ? `// Only an improvement-phase rejection (not already a max-p invalid input) adds to the count.
  invalid = max(invalid, improvementStatus[improvementStatusOffset + 3u]);
  let moveCount = improvementStatus[improvementStatusOffset];
  let withinSquares = improvementObjective[improvementObjectiveOffset];`
      : `let moveCount = 0u;
  let withinSquares = bitcast<f32>(startState[base + 2u]);`
  }
  let infeasible = startState[base + 1u];
  objective[objectiveOffset] = select(withinSquares, 0.0, invalid > 0u);
  status[statusOffset] = select(startState[base], 0u, invalid > 0u);
  status[statusOffset + 1u] = select(0u, 1u, infeasible == 0u && invalid == 0u);
  status[statusOffset + 2u] = infeasible;
  status[statusOffset + 3u] = invalid;
  status[statusOffset + 4u] = best;
  status[statusOffset + 5u] = moveCount;`
      })
    );
    return nodes;
  }
}
