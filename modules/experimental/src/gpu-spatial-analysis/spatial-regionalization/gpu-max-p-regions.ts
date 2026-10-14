// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../utils/philox-random';
import {GPUAZPRegions, GPU_AZP_MAXIMUM_ITERATIONS} from './gpu-azp-regions';
import {
  GPURegionPartitionEvaluation,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT
} from './gpu-region-partition-evaluation';
import {
  createStandardizeColumnsNode,
  createSymmetricAdjacency,
  getRegionalizationSeedParameterValues,
  REGIONALIZATION_SEED_PARAMETER_LENGTH,
  validateRegionalizationParameters,
  validateRegionalizationSlotCount
} from './regionalization-inputs';

export const GPU_MAX_P_MAXIMUM_COLUMNS = 8;
/** Number of independent constructions run when `startCount` is not given. */
export const GPU_MAX_P_DEFAULT_START_COUNT = 16;
/** Largest `startCount`: each start owns scratch rows and a region table of the input size. */
export const GPU_MAX_P_MAXIMUM_START_COUNT = 64;
/**
 * Number of construction rounds when `maximumGrowthSteps` is not given. A region claims one row per
 * round, so this is also the largest region (in rows) that the construction can complete.
 */
export const GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS = 256;
/** Largest `maximumGrowthSteps`: each round is a fixed set of about nine command nodes. */
export const GPU_MAX_P_MAXIMUM_ROUNDS = 4096;
/**
 * Largest `weights.neighbors` slot count of {@link GPUMaxPRegions} on `device`: the improvement
 * phase hands the symmetric adjacency (two slots per input slot) to an AZP refinement that
 * symmetrizes it again, 16 bytes per input slot in one storage binding, so half of
 * {@link getGPURegionalizationMaximumSlotCount}: 8388608 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUMaxPMaximumSlotCount(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}
const FINALIZE_WORKGROUP_SIZE = 256;
const START_STATE_STRIDE = 4;
/** Words of one row's claim record: best key, best tie, claim key, claim tie. */
const CLAIM_STRIDE = 4;
/** Words of the region table before the column means: row count, extensive sum, squares. */
const REGION_HEADER = 3;
/** Fallback for the largest storage binding when the device does not report one (128 MiB). */
const DEFAULT_MAXIMUM_BINDING_BYTES = 128 * 1024 * 1024;
/** Odd multiplier of the bijective tie order `tie(x) = (x * multiplier) ^ salt`. */
const TIE_MULTIPLIER = 0x9e3779b1;
/** First wave plants seeds at this fraction of `1 / expectedRegionSize` of the rows. */
const WAVE_START_FRACTION = '0.4';
/** Each later wave multiplies the seed fraction by this factor. */
const WAVE_GROWTH = '1.3';
/** Largest expected region size in rows (keeps the float conversions exact). */
const MAXIMUM_EXPECTED_REGION_SIZE = 1048576;

/** Global words of the control buffer; the per-start records follow at `startRecords`. */
const STATE = {
  invalid: 0,
  selectedStart: 1,
  round: 2,
  expectedSize: 3,
  moved: 4,
  enclaveRound: 5,
  /** First of `startCount` per-start records of {@link START_RECORD_WORDS} words. */
  startRecords: 16
} as const;
/**
 * Words of a per-start record: growing region count, unassigned row count (both atomic), next
 * wave round, wave seed fraction (float bits), cut key, wave flag, tie salt, spare. Every start
 * decides its waves from its own record, so a start's construction does not depend on the others.
 */
const START_RECORD_WORDS = 8;
const START_WORD = {
  growing: 0,
  unassigned: 1,
  nextWave: 2,
  waveFraction: 3,
  cutKey: 4,
  waveFlag: 5,
  salt: 6
} as const;

const PHASE = {growing: 1, complete: 2, failed: 3, marked: 4} as const;

/** Modular inverse of the (odd) tie multiplier modulo 2^32, by Newton iteration. */
function getTieInverse(): number {
  const modulus = 1n << 32n;
  let inverse = 1n;
  for (let step = 0; step < 6; step++) {
    inverse = (inverse * (2n - ((BigInt(TIE_MULTIPLIER) * inverse) % modulus) + modulus)) % modulus;
  }
  return Number(inverse);
}

/**
 * Compile-time round counts of the construction: growth rounds and enclave rounds. Both are
 * capped by the input size, because a round that assigns nothing ends the phase.
 *
 * @internal
 */
export function getGPUMaxPRoundCounts(
  rowCount: number,
  maximumGrowthSteps: number
): {growthRounds: number; enclaveRounds: number} {
  return {
    growthRounds: Math.min(maximumGrowthSteps, 2 * rowCount + 2),
    enclaveRounds: Math.min(maximumGrowthSteps, rowCount + 1)
  };
}

/**
 * Number of `uint32` words of {@link GPUMaxPRegionsProps.parameters}: `[seed]`. The layout equals
 * the AZP parameters, so the local-improvement phase reads the same view.
 */
export const GPU_MAX_P_REGIONS_PARAMETER_LENGTH = REGIONALIZATION_SEED_PARAMETER_LENGTH;

/**
 * Per-frame settings of {@link GPUMaxPRegions}, packed by
 * {@link getGPUMaxPRegionsParameterValues}.
 */
export type GPUMaxPRegionsSettings = {
  /**
   * Reproducible seed of the random seed-row order of every start (and of the improvement
   * phase's tie order), an integer in `[0, 2^32 - 1]`. Defaults to 0.
   */
  seed?: number;
};

/**
 * Packs {@link GPUMaxPRegionsSettings} into the {@link GPU_MAX_P_REGIONS_PARAMETER_LENGTH}
 * `uint32` words of {@link GPUMaxPRegionsProps.parameters}.
 */
export function getGPUMaxPRegionsParameterValues(
  settings: GPUMaxPRegionsSettings = {}
): Uint32Array {
  return getRegionalizationSeedParameterValues('GPUMaxPRegions', settings.seed);
}

/** Words of {@link GPUMaxPRegionsOutput.status}. */
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
  /** Moves made by the local-improvement phase (0 when `maximumIterations` is 0). */
  improvementMoveCount: 5,
  length: 6
} as const;

/** Caller-owned outputs of {@link GPUMaxPRegions}. */
export type GPUMaxPRegionsOutput = {
  /** Region label per row, dense in `[0, regionCount)`. */
  labels: GraphDataView<'uint32'>;
  /**
   * Total within-region sum of squared deviations of the returned partition (standardized
   * units unless `standardize` is false), the same quantity as `withinSumOfSquares` of
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
   * contributor; weights that declare `information.symmetric === true` are used as they are. At most
   * {@link getGPUMaxPMaximumSlotCount} `neighbors` slots (8388608 on a default device).
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
   * Compile-time number of construction rounds (default {@link GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS},
   * at most {@link GPU_MAX_P_MAXIMUM_ROUNDS}). A growing region claims one row per round, so a
   * region needing more rows than this cannot complete and is reported under-threshold;
   * rounds after the construction has settled launch nothing. Raise it when the threshold needs
   * regions of more than about `maximumGrowthSteps / 4` rows (seed waves are spaced half an expected
   * region apart), at the cost of nine more command nodes per round.
   */
  maximumGrowthSteps?: number;
  /**
   * Compile-time maximum moves of the optional local-improvement phase (0, the default, skips it;
   * at most 256). It runs `GPUAZPRegions` on the selected construction with the extensive
   * threshold as a move constraint, so the region count never changes, and it needs at least
   * 2 rows.
   */
  maximumIterations?: number;
  /**
   * Per-frame `uint32` words ({@link GPU_MAX_P_REGIONS_PARAMETER_LENGTH}): the seed, written with
   * {@link getGPUMaxPRegionsParameterValues}.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUMaxPRegionsOutput;
};

/**
 * Parallel multi-seed Max-P regionalization.
 *
 * Each of `startCount` independent constructions plants many region seeds at once and grows all
 * regions together, in rounds, until every region reaches the extensive threshold:
 *
 * 1. **Seed waves.** A wave plants a seed at every unassigned row that is a local minimum of its
 *    Philox 4x32-10 key of `(row, start; seed)` among the unassigned rows next to it, is not next
 *    to a growing region, and whose key is below the wave's cut. The first wave cuts to about
 *    `0.4 / s` of the rows with `s = ceil(threshold / mean(extensiveValues))`, the expected region
 *    size, and every later wave cuts 1.3 times as many; a wave starts when no region is growing or
 *    `ceil(s / 2)` rounds after the previous one. Spacing the waves this way gives sequential-like
 *    packing (seeds are not crowded into regions that cannot reach the threshold).
 * 2. **Growth.** Each growing region claims one frontier row per round. Every unassigned row next
 *    to a growing region offers itself to it with a key made of its number of unassigned
 *    neighbors (fewest first, which keeps regions compact and leaves few unreachable pockets) and
 *    its squared distance to the region mean (nearest first); the region keeps its smallest key
 *    (ties by a seeded bijective row order) and a row offered by several regions joins the one
 *    with the smallest key. Region sums are updated by the one region that owns them, so no
 *    float atomics are involved. A region at the threshold stops growing; a region with no
 *    frontier under the threshold is marked failed.
 * 3. **Enclaves.** Rows of failed regions and rows that no region reached are given, round by
 *    round, to the most similar adjacent completed region. A failed region with no completed
 *    neighbor stays as an under-threshold region (reported in the status).
 *
 * Every dispatch is `O(startCount * (rowCount + neighborCount))` and no thread runs a loop over
 * the whole input, so the construction has no row limit besides the storage-binding size of its
 * `startCount * rowCount * (columnCount + 3)` float region table. All starts and all regions
 * share each dispatch. The rounds are unrolled to `maximumGrowthSteps` and gated on the GPU, so the
 * work ends when the construction has settled.
 *
 * **Start selection.** The returned start has the fewest under-threshold regions (a partition
 * whose regions all reach the threshold beats one that does not), then the most regions (the
 * Max-P objective), then the smallest within-region sum of squares of the grown regions (the
 * sum of the regions' incremental squared deviations, before enclave rows are added), then the
 * lowest start index. The result is deterministic for a given seed and `startCount`.
 *
 * **Local improvement.** With `maximumIterations > 0` the selected construction is refined by
 * greedy steepest-descent AZP moves under the extensive-threshold constraint (a region never
 * drops below the threshold), which lowers the within-region sum of squares at the same region
 * count. spopt's `MaxPHeuristic` instead anneals; this is the greedy variant, off by default
 * because it uses the AZP candidate search (an `O(rowCount)` connectivity check per improving
 * candidate).
 *
 * `output.objective` is the within-region sum of squared deviations of the returned labels (in
 * standardized units unless `standardize` is false).
 *
 * Default change: columns are standardized (`standardize`, default true); the previous raw-value
 * behavior is `standardize: false`. The construction itself changed from one sequential growth
 * per start to the parallel one above, so the regions differ from earlier releases for the same
 * seed; the region count is comparable to spopt's `MaxPHeuristic` (see the oracle spec).
 *
 * Scale: scratch memory is `startCount * rowCount * (columnCount + 12) * 4` bytes in at most
 * four storage bindings and must fit one binding each. There is no row cap: a region grows by one
 * row per round, so very large regions need a larger `maximumGrowthSteps`.
 */
export class GPUMaxPRegions implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMaxPRegionsProps;
  readonly rowCount: number;
  readonly columnCount: number;
  /** Number of independent constructions. */
  readonly startCount: number;
  /** Number of unrolled construction rounds. */
  readonly maximumGrowthSteps: number;
  /** Maximum moves of the local-improvement phase. */
  readonly maximumIterations: number;

  constructor(props: GPUMaxPRegionsProps) {
    this.id = props.id ?? 'max-p-regions';
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(this.id, props.weights);
    this.columnCount = props.columnCount ?? 1;
    this.startCount = props.startCount ?? GPU_MAX_P_DEFAULT_START_COUNT;
    this.maximumGrowthSteps = props.maximumGrowthSteps ?? GPU_MAX_P_DEFAULT_MAXIMUM_ROUNDS;
    this.maximumIterations = props.maximumIterations ?? 0;
    const {id, rowCount, columnCount, startCount, maximumIterations} = this;
    const {output} = props;
    validateCapacity(id, 'columnCount', columnCount, 1, GPU_MAX_P_MAXIMUM_COLUMNS);
    validateCapacity(id, 'startCount', startCount, 1, GPU_MAX_P_MAXIMUM_START_COUNT);
    validateCapacity(
      id,
      'maximumGrowthSteps',
      this.maximumGrowthSteps,
      1,
      GPU_MAX_P_MAXIMUM_ROUNDS
    );
    validateCapacity(id, 'maximumIterations', maximumIterations, 0, GPU_AZP_MAXIMUM_ITERATIONS);
    if (maximumIterations > 0 && rowCount < 2) {
      throw new Error(`${id} maximumIterations needs at least 2 rows`);
    }
    validateRegionalizationParameters(id, props.parameters, GPU_MAX_P_REGIONS_PARAMETER_LENGTH);
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
    validatePackedView(output.labels, ['uint32'], `${id} output.labels`);
    validatePackedView(output.status, ['uint32'], `${id} output.status`);
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
        props.threshold,
        props.parameters
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, columnCount, startCount, maximumIterations} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
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
    validateRegionalizationSlotCount(id, props.weights, getGPUMaxPMaximumSlotCount(graph.device));
    // Region record: row count, extensive sum, squared deviations, then one sum per column.
    const regionStride = columnCount + REGION_HEADER;
    const maximumBindingBytes =
      graph.device.limits.maxStorageBufferBindingSize ?? DEFAULT_MAXIMUM_BINDING_BYTES;
    // The largest scratch binding is the region table or the four-word claim records.
    const scratchWordsPerRow = Math.max(regionStride, CLAIM_STRIDE, 2);
    if (startCount * rowCount * scratchWordsPerRow * 4 > maximumBindingBytes) {
      throw new Error(`${id} scratch exceeds one storage binding; reduce startCount or rowCount`);
    }
    const {growthRounds, enclaveRounds} = getGPUMaxPRoundCounts(rowCount, this.maximumGrowthSteps);
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
    const scratchRows = startCount * rowCount;
    // Per start and row: the region ID (seed row) or NONE, then the random seed-order key.
    const rowState = createTransientView(graph, `${id}-row-state`, 'uint32', 2 * scratchRows);
    // Per start and row: the region phase (rows that are not region IDs hold 0).
    const regionPhase = createTransientView(graph, `${id}-region-phase`, 'uint32', scratchRows);
    const regionData = createTransientView(
      graph,
      `${id}-region-data`,
      'float32',
      scratchRows * regionStride
    );
    // Store means across dispatches so Metal cannot fuse value - sum / count into an FMA.
    // Keep sums separately for exact integer accumulation and incremental mean updates.
    const regionSums = createTransientView(
      graph,
      `${id}-region-sums`,
      'float32',
      scratchRows * columnCount
    );
    const claims = createTransientView(graph, `${id}-claims`, 'uint32', scratchRows * CLAIM_STRIDE);
    // Per start: region count, under-threshold count, grown within-region sum of squares (float
    // bits), spare.
    const startState = createTransientView(
      graph,
      `${id}-start-state`,
      'uint32',
      startCount * START_STATE_STRIDE
    );
    // Control words (see STATE): invalid-input copy, selected start, round cursor, counters.
    const state = createTransientView(
      graph,
      `${id}-state`,
      'uint32',
      STATE.startRecords + startCount * START_RECORD_WORDS
    );
    const flags = createTransientView(graph, `${id}-region-flags`, 'uint32', rowCount);
    const ranks = createTransientView(graph, `${id}-region-ranks`, 'uint32', rowCount);
    const improving = maximumIterations > 0;
    const bestLabels = improving
      ? createTransientView(graph, `${id}-best-labels`, 'uint32', rowCount)
      : output.labels;
    const improvementObjective = improving
      ? createTransientView(graph, `${id}-improvement-objective`, 'float32', 1)
      : null;
    const improvementStatus = improving
      ? createTransientView(graph, `${id}-improvement-status`, 'uint32', 4)
      : null;
    const evaluationSummary = improving
      ? null
      : createTransientView(
          graph,
          `${id}-evaluation-summary`,
          'float32',
          GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
        );
    // Rule 13: indirect dispatch triples of the gated rounds. Slot 0 growth, slot 1 seed wave,
    // slot 2 enclave; every slot has the shape of one invocation per start and row.
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-round-dispatch`,
      operation: 'GPUMaxPRegions',
      shapes: [[scratchRows], [scratchRows], [scratchRows]]
    });
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const ROWS: u32 = ${rowCount}u;
const COLUMN_COUNT: u32 = ${columnCount}u;
const COLUMNS: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const REGION_HEADER: u32 = ${REGION_HEADER}u;
const START_COUNT: u32 = ${startCount}u;
const STARTS: u32 = ${startCount}u;
const NONE: u32 = 0xffffffffu;
const START_STATE_STRIDE: u32 = ${START_STATE_STRIDE}u;
const CLAIM_STRIDE: u32 = ${CLAIM_STRIDE}u;
const PHASE_GROWING: u32 = ${PHASE.growing}u;
const PHASE_COMPLETE: u32 = ${PHASE.complete}u;
const PHASE_FAILED: u32 = ${PHASE.failed}u;
const PHASE_MARKED: u32 = ${PHASE.marked}u;
const TIE_MULTIPLIER: u32 = ${TIE_MULTIPLIER}u;
const TIE_INVERSE: u32 = ${getTieInverse()}u;
const STATE_INVALID: u32 = ${STATE.invalid}u;
const STATE_SELECTED: u32 = ${STATE.selectedStart}u;
const STATE_ROUND: u32 = ${STATE.round}u;
const STATE_EXPECTED_SIZE: u32 = ${STATE.expectedSize}u;
const STATE_MOVED: u32 = ${STATE.moved}u;
const STATE_ENCLAVE_ROUND: u32 = ${STATE.enclaveRound}u;
const STATE_STARTS: u32 = ${STATE.startRecords}u;
const START_RECORD: u32 = ${START_RECORD_WORDS}u;
const WORD_GROWING: u32 = ${START_WORD.growing}u;
const WORD_UNASSIGNED: u32 = ${START_WORD.unassigned}u;
const WORD_NEXT_WAVE: u32 = ${START_WORD.nextWave}u;
const WORD_WAVE_FRACTION: u32 = ${START_WORD.waveFraction}u;
const WORD_CUT_KEY: u32 = ${START_WORD.cutKey}u;
const WORD_WAVE_FLAG: u32 = ${START_WORD.waveFlag}u;
const WORD_SALT: u32 = ${START_WORD.salt}u;
${IS_FINITE_FLOAT_WGSL}
// Bijective tie order of rows and regions: unique per value, reshuffled by the start's salt.
fn getTie(value: u32, salt: u32) -> u32 { return (value * TIE_MULTIPLIER) ^ salt; }
fn getTieValue(tie: u32, salt: u32) -> u32 { return (tie ^ salt) * TIE_INVERSE; }`;
    // Squared distance of a row to the mean of a region of the same start; needs the bindings
    // \`regionData\` and \`values\`.
    const distanceWGSL = `
fn getDistance(start: u32, row: u32, region: u32) -> f32 {
  let regionBase = regionDataOffset + (start * ROWS + region) * REGION_STRIDE;
  var distance = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let mean = regionData[regionBase + REGION_HEADER + column];
    let difference = values[valuesOffset + row * COLUMNS + column] - mean;
    distance += difference * difference;
  }
  return distance;
}`;
    const startAndRow = `let start = index / ROWS;
  let row = index - start * ROWS;`;
    const gated = (slot: number) => gate.getGate(slot);
    const writeGate = (armGrowth: string, armSeed: string, armEnclave: string) => `${gate.armWGSL}
  if (!(${armGrowth})) { dispatch[dispatchOffset] = 0u; }
  if (!(${armSeed})) { dispatch[dispatchOffset + 3u] = 0u; }
  if (!(${armEnclave})) { dispatch[dispatchOffset + 6u] = 0u; }`;
    const bindingsOf = (...items: WGSLKernelBinding[]) => items;
    const offsetsBinding = (): WGSLKernelBinding => ({
      name: 'offsets',
      view: adjacency.weights.offsets,
      type: 'u32',
      access: 'read'
    });
    const neighborsBinding = (): WGSLKernelBinding => ({
      name: 'neighbors',
      view: adjacency.weights.neighbors,
      type: 'u32',
      access: 'read'
    });
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
      // Control words of the construction: the expected region size from the mean extensive
      // value (a fixed lane-strided float sum, deterministic), the first wave's seed fraction,
      // the counters, and one tie salt per start drawn from Philox.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-control`,
        operation: 'GPUMaxPRegions',
        variant: 'initialize-control',
        invocationCount: FINALIZE_WORKGROUP_SIZE,
        workgroupSize: FINALIZE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'extensive', view: packedExtensive, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
${PHILOX_WGSL}
const PURPOSE: u32 = ${PHILOX_PURPOSES.maxPRegionsSeedOrder.id}u;
var<workgroup> partials: array<f32, ${FINALIZE_WORKGROUP_SIZE}>;`,
        body: `let lane = localInvocationIndex;
  var sum = 0.0;
  for (var row = lane; row < ROWS; row += ${FINALIZE_WORKGROUP_SIZE}u) {
    sum += extensive[extensiveOffset + row];
  }
  partials[lane] = sum;
  workgroupBarrier();
  for (var stride = ${FINALIZE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  // Every lane holds the total, so every lane derives the same expected region size.
  let mean = partials[0] / f32(ROWS);
  let threshold = extensive[extensiveOffset + ROWS];
  var expectedSize = 1u;
  if (mean > 0.0) {
    expectedSize = u32(clamp(ceil(threshold / mean), 1.0, ${MAXIMUM_EXPECTED_REGION_SIZE}.0));
  }
  if (lane < STARTS) {
    let record = stateOffset + STATE_STARTS + lane * START_RECORD;
    state[record + WORD_GROWING] = 0u;
    state[record + WORD_UNASSIGNED] = ROWS;
    state[record + WORD_NEXT_WAVE] = 0u;
    state[record + WORD_WAVE_FRACTION] = bitcast<u32>(${WAVE_START_FRACTION} / f32(expectedSize));
    state[record + WORD_CUT_KEY] = 0u;
    state[record + WORD_WAVE_FLAG] = 0u;
    state[record + WORD_SALT] =
      philox4x32(vec4<u32>(NONE, lane, 0u, 0u), vec2<u32>(parameters[parametersOffset], PURPOSE)).y;
  }
  if (lane == 0u) {
    state[stateOffset + STATE_ROUND] = 0u;
    state[stateOffset + STATE_EXPECTED_SIZE] = expectedSize;
    state[stateOffset + STATE_MOVED] = 0u;
    state[stateOffset + STATE_ENCLAVE_ROUND] = 0u;
  }`
      }),
      // Unassigned labels, the per-start random seed-order keys and empty phases: Philox
      // 4x32-10 of the row and start under the key (seed, purpose), so every start draws its
      // own independent order.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-starts`,
        operation: 'GPUMaxPRegions',
        variant: 'initialize-starts',
        invocationCount: scratchRows,
        bindings: [
          {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
          {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read_write'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'}
        ],
        declarations: `${declarations}
${PHILOX_WGSL}
const PURPOSE: u32 = ${PHILOX_PURPOSES.maxPRegionsSeedOrder.id}u;`,
        body: `${startAndRow}
  rowState[rowStateOffset + index * 2u] = NONE;
  rowState[rowStateOffset + index * 2u + 1u] =
    philox4x32(vec4<u32>(row, start, 0u, 0u), vec2<u32>(parameters[parametersOffset], PURPOSE)).x;
  regionPhase[regionPhaseOffset + index] = 0u;`
      })
    ];

    for (let round = 0; round < growthRounds; round++) {
      const roundId = `${id}-round-${round}`;
      // Ungated, one thread: decides from the counters whether this round runs, whether it
      // starts a seed wave, and advances the round cursor (rule 12: no loop index in the WGSL).
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-control`,
          operation: 'GPUMaxPRegions',
          variant: 'round-control',
          invocationCount: 1,
          bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}, gate.binding],
          declarations,
          body: `let round = state[stateOffset + STATE_ROUND];
  let period = (state[stateOffset + STATE_EXPECTED_SIZE] + 1u) / 2u;
  var anyActive = false;
  var anyWave = false;
  // Every start decides from its own counters, so its construction does not depend on the
  // other starts (a start that has settled never becomes active again).
  for (var start = 0u; start < STARTS; start++) {
    let record = stateOffset + STATE_STARTS + start * START_RECORD;
    let growing = state[record + WORD_GROWING];
    let unassigned = state[record + WORD_UNASSIGNED];
    let isActive = state[stateOffset + STATE_INVALID] == 0u && (growing > 0u || unassigned > 0u);
    let isWave = isActive && unassigned > 0u && (growing == 0u || round >= state[record + WORD_NEXT_WAVE]);
    if (isWave) {
      let fraction = bitcast<f32>(state[record + WORD_WAVE_FRACTION]);
      state[record + WORD_CUT_KEY] = select(u32(fraction * 4294967040.0), NONE, fraction >= 1.0);
      state[record + WORD_NEXT_WAVE] = round + period;
      state[record + WORD_WAVE_FRACTION] = bitcast<u32>(min(1.0, fraction * ${WAVE_GROWTH}));
    }
    state[record + WORD_WAVE_FLAG] = select(0u, 1u, isWave);
    anyActive = anyActive || isActive;
    anyWave = anyWave || isWave;
  }
  ${writeGate('anyActive', 'anyWave', 'false')}
  if (anyActive) { state[stateOffset + STATE_ROUND] = round + 1u; }`
        }),
        // Clear the claim records of the round.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-reset`,
          operation: 'GPUMaxPRegions',
          variant: 'round-reset',
          invocationCount: scratchRows,
          bindings: [{name: 'claims', view: claims, type: 'u32', access: 'read_write'}],
          declarations,
          body: `for (var word = 0u; word < CLAIM_STRIDE; word++) { claims[claimsOffset + index * CLAIM_STRIDE + word] = NONE; }`,
          ...gated(0)
        }),
        // Seed wave, step 1: mark the unassigned rows that are local key minima among the
        // eligible unassigned rows next to them and are not next to a growing region.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-seed-mark`,
          operation: 'GPUMaxPRegions',
          variant: 'seed-mark',
          invocationCount: scratchRows,
          bindings: bindingsOf(
            offsetsBinding(),
            neighborsBinding(),
            {name: 'rowState', view: rowState, type: 'u32', access: 'read'},
            {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ),
          declarations,
          body: `${startAndRow}
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  let record = stateOffset + STATE_STARTS + start * START_RECORD;
  if (state[record + WORD_WAVE_FLAG] == 0u) { return; }
  let cutKey = state[record + WORD_CUT_KEY];
  let key = rowState[stateBase + row * 2u + 1u];
  if (rowState[stateBase + row * 2u] != NONE || key > cutKey) { return; }
  var blocked = false;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u] && !blocked; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor == row) { continue; }
    let neighborLabel = rowState[stateBase + neighbor * 2u];
    if (neighborLabel != NONE) {
      blocked = regionPhase[phaseBase + neighborLabel] == PHASE_GROWING;
    } else {
      let neighborKey = rowState[stateBase + neighbor * 2u + 1u];
      blocked = neighborKey <= cutKey && (neighborKey < key || (neighborKey == key && neighbor < row));
    }
  }
  if (!blocked) { regionPhase[phaseBase + row] = PHASE_MARKED; }`,
          ...gated(1)
        }),
        // Seed wave, step 2: the marked rows become regions of one row.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-seed-commit`,
          operation: 'GPUMaxPRegions',
          variant: 'seed-commit',
          invocationCount: scratchRows,
          bindings: bindingsOf(
            {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
            {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read_write'},
            {name: 'regionData', view: regionData, type: 'f32', access: 'read_write'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'regionSums', view: regionSums, type: 'f32', access: 'read_write'},
            {name: 'extensive', view: packedExtensive, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ),
          declarations,
          body: `${startAndRow}
  if (regionPhase[regionPhaseOffset + index] != PHASE_MARKED) { return; }
  rowState[rowStateOffset + index * 2u] = row;
  let regionBase = regionDataOffset + index * REGION_STRIDE;
  let rowExtensive = extensive[extensiveOffset + row];
  regionData[regionBase] = 1.0;
  regionData[regionBase + 1u] = rowExtensive;
  regionData[regionBase + 2u] = 0.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + row * COLUMNS + column];
    regionData[regionBase + REGION_HEADER + column] = value;
    regionSums[regionSumsOffset + index * COLUMNS + column] = value;
  }
  let record = stateOffset + STATE_STARTS + start * START_RECORD;
  atomicSub(&state[record + WORD_UNASSIGNED], 1u);
  if (rowExtensive >= extensive[extensiveOffset + ROWS]) {
    regionPhase[regionPhaseOffset + index] = PHASE_COMPLETE;
  } else {
    regionPhase[regionPhaseOffset + index] = PHASE_GROWING;
    atomicAdd(&state[record + WORD_GROWING], 1u);
  }`,
          ...gated(1)
        })
      );
      // Growth: every unassigned row offers itself to each adjacent growing region. Pass 1 keeps
      // each region's smallest key, pass 2 its smallest tie among the rows at that key.
      for (const pass of [1, 2]) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${roundId}-propose-${pass}`,
            operation: 'GPUMaxPRegions',
            variant: `propose-${pass}`,
            invocationCount: scratchRows,
            bindings: bindingsOf(
              offsetsBinding(),
              neighborsBinding(),
              {name: 'rowState', view: rowState, type: 'u32', access: 'read'},
              {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read'},
              {name: 'regionData', view: regionData, type: 'f32', access: 'read'},
              {name: 'values', view: table, type: 'f32', access: 'read'},
              {name: 'claims', view: claims, type: 'atomic<u32>', access: 'read_write'},
              {name: 'state', view: state, type: 'u32', access: 'read'}
            ),
            declarations: `${declarations}${distanceWGSL}`,
            body: `${startAndRow}
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  if (rowState[stateBase + row * 2u] != NONE) { return; }
  var priority = 0u;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    priority += select(0u, 1u, neighbor != row && rowState[stateBase + neighbor * 2u] == NONE);
  }
  priority = min(priority, 63u);
  let salt = state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_SALT];
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let region = rowState[stateBase + neighbors[neighborsOffset + slot] * 2u];
    if (region == NONE || regionPhase[phaseBase + region] != PHASE_GROWING) { continue; }
    // Fewest unassigned neighbors first, then nearest to the region mean (positive float bits
    // order like the values; the low six bits are dropped to make room for the priority).
    let key = (priority << 26u) | (bitcast<u32>(getDistance(start, row, region)) >> 6u);
    let claimBase = claimsOffset + (start * ROWS + region) * CLAIM_STRIDE;
    ${
      pass === 1
        ? 'atomicMin(&claims[claimBase], key);'
        : 'if (key == atomicLoad(&claims[claimBase])) { atomicMin(&claims[claimBase + 1u], getTie(row, salt)); }'
    }
  }`,
            ...gated(0)
          })
        );
      }
      // Each region names its best row; a row named by several regions goes to the smallest
      // (key, tie).
      for (const pass of [1, 2]) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${roundId}-claim-${pass}`,
            operation: 'GPUMaxPRegions',
            variant: `claim-${pass}`,
            invocationCount: scratchRows,
            bindings: bindingsOf(
              {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read'},
              {name: 'claims', view: claims, type: 'atomic<u32>', access: 'read_write'},
              {name: 'state', view: state, type: 'u32', access: 'read'}
            ),
            declarations,
            body: `${startAndRow}
  if (regionPhase[regionPhaseOffset + index] != PHASE_GROWING) { return; }
  let ownBase = claimsOffset + index * CLAIM_STRIDE;
  let bestKey = atomicLoad(&claims[ownBase]);
  if (bestKey == NONE) { return; }
  let salt = state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_SALT];
  let claimedRow = getTieValue(atomicLoad(&claims[ownBase + 1u]), salt);
  let targetBase = claimsOffset + (start * ROWS + claimedRow) * CLAIM_STRIDE;
  ${
    pass === 1
      ? 'atomicMin(&claims[targetBase + 2u], bestKey);'
      : 'if (bestKey == atomicLoad(&claims[targetBase + 2u])) { atomicMin(&claims[targetBase + 3u], getTie(row, salt)); }'
  }`,
            ...gated(0)
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-commit`,
          operation: 'GPUMaxPRegions',
          variant: 'commit',
          invocationCount: scratchRows,
          bindings: bindingsOf(
            {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
            {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read_write'},
            {name: 'regionData', view: regionData, type: 'f32', access: 'read_write'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'regionSums', view: regionSums, type: 'f32', access: 'read_write'},
            {name: 'extensive', view: packedExtensive, type: 'f32', access: 'read'},
            {name: 'claims', view: claims, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ),
          declarations: `${declarations}${distanceWGSL}`,
          body: `${startAndRow}
  if (regionPhase[regionPhaseOffset + index] != PHASE_GROWING) { return; }
  let ownBase = claimsOffset + index * CLAIM_STRIDE;
  if (claims[ownBase] == NONE) {
    // No frontier row is left and the threshold is not reached.
    regionPhase[regionPhaseOffset + index] = PHASE_FAILED;
    atomicSub(&state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_GROWING], 1u);
    return;
  }
  let record = stateOffset + STATE_STARTS + start * START_RECORD;
  let salt = atomicLoad(&state[record + WORD_SALT]);
  let claimedRow = getTieValue(claims[ownBase + 1u], salt);
  if (claims[claimsOffset + (start * ROWS + claimedRow) * CLAIM_STRIDE + 3u] != getTie(row, salt)) { return; }
  let regionBase = regionDataOffset + index * REGION_STRIDE;
  let count = regionData[regionBase];
  // Incremental squared deviation about the mean: n / (n + 1) times the squared distance.
  regionData[regionBase + 2u] = regionData[regionBase + 2u] + (count / (count + 1.0)) * getDistance(start, claimedRow, row);
  regionData[regionBase] = count + 1.0;
  regionData[regionBase + 1u] = regionData[regionBase + 1u] + extensive[extensiveOffset + claimedRow];
  for (var column = 0u; column < COLUMNS; column++) {
    let sumIndex = regionSumsOffset + index * COLUMNS + column;
    let sum = regionSums[sumIndex] + values[valuesOffset + claimedRow * COLUMNS + column];
    regionSums[sumIndex] = sum;
    regionData[regionBase + REGION_HEADER + column] = sum / (count + 1.0);
  }
  rowState[rowStateOffset + (start * ROWS + claimedRow) * 2u] = row;
  atomicSub(&state[record + WORD_UNASSIGNED], 1u);
  if (regionData[regionBase + 1u] >= extensive[extensiveOffset + ROWS]) {
    regionPhase[regionPhaseOffset + index] = PHASE_COMPLETE;
    atomicSub(&state[record + WORD_GROWING], 1u);
  }`,
          ...gated(0)
        })
      );
    }

    // Enclaves: rows of failed and unfinished regions, and rows no region reached, join the most
    // similar adjacent completed region, round by round while some row moves.
    for (let round = 0; round < enclaveRounds; round++) {
      const roundId = `${id}-enclave-${round}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-control`,
          operation: 'GPUMaxPRegions',
          variant: 'enclave-control',
          invocationCount: 1,
          bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}, gate.binding],
          declarations,
          body: `let enclaveRound = state[stateOffset + STATE_ENCLAVE_ROUND];
  let isActive = state[stateOffset + STATE_INVALID] == 0u && (enclaveRound == 0u || state[stateOffset + STATE_MOVED] > 0u);
  ${writeGate('false', 'false', 'isActive')}
  state[stateOffset + STATE_MOVED] = 0u;
  state[stateOffset + STATE_ENCLAVE_ROUND] = enclaveRound + 1u;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-propose`,
          operation: 'GPUMaxPRegions',
          variant: 'enclave-propose',
          invocationCount: scratchRows,
          bindings: bindingsOf(
            offsetsBinding(),
            neighborsBinding(),
            {name: 'rowState', view: rowState, type: 'u32', access: 'read'},
            {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read'},
            {name: 'regionData', view: regionData, type: 'f32', access: 'read'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'claims', view: claims, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ),
          declarations: `${declarations}${distanceWGSL}`,
          body: `${startAndRow}
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  let label = rowState[stateBase + row * 2u];
  var bestRegion = NONE;
  if (label == NONE || regionPhase[phaseBase + label] != PHASE_COMPLETE) {
    let salt = state[stateOffset + STATE_STARTS + start * START_RECORD + WORD_SALT];
    var bestKey = NONE;
    var bestTie = NONE;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let region = rowState[stateBase + neighbors[neighborsOffset + slot] * 2u];
      if (region == NONE || regionPhase[phaseBase + region] != PHASE_COMPLETE) { continue; }
      let key = bitcast<u32>(getDistance(start, row, region));
      let tie = getTie(region, salt);
      if (key < bestKey || (key == bestKey && tie < bestTie)) {
        bestKey = key;
        bestTie = tie;
        bestRegion = region;
      }
    }
  }
  claims[claimsOffset + index * CLAIM_STRIDE + 3u] = bestRegion;`,
          ...gated(2)
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-commit`,
          operation: 'GPUMaxPRegions',
          variant: 'enclave-commit',
          invocationCount: scratchRows,
          bindings: bindingsOf(
            {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
            {name: 'claims', view: claims, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ),
          declarations,
          body: `let proposal = claims[claimsOffset + index * CLAIM_STRIDE + 3u];
  if (proposal == NONE) { return; }
  rowState[rowStateOffset + index * 2u] = proposal;
  atomicAdd(&state[stateOffset + STATE_MOVED], 1u);`,
          ...gated(2)
        })
      );
    }

    nodes.push(
      // One workgroup per start: rows nobody reached become singleton under-threshold regions,
      // the regions in use are flagged, and the start's statistics are reduced in a fixed order.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize-starts`,
        operation: 'GPUMaxPRegions',
        variant: 'finalize-starts',
        invocationCount: startCount * FINALIZE_WORKGROUP_SIZE,
        workgroupSize: FINALIZE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'rowState', view: rowState, type: 'u32', access: 'read_write'},
          {name: 'regionPhase', view: regionPhase, type: 'u32', access: 'read_write'},
          {name: 'regionData', view: regionData, type: 'f32', access: 'read_write'},
          {name: 'claims', view: claims, type: 'u32', access: 'read_write'},
          {name: 'startState', view: startState, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
var<workgroup> sumScratch: array<f32, ${FINALIZE_WORKGROUP_SIZE}>;
var<workgroup> countScratch: array<u32, ${FINALIZE_WORKGROUP_SIZE}>;

// Sum of one value per lane over the workgroup in a fixed order, available to every lane.
fn reduceSum(lane: u32, value: f32) -> f32 {
  sumScratch[lane] = value;
  workgroupBarrier();
  for (var stride = ${FINALIZE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { sumScratch[lane] = sumScratch[lane] + sumScratch[lane + stride]; }
    workgroupBarrier();
  }
  let total = sumScratch[0];
  workgroupBarrier();
  return total;
}

fn reduceCount(lane: u32, value: u32) -> u32 {
  countScratch[lane] = value;
  workgroupBarrier();
  for (var stride = ${FINALIZE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) { countScratch[lane] = countScratch[lane] + countScratch[lane + stride]; }
    workgroupBarrier();
  }
  let total = countScratch[0];
  workgroupBarrier();
  return total;
}`,
        body: `let lane = localInvocationIndex;
  // One workgroup per start, so \`start\` is uniform across the workgroup.
  let start = index / ${FINALIZE_WORKGROUP_SIZE}u;
  let stateBase = rowStateOffset + start * ROWS * 2u;
  let phaseBase = regionPhaseOffset + start * ROWS;
  let claimBase = claimsOffset + start * ROWS * CLAIM_STRIDE;
  for (var row = lane; row < ROWS; row += ${FINALIZE_WORKGROUP_SIZE}u) {
    claims[claimBase + row * CLAIM_STRIDE] = 0u;
  }
  storageBarrier();
  workgroupBarrier();
  for (var row = lane; row < ROWS; row += ${FINALIZE_WORKGROUP_SIZE}u) {
    var label = rowState[stateBase + row * 2u];
    if (label == NONE) {
      // Nobody reached this row: it is its own under-threshold region.
      label = row;
      rowState[stateBase + row * 2u] = row;
      regionPhase[phaseBase + row] = PHASE_FAILED;
      regionData[regionDataOffset + (start * ROWS + row) * REGION_STRIDE + 2u] = 0.0;
    }
    claims[claimBase + label * CLAIM_STRIDE] = 1u;
  }
  storageBarrier();
  workgroupBarrier();
  var squares = 0.0;
  var regions = 0u;
  var infeasible = 0u;
  for (var row = lane; row < ROWS; row += ${FINALIZE_WORKGROUP_SIZE}u) {
    if (claims[claimBase + row * CLAIM_STRIDE] == 1u) {
      regions += 1u;
      infeasible += select(1u, 0u, regionPhase[phaseBase + row] == PHASE_COMPLETE);
      squares += regionData[regionDataOffset + (start * ROWS + row) * REGION_STRIDE + 2u];
    }
  }
  let regionTotal = reduceCount(lane, regions);
  let infeasibleTotal = reduceCount(lane, infeasible);
  let squaresTotal = reduceSum(lane, squares);
  if (lane == 0u) {
    let base = startStateOffset + start * START_STATE_STRIDE;
    startState[base] = regionTotal;
    startState[base + 1u] = infeasibleTotal;
    startState[base + 2u] = bitcast<u32>(squaresTotal);
    startState[base + 3u] = 0u;
  }`
      }),
      // Fewest leftover regions, then most regions, then smallest grown sum of squares, then
      // lowest start (strict comparisons keep the earlier start on a full tie).
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
  state[stateOffset + STATE_SELECTED] = best;`
      }),
      // Dense labels of the selected start: rank of the region ID among the regions in use.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-flags`,
        operation: 'GPUMaxPRegions',
        variant: 'region-flags',
        invocationCount: rowCount,
        bindings: [
          {name: 'claims', view: claims, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: 'flags[flagsOffset + index] = claims[claimsOffset + (state[stateOffset + STATE_SELECTED] * ROWS + index) * CLAIM_STRIDE];'
      }),
      ...new GPUScan({id: `${id}-region-ranks`, input: flags, output: ranks}).getCommandNodes(
        graph
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-copy-best`,
        operation: 'GPUMaxPRegions',
        variant: 'copy-best',
        invocationCount: rowCount,
        bindings: [
          {name: 'rowState', view: rowState, type: 'u32', access: 'read'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'labelsOut', view: bestLabels, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: 'labelsOut[labelsOutOffset + index] = ranks[ranksOffset + rowState[rowStateOffset + (state[stateOffset + STATE_SELECTED] * ROWS + index) * 2u]];'
      })
    );

    if (improving) {
      // The selected construction is refined by greedy AZP moves that keep every region at or
      // above the extensive threshold. The adjacency is already symmetric (`information.symmetric`).
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
          maximumIterations,
          // Same `[seed]` layout as the AZP parameters (rule 19: one source of the seed).
          parameters: props.parameters,
          output: {
            labels: output.labels,
            objective: improvementObjective!,
            status: improvementStatus!
          }
        }).getCommandNodes(graph)
      );
    } else {
      // The exact within-region sum of squares of the returned labels.
      nodes.push(
        ...new GPURegionPartitionEvaluation({
          id: `${id}-evaluation`,
          values: table,
          columnCount,
          labels: bestLabels,
          labelCapacity: rowCount,
          output: {summary: evaluationSummary!}
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
          improving
            ? {
                name: 'improvementObjective',
                view: improvementObjective!,
                type: 'f32',
                access: 'read'
              }
            : {name: 'evaluationSummary', view: evaluationSummary!, type: 'f32', access: 'read'},
          ...(improving
            ? [
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
        body: `let best = state[stateOffset + STATE_SELECTED];
  let base = startStateOffset + best * START_STATE_STRIDE;
  var invalid = state[stateOffset + STATE_INVALID];
  ${
    improving
      ? `// Only an improvement-phase rejection (not already a max-p invalid input) adds to the count.
  invalid = max(invalid, improvementStatus[improvementStatusOffset + 3u]);
  let moveCount = improvementStatus[improvementStatusOffset];
  let withinSquares = improvementObjective[improvementObjectiveOffset];`
      : `let moveCount = 0u;
  let withinSquares = evaluationSummary[evaluationSummaryOffset + ${GPU_REGION_PARTITION_EVALUATION_LAYOUT.withinSumOfSquares}u];`
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
