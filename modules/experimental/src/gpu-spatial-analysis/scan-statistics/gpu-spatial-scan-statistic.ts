// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {PERMUTATION_RANDOM_WGSL} from '../permutation-inference/permutation-random';
import {
  GPU_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SCAN_STATISTIC_MAXIMUM_CLUSTERS,
  GPU_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS,
  GPU_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS,
  GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES,
  GPU_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SCAN_STATISTIC_SUMMARY_LENGTH
} from './scan-statistic-parameters';

const OPERATION = 'GPUSpatialScanStatistic';
/** Fixed-order reduction blocks (at most 256 invocations walk the cells serially). */
const MAXIMUM_BLOCKS = 256;
/** Case-draw blocks per replicate: each owns a Philox stream, so results ignore the dispatch. */
const CASE_BLOCKS = 1024;
/** Philox counter tag of the multinomial case draws, apart from the permutation streams. */
const SCAN_STREAM = 0x5ca75747;
/** Words of one per-center record: LLR bits, zones, first bucket, last bucket, cases, expected. */
const CENTER_WORDS = 8;
/** Invocations of the evaluate workgroups: each reduces the window maxima of 256 centers of one replicate. */
const EVALUATE_WORKGROUP_SIZE = 256;
/** Largest replicate count buffer, in words (the default storage binding limit is 128 MiB). */
const MAXIMUM_REPLICATE_WORDS = 2 ** 25;

/**
 * Properties for {@link GPUSpatialScanStatistic}.
 *
 * Per-frame (no recompile): the contents of `cases`, `baseline`, `positions` and `parameters`
 * (seed, replicate count, window limits, window shape). Compile-time: the zone count,
 * `timeBuckets`, `maximumWindowZones`, `maximumPermutations` and `maximumClusters`.
 */
export type GPUSpatialScanStatisticProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'scan-statistic'`. */
  id?: string;
  /** Zone centroids in planar coordinates, finite, one row per zone. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Observed case counts per (zone, time bucket), indexed `zone * timeBuckets + bucket`, the
   * layout of `GPUTemporalReduction` counts. The total must be below 2^32.
   */
  cases: GraphDataView<'uint32'>;
  /**
   * Baseline per (zone, time bucket) in the same layout: population at risk or expected counts,
   * finite and non-negative, with any total scale. The null model is a Poisson process with
   * `E[cases_i] = baseline_i * C / sum(baseline)`, `C` the total case count.
   */
  baseline: GraphDataView<'float32'>;
  /** Time buckets per zone: 1 for a purely spatial scan. Compile-time, at most 32. */
  timeBuckets?: number;
  /**
   * Largest window in zones, 1 to 32. Windows are the `k` nearest zones of each center, `k <=
   * maximumWindowZones` (per-frame `maximumWindowZones` may lower it). Compile-time.
   * Defaults to 32.
   */
  maximumWindowZones?: number;
  /** Compile-time upper bound on the per-frame replicate count, at most 16384. */
  maximumPermutations: number;
  /** Compile-time capacity of the cluster list, at most 64. Defaults to 8. */
  maximumClusters?: number;
  /**
   * Per-frame parameters: `GPU_SCAN_STATISTIC_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpatialScanParameterValues`.
   */
  parameters: GraphDataView<'uint32'>;
  /**
   * Caller-owned cluster geometry, `maximumClusters * GPU_SCAN_STATISTIC_INDEX_WORDS` uint32
   * words, laid out as `GPU_SCAN_STATISTIC_CLUSTER_INDEX`. Records past `clusterCount` are 0.
   */
  clusterIndices: GraphDataView<'uint32'>;
  /**
   * Caller-owned cluster statistics, `maximumClusters * GPU_SCAN_STATISTIC_STATISTIC_WORDS`
   * float32 words, laid out as `GPU_SCAN_STATISTIC_CLUSTER`.
   */
  clusterStatistics: GraphDataView<'float32'>;
  /**
   * Caller-owned float32, `maximumPermutations + 1` rows: element 0 is the largest observed LLR
   * (the most likely cluster), elements `1..P` the largest LLR of each Monte Carlo replicate
   * (rows past `P` are 0).
   */
  statistics: GraphDataView<'float32'>;
  /** Caller-owned `GPU_SCAN_STATISTIC_SUMMARY_LENGTH` uint32 words, see `GPU_SCAN_STATISTIC_SUMMARY`. */
  summary: GraphDataView<'uint32'>;
  /** Optional caller-owned float32, one row per zone: the best observed LLR of windows centered there. */
  zoneStatistics?: GraphDataView<'float32'>;
};

/**
 * Kulldorff's spatial and space-time scan statistic for the Poisson model (SaTScan; CARTO
 * `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or points, with case counts and
 * baselines.
 *
 * Windows: for every center zone, the `k` nearest zones for every `k` up to the maximum (a circle:
 * with the default shape a window never splits zones at equal distance; ties are exact f32
 * squared-distance equality), kept while they hold at most `maximumPopulationFraction` of the total
 * baseline. With `timeBuckets > 1` each window is a cylinder over every run of at most
 * `maximumTimeBuckets` consecutive buckets. A zone list is the `maximumWindowZones` nearest zones,
 * so a circle that would extend past the list is cut there.
 *
 * Statistic: for a window with `c` cases and null expectation `e` (total cases `C`), the Poisson
 * log-likelihood ratio `c ln(c/e) + (C - c) ln((C - c)/(C - e))` when `c > e` (high rates only), else
 * 0. Windows with no baseline are skipped. Evaluated in f32; the sum of the two terms cancels
 * for `c` close to `e`, so LLRs below about 1e-3 are approximate.
 *
 * Clusters: the most likely cluster is the window with the largest LLR. Secondary clusters are
 * found greedily: the best window sharing no zone with any earlier cluster (time is ignored for
 * overlap), until `maximumClusters` or no positive LLR remains. Ties pick the lowest center zone.
 *
 * Inference: `P` Monte Carlo replicates redistribute the `C` cases over the zone-buckets by a
 * multinomial with probabilities proportional to the baseline (each case an independent draw from
 * an integer cumulative table, Philox 4x32-10 streams from `permutation-inference`), and record the
 * largest LLR of each replicate. A cluster's p-value is `(1 + #{replicate maxima >= LLR}) /
 * (P + 1)`; secondary clusters use the same maxima distribution, as SaTScan does.
 *
 * Reductions run in fixed order: integer atomic adds for the draws, serial block sums, and
 * argmax with the lowest index winning ties. Results are a pure function of the inputs and seed.
 * Capacity: one `(P + 1) * zones * timeBuckets` word replicate table (at most 2^25 words) and
 * `(P + 1) * zones` window maxima; work is about `(P + 1) * zones * maximumWindowZones *
 * timeBuckets^2`.
 */
export class GPUSpatialScanStatistic implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialScanStatisticProps;
  /** Number of zones. */
  readonly zones: number;
  /** Time buckets per zone. */
  readonly timeBuckets: number;
  /** Zones listed per center, `min(maximumWindowZones, zones)`. */
  readonly listLength: number;
  /** Cluster capacity. */
  readonly maximumClusters: number;

  constructor(props: GPUSpatialScanStatisticProps) {
    const id = props.id ?? 'scan-statistic';
    this.id = id;
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedUint32View(props.cases, `${id} cases`);
    validatePackedView(props.baseline, ['float32'], `${id} baseline`);
    validatePackedUint32View(props.parameters, `${id} parameters`);
    validatePackedUint32View(props.clusterIndices, `${id} clusterIndices`);
    validatePackedView(props.clusterStatistics, ['float32'], `${id} clusterStatistics`);
    validatePackedView(props.statistics, ['float32'], `${id} statistics`);
    validatePackedUint32View(props.summary, `${id} summary`);
    this.zones = props.positions.length;
    this.timeBuckets = props.timeBuckets ?? 1;
    this.maximumClusters = props.maximumClusters ?? 8;
    const maximumWindowZones = props.maximumWindowZones ?? GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES;
    const {maximumPermutations} = props;
    if (this.zones < 1) {
      throw new Error(`${id} needs at least one zone`);
    }
    if (
      !Number.isInteger(this.timeBuckets) ||
      this.timeBuckets < 1 ||
      this.timeBuckets > GPU_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS
    ) {
      throw new Error(
        `${id} timeBuckets must be an integer in [1, ${GPU_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS}]`
      );
    }
    if (
      !Number.isInteger(maximumWindowZones) ||
      maximumWindowZones < 1 ||
      maximumWindowZones > GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES
    ) {
      throw new Error(
        `${id} maximumWindowZones must be an integer in [1, ${GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES}]`
      );
    }
    if (
      !Number.isInteger(this.maximumClusters) ||
      this.maximumClusters < 1 ||
      this.maximumClusters > GPU_SCAN_STATISTIC_MAXIMUM_CLUSTERS
    ) {
      throw new Error(
        `${id} maximumClusters must be an integer in [1, ${GPU_SCAN_STATISTIC_MAXIMUM_CLUSTERS}]`
      );
    }
    if (
      !Number.isInteger(maximumPermutations) ||
      maximumPermutations < 1 ||
      maximumPermutations > GPU_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS
    ) {
      throw new Error(
        `${id} maximumPermutations must be an integer in [1, ${GPU_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS}]`
      );
    }
    const cells = this.zones * this.timeBuckets;
    if (props.cases.length !== cells || props.baseline.length !== cells) {
      throw new Error(`${id} cases and baseline must hold zones * timeBuckets rows`);
    }
    if ((maximumPermutations + 1) * cells > MAXIMUM_REPLICATE_WORDS) {
      throw new Error(
        `${id} (maximumPermutations + 1) * zones * timeBuckets must not exceed ${MAXIMUM_REPLICATE_WORDS}`
      );
    }
    this.listLength = Math.min(maximumWindowZones, this.zones);
    if (props.parameters.length < GPU_SCAN_STATISTIC_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_SCAN_STATISTIC_PARAMETER_LENGTH} words`);
    }
    if (props.clusterIndices.length < this.maximumClusters * GPU_SCAN_STATISTIC_INDEX_WORDS) {
      throw new Error(`${id} clusterIndices must hold maximumClusters * 4 rows`);
    }
    if (
      props.clusterStatistics.length <
      this.maximumClusters * GPU_SCAN_STATISTIC_STATISTIC_WORDS
    ) {
      throw new Error(`${id} clusterStatistics must hold maximumClusters * 8 rows`);
    }
    if (props.statistics.length < maximumPermutations + 1) {
      throw new Error(`${id} statistics must hold maximumPermutations + 1 rows`);
    }
    if (props.summary.length < GPU_SCAN_STATISTIC_SUMMARY_LENGTH) {
      throw new Error(`${id} summary must hold ${GPU_SCAN_STATISTIC_SUMMARY_LENGTH} rows`);
    }
    if (props.zoneStatistics) {
      validatePackedView(props.zoneStatistics, ['float32'], `${id} zoneStatistics`);
      if (props.zoneStatistics.length < this.zones) {
        throw new Error(`${id} zoneStatistics must hold one row per zone`);
      }
    }
    const outputs = [
      props.clusterIndices,
      props.clusterStatistics,
      props.statistics,
      props.summary,
      ...(props.zoneStatistics ? [props.zoneStatistics] : [])
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.positions,
      props.cases,
      props.baseline,
      props.parameters
    ]);
    if (new Set(outputs.map(output => output.buffer)).size !== outputs.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the nodes in order: `nearest`, `mask`, `block-totals`, `totals`, the cumulative table
   * (`weight-sums`, `weight-scan`, `cdf`), `init-counts`, `scatter`, `evaluate`,
   * `select-init`, then per cluster `round-mask`, `round-partials` and `round-select`, and `summary`.
   * `evaluate` reduces the per-replicate maxima itself (workgroup max, one `atomicMax` per
   * workgroup), so there is no per-center candidate table and no separate maximum pass.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zones, timeBuckets, listLength, maximumClusters} = this;
    const {positions, cases, baseline, parameters, maximumPermutations} = props;
    const {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      cases,
      baseline,
      parameters,
      clusterIndices,
      clusterStatistics,
      statistics,
      summary,
      ...(zoneStatistics ? [zoneStatistics] : [])
    ]);
    const cells = zones * timeBuckets;
    const blocks = Math.min(MAXIMUM_BLOCKS, cells);
    const cellsPerBlock = Math.ceil(cells / blocks);
    const zoneBlocks = Math.min(MAXIMUM_BLOCKS, zones);
    const zonesPerBlock = Math.ceil(zones / zoneBlocks);
    const slotCount = maximumPermutations + 1;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const neighbors = u32('neighbors', zones * listLength);
    const distances = f32('distances', zones * listLength);
    const windowMask = u32('window-mask', zones);
    const blockBaseline = f32('block-baseline', blocks);
    const blockCases = u32('block-cases', blocks);
    const blockWeights = u32('block-weights', blocks);
    const blockOffsets = u32('block-offsets', blocks);
    const totals = u32('totals', 4);
    const cumulative = u32('cumulative', cells);
    const counts = u32('counts', slotCount * cells);
    const evaluateGroups = Math.ceil(zones / EVALUATE_WORKGROUP_SIZE);
    const centerBest = u32('center-best', zones * CENTER_WORDS);
    const alive = u32('alive', zones);
    const partials = u32('partials', zoneBlocks * 2);
    const state = u32('state', 8);

    const common = /* wgsl */ `
const ZONES: u32 = ${zones}u;
const LIST: u32 = ${listLength}u;
const BUCKETS: u32 = ${timeBuckets}u;
const CELLS: u32 = ${cells}u;
const BLOCKS: u32 = ${blocks}u;
const CELLS_PER_BLOCK: u32 = ${cellsPerBlock}u;
const ZONE_BLOCKS: u32 = ${zoneBlocks}u;
const ZONES_PER_BLOCK: u32 = ${zonesPerBlock}u;
const MAXIMUM_PERMUTATIONS: u32 = ${maximumPermutations}u;
const CENTER_WORDS: u32 = ${CENTER_WORDS}u;
const NO_CENTER: u32 = 0xffffffffu;
// Weight of one cell in the integer cumulative table: its baseline share of 2^31, rounded down.
fn getCellWeight(cellBaseline: f32, totalBaseline: f32) -> u32 {
  if (!(totalBaseline > 0.0) || !(cellBaseline > 0.0)) {
    return 0u;
  }
  return u32(floor(cellBaseline / totalBaseline * 2147483648.0));
}`;
    const readers = /* wgsl */ `
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutations() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn readMaximumFraction() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
fn readMaximumZones() -> u32 {
  return clamp(parameters[parametersOffset + 4u], 1u, LIST);
}
fn readMaximumBuckets() -> u32 {
  return clamp(parameters[parametersOffset + 5u], 1u, BUCKETS);
}
fn readWindowShape() -> u32 {
  return parameters[parametersOffset + 6u];
}`;
    const withParameters = `${common}${readers}`;
    const likelihood = /* wgsl */ `
fn getPoissonLogLikelihoodRatio(observed: f32, windowBaseline: f32, totalCases: f32, totalBaseline: f32) -> f32 {
  let expected = windowBaseline * totalCases / totalBaseline;
  if (!(expected > 0.0) || observed <= expected) {
    return 0.0;
  }
  var ratio = observed * log(observed / expected);
  let remaining = totalCases - observed;
  if (remaining > 0.0) {
    ratio += remaining * log(remaining / (totalCases - expected));
  }
  return ratio;
}`;

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-nearest`,
        operation: OPERATION,
        variant: 'nearest',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
        ],
        invocationCount: zones,
        declarations: `const ZONES: u32 = ${zones}u;
const LIST: u32 = ${listLength}u;`,
        // Insertion into a sorted private list, ties by ascending zone index.
        body: `var bestIndex: array<u32, LIST>;
  var bestDistance: array<f32, LIST>;
  var filled = 0u;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  for (var zone = 0u; zone < ZONES; zone++) {
    let dx = positions[positionsOffset + zone * 2u] - centerX;
    let dy = positions[positionsOffset + zone * 2u + 1u] - centerY;
    let distance = dx * dx + dy * dy;
    if (filled < LIST || distance < bestDistance[LIST - 1u]) {
      var slot = select(LIST - 1u, filled, filled < LIST);
      filled = min(filled + 1u, LIST);
      while (slot > 0u && bestDistance[slot - 1u] > distance) {
        bestDistance[slot] = bestDistance[slot - 1u];
        bestIndex[slot] = bestIndex[slot - 1u];
        slot--;
      }
      bestDistance[slot] = distance;
      bestIndex[slot] = zone;
    }
  }
  for (var slot = 0u; slot < LIST; slot++) {
    neighbors[neighborsOffset + index * LIST + slot] = bestIndex[slot];
    distances[distancesOffset + index * LIST + slot] = bestDistance[slot];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-totals`,
        operation: OPERATION,
        variant: 'block-totals',
        bindings: [
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'cases', view: cases, type: 'u32', access: 'read'},
          {name: 'blockBaseline', view: blockBaseline, type: 'f32', access: 'read_write'},
          {name: 'blockCases', view: blockCases, type: 'u32', access: 'read_write'}
        ],
        invocationCount: blocks,
        declarations: common,
        body: `var baselineSum = 0.0;
  var caseSum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    baselineSum += baseline[baselineOffset + cell];
    caseSum += cases[casesOffset + cell];
  }
  blockBaseline[blockBaselineOffset + index] = baselineSum;
  blockCases[blockCasesOffset + index] = caseSum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-totals`,
        operation: OPERATION,
        variant: 'totals',
        bindings: [
          {name: 'blockBaseline', view: blockBaseline, type: 'f32', access: 'read'},
          {name: 'blockCases', view: blockCases, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: common,
        body: `var baselineSum = 0.0;
  var caseSum = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    baselineSum += blockBaseline[blockBaselineOffset + block];
    caseSum += blockCases[blockCasesOffset + block];
  }
  totals[totalsOffset] = caseSum;
  totals[totalsOffset + 1u] = bitcast<u32>(baselineSum);
  totals[totalsOffset + 2u] = 0u;
  totals[totalsOffset + 3u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weight-sums`,
        operation: OPERATION,
        variant: 'weight-sums',
        bindings: [
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'blockWeights', view: blockWeights, type: 'u32', access: 'read_write'}
        ],
        invocationCount: blocks,
        declarations: common,
        body: `let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  var weightSum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    weightSum += getCellWeight(baseline[baselineOffset + cell], totalBaseline);
  }
  blockWeights[blockWeightsOffset + index] = weightSum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weight-scan`,
        operation: OPERATION,
        variant: 'weight-scan',
        bindings: [
          {name: 'blockWeights', view: blockWeights, type: 'u32', access: 'read'},
          {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read_write'},
          {name: 'totals', view: totals, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: common,
        body: `var running = 0u;
  for (var block = 0u; block < BLOCKS; block++) {
    blockOffsets[blockOffsetsOffset + block] = running;
    running += blockWeights[blockWeightsOffset + block];
  }
  totals[totalsOffset + 2u] = running;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cdf`,
        operation: OPERATION,
        variant: 'cdf',
        bindings: [
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read'},
          {name: 'cumulative', view: cumulative, type: 'u32', access: 'read_write'}
        ],
        invocationCount: blocks,
        declarations: common,
        body: `let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  var running = blockOffsets[blockOffsetsOffset + index];
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    running += getCellWeight(baseline[baselineOffset + cell], totalBaseline);
    cumulative[cumulativeOffset + cell] = running;
  }`
      }),
      // Slot 0 of the count table is the observed data, slots 1..P the replicates.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-init-counts`,
        operation: OPERATION,
        variant: 'init-counts',
        bindings: [
          {name: 'cases', view: cases, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
          {name: 'statistics', view: statistics, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(slotCount * cells, statistics.length),
        declarations: `${common}
const COUNT_WORDS: u32 = ${slotCount * cells}u;
const STATISTIC_WORDS: u32 = ${statistics.length}u;`,
        // The same pass zeroes `statistics` (0 bits are 0.0), the target of the evaluate maxima.
        body: `if (index < COUNT_WORDS) {
    counts[countsOffset + index] = select(0u, cases[casesOffset + index], index < CELLS);
  }
  if (index < STATISTIC_WORDS) {
    statistics[statisticsOffset + index] = 0u;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: OPERATION,
        variant: 'scatter',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'cumulative', view: cumulative, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: maximumPermutations * CASE_BLOCKS,
        declarations: `${PERMUTATION_RANDOM_WGSL}
${withParameters}
const CASE_BLOCKS: u32 = ${CASE_BLOCKS}u;
const SCAN_STREAM: u32 = ${SCAN_STREAM}u;`,
        body: `let replicate = index / CASE_BLOCKS;
  let block = index - replicate * CASE_BLOCKS;
  let totalCases = totals[totalsOffset];
  let totalWeight = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalWeight == 0u) {
    return;
  }
  let chunk = (totalCases + CASE_BLOCKS - 1u) / CASE_BLOCKS;
  let first = block * chunk;
  let stop = min(first + chunk, totalCases);
  var stream = createPhiloxStream(readSeedKey(), replicate, block, SCAN_STREAM);
  for (var draw = first; draw < stop; draw++) {
    let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), totalWeight);
    var low = 0u;
    var high = CELLS;
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (cumulative[cumulativeOffset + middle] > drawTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    atomicAdd(&counts[countsOffset + (replicate + 1u) * CELLS + low], 1u);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mask`,
        operation: OPERATION,
        variant: 'mask',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'windowMask', view: windowMask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: zones,
        declarations: withParameters,
        // Bit m-1 of the mask: the window of the m nearest zones is a candidate.
        body: `let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let limit = readMaximumFraction() * totalBaseline;
  let maximumZones = readMaximumZones();
  let nearestShape = readWindowShape() == 1u;
  var populated = 0.0;
  var mask = 0u;
  for (var slot = 0u; slot < maximumZones; slot++) {
    let zone = neighbors[neighborsOffset + index * LIST + slot];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    if (populated > limit) {
      break;
    }
    let separated = nearestShape || slot + 1u == LIST ||
      distances[distancesOffset + index * LIST + slot] < distances[distancesOffset + index * LIST + slot + 1u];
    if (separated) {
      mask |= 1u << slot;
    }
  }
  windowMask[windowMaskOffset + index] = mask;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-evaluate`,
        operation: OPERATION,
        variant: 'evaluate',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'windowMask', view: windowMask, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'baseline', view: baseline, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'atomic<u32>', access: 'read_write'},
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read_write'}
        ],
        // One workgroup per (replicate, 256 consecutive centers); every invocation reaches the
        // barriers, so the body is unguarded.
        invocationCount: slotCount * evaluateGroups * EVALUATE_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${withParameters}
${likelihood}
const EVALUATE_GROUPS: u32 = ${evaluateGroups}u;
const EVALUATE_SIZE: u32 = ${EVALUATE_WORKGROUP_SIZE}u;
var<workgroup> maxima: array<f32, ${EVALUATE_WORKGROUP_SIZE}>;`,
        body: `let group = index / EVALUATE_SIZE;
  let slot = group / EVALUATE_GROUPS;
  let center = (group - slot * EVALUATE_GROUPS) * EVALUATE_SIZE + localInvocationIndex;
  var best = 0.0;
  if (slot <= readPermutations() && center < ZONES) {
  let totalCases = f32(totals[totalsOffset]);
  let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let maximumBuckets = readMaximumBuckets();
  let mask = windowMask[windowMaskOffset + center];
  let slotStart = slot * CELLS;
  var caseSums: array<u32, BUCKETS>;
  var baselineSums: array<f32, BUCKETS>;
  var bestZones = 0u;
  var bestFirst = 0u;
  var bestLast = 0u;
  var bestCases = 0.0;
  var bestExpected = 0.0;
  for (var slotIndex = 0u; slotIndex < LIST; slotIndex++) {
    if ((mask >> slotIndex) == 0u) {
      break;
    }
    let zone = neighbors[neighborsOffset + center * LIST + slotIndex];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      caseSums[bucket] += counts[countsOffset + slotStart + zone * BUCKETS + bucket];
      baselineSums[bucket] += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    if (((mask >> slotIndex) & 1u) == 0u) {
      continue;
    }
    for (var first = 0u; first < BUCKETS; first++) {
      var windowCases = 0u;
      var windowBaseline = 0.0;
      for (var last = first; last < min(first + maximumBuckets, BUCKETS); last++) {
        windowCases += caseSums[last];
        windowBaseline += baselineSums[last];
        let ratio = getPoissonLogLikelihoodRatio(f32(windowCases), windowBaseline, totalCases, totalBaseline);
        if (ratio > best) {
          best = ratio;
          bestZones = slotIndex + 1u;
          bestFirst = first;
          bestLast = last;
          bestCases = f32(windowCases);
          bestExpected = windowBaseline * totalCases / totalBaseline;
        }
      }
    }
  }
  if (slot == 0u) {
    let record = centerBestOffset + center * CENTER_WORDS;
    centerBest[record] = bitcast<u32>(best);
    centerBest[record + 1u] = bestZones;
    centerBest[record + 2u] = bestFirst;
    centerBest[record + 3u] = bestLast;
    centerBest[record + 4u] = bitcast<u32>(bestCases);
    centerBest[record + 5u] = bitcast<u32>(bestExpected);
    centerBest[record + 6u] = 0u;
    centerBest[record + 7u] = 0u;
  }
  }
  // Workgroup max of the window maxima, then one atomicMax per workgroup: LLRs are
  // non-negative, so their f32 bits order as u32, and max is order independent (deterministic).
  maxima[localInvocationIndex] = best;
  workgroupBarrier();
  for (var stride = EVALUATE_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      maxima[localInvocationIndex] = max(maxima[localInvocationIndex], maxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && maxima[0] > 0.0) {
    atomicMax(&statistics[statisticsOffset + slot], bitcast<u32>(maxima[0]));
  }`
      })
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select-init`,
        operation: OPERATION,
        variant: 'select-init',
        bindings: [
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
          {name: 'alive', view: alive, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'clusterIndices', view: clusterIndices, type: 'u32', access: 'read_write'},
          {name: 'clusterStatistics', view: clusterStatistics, type: 'f32', access: 'read_write'},
          ...(zoneStatistics
            ? [
                {
                  name: 'zoneStatistics',
                  view: zoneStatistics,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: Math.max(zones, maximumClusters, 8),
        declarations: `${common}
const CLUSTERS: u32 = ${maximumClusters}u;`,
        body: `if (index < ZONES) {
    let best = bitcast<f32>(centerBest[centerBestOffset + index * CENTER_WORDS]);
    alive[aliveOffset + index] = select(0u, 1u, best > 0.0);
    ${zoneStatistics ? 'zoneStatistics[zoneStatisticsOffset + index] = best;' : ''}
  }
  if (index < 8u) {
    state[stateOffset + index] = 0u;
  }
  if (index < CLUSTERS) {
    for (var word = 0u; word < ${GPU_SCAN_STATISTIC_INDEX_WORDS}u; word++) {
      clusterIndices[clusterIndicesOffset + index * ${GPU_SCAN_STATISTIC_INDEX_WORDS}u + word] = 0u;
    }
    for (var word = 0u; word < ${GPU_SCAN_STATISTIC_STATISTIC_WORDS}u; word++) {
      clusterStatistics[clusterStatisticsOffset + index * ${GPU_SCAN_STATISTIC_STATISTIC_WORDS}u + word] = 0.0;
    }
  }`
      })
    );

    for (let round = 0; round < maximumClusters; round++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-round-mask-${round}`,
          operation: OPERATION,
          variant: 'round-mask',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
            {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
            {name: 'alive', view: alive, type: 'u32', access: 'read_write'}
          ],
          invocationCount: zones,
          declarations: common,
          // A cluster blocks every later window sharing a zone with it.
          body: `if (state[stateOffset + 3u] == 0u || alive[aliveOffset + index] == 0u) {
    return;
  }
  let selectedCenter = state[stateOffset + 1u];
  let selectedZones = state[stateOffset + 2u];
  let ownZones = centerBest[centerBestOffset + index * CENTER_WORDS + 1u];
  for (var own = 0u; own < ownZones; own++) {
    let zone = neighbors[neighborsOffset + index * LIST + own];
    for (var other = 0u; other < selectedZones; other++) {
      if (neighbors[neighborsOffset + selectedCenter * LIST + other] == zone) {
        alive[aliveOffset + index] = 0u;
        return;
      }
    }
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-round-partials-${round}`,
          operation: OPERATION,
          variant: 'round-partials',
          bindings: [
            {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
            {name: 'alive', view: alive, type: 'u32', access: 'read'},
            {name: 'partials', view: partials, type: 'u32', access: 'read_write'}
          ],
          invocationCount: zoneBlocks,
          declarations: common,
          body: `var best = 0.0;
  var bestCenter = NO_CENTER;
  let first = index * ZONES_PER_BLOCK;
  for (var center = first; center < min(first + ZONES_PER_BLOCK, ZONES); center++) {
    let ratio = bitcast<f32>(centerBest[centerBestOffset + center * CENTER_WORDS]);
    if (alive[aliveOffset + center] == 1u && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  partials[partialsOffset + index * 2u] = bitcast<u32>(best);
  partials[partialsOffset + index * 2u + 1u] = bestCenter;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-round-select-${round}`,
          operation: OPERATION,
          variant: 'round-select',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'partials', view: partials, type: 'u32', access: 'read'},
            {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
            {name: 'distances', view: distances, type: 'f32', access: 'read'},
            {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'clusterIndices', view: clusterIndices, type: 'u32', access: 'read_write'},
            {name: 'clusterStatistics', view: clusterStatistics, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: withParameters,
          body: `if (state[stateOffset + 4u] == 1u) {
    return;
  }
  var best = 0.0;
  var bestCenter = NO_CENTER;
  for (var block = 0u; block < ZONE_BLOCKS; block++) {
    let ratio = bitcast<f32>(partials[partialsOffset + block * 2u]);
    let center = partials[partialsOffset + block * 2u + 1u];
    if (center != NO_CENTER && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  if (bestCenter == NO_CENTER) {
    state[stateOffset + 3u] = 0u;
    state[stateOffset + 4u] = 1u;
    return;
  }
  let count = state[stateOffset];
  let record = centerBestOffset + bestCenter * CENTER_WORDS;
  let zoneCount = centerBest[record + 1u];
  let observed = bitcast<f32>(centerBest[record + 4u]);
  let expected = bitcast<f32>(centerBest[record + 5u]);
  var exceeding = 0u;
  let permutations = readPermutations();
  for (var replicate = 1u; replicate <= permutations; replicate++) {
    if (statistics[statisticsOffset + replicate] >= best) {
      exceeding++;
    }
  }
  clusterIndices[clusterIndicesOffset + count * ${GPU_SCAN_STATISTIC_INDEX_WORDS}u] = bestCenter;
  clusterIndices[clusterIndicesOffset + count * ${GPU_SCAN_STATISTIC_INDEX_WORDS}u + 1u] = zoneCount;
  clusterIndices[clusterIndicesOffset + count * ${GPU_SCAN_STATISTIC_INDEX_WORDS}u + 2u] = centerBest[record + 2u];
  clusterIndices[clusterIndicesOffset + count * ${GPU_SCAN_STATISTIC_INDEX_WORDS}u + 3u] = centerBest[record + 3u];
  let statisticStart = clusterStatisticsOffset + count * ${GPU_SCAN_STATISTIC_STATISTIC_WORDS}u;
  clusterStatistics[statisticStart] = best;
  clusterStatistics[statisticStart + 1u] = observed;
  clusterStatistics[statisticStart + 2u] = expected;
  clusterStatistics[statisticStart + 3u] = f32(exceeding + 1u) / f32(permutations + 1u);
  clusterStatistics[statisticStart + 4u] = sqrt(distances[distancesOffset + bestCenter * LIST + zoneCount - 1u]);
  clusterStatistics[statisticStart + 5u] = observed / expected;
  state[stateOffset] = count + 1u;
  state[stateOffset + 1u] = bestCenter;
  state[stateOffset + 2u] = zoneCount;
  state[stateOffset + 3u] = 1u;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: withParameters,
        body: `summary[summaryOffset] = state[stateOffset];
  summary[summaryOffset + 1u] = totals[totalsOffset];
  summary[summaryOffset + 2u] = readPermutations();
  summary[summaryOffset + 3u] = 0u;`
      })
    );
    return nodes;
  }
}
