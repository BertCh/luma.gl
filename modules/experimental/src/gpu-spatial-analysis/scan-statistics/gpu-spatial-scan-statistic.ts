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
import type {GPUInferenceMetadata} from '../contracts/index';
import {
  getGPUSpatialScanLayout,
  GPU_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SCAN_STATISTIC_MAXIMUM_CLUSTERS,
  GPU_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS,
  GPU_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS,
  GPU_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SCAN_STATISTIC_SUMMARY_LENGTH,
  type GPUSpatialScanModel,
  type GPUSpatialScanStatisticParameters
} from './scan-statistic-parameters';

const OPERATION = 'GPUSpatialScanStatistic';
/** Fixed-order reduction blocks (at most 256 invocations walk the cells serially). */
const MAXIMUM_BLOCKS = 256;
/** Case-draw blocks per replicate: each owns a Philox stream, so results ignore the dispatch. */
const CASE_BLOCKS = 1024;
/** Philox counter tag of the multinomial case draws, apart from the permutation streams. */
const SCAN_STREAM = 0x5ca75747;
/** Zones per pass of the neighbor selection: each pass keeps the next `TILE` nearest zones. */
const NEIGHBOR_TILE = 32;
/** Words of one per-center record: LLR bits, zones, first bucket, last bucket, cases, expected, truncated. */
const CENTER_WORDS = 8;
/** Invocations of the evaluate workgroups: each reduces the window maxima of 256 centers of one replicate. */
const EVALUATE_WORKGROUP_SIZE = 256;
/** Largest replicate count buffer, in words (the default storage binding limit is 128 MiB). */
const MAXIMUM_REPLICATE_WORDS = 2 ** 25;

/**
 * Properties for {@link GPUSpatialScanStatistic}.
 *
 * Per-frame (no recompile): the contents of `cases`, `baseline`, `controls`, `positions` and
 * `parameters` (seed, replicate count, window limits, window shape, rate direction). Compile-time:
 * the zone count, `model`, `timeBuckets`, `maximumWindowZones`, `maximumPermutations` and
 * `maximumClusters`.
 */
export type GPUSpatialScanStatisticProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'scan-statistic'`. */
  id?: string;
  /** Zone centroids in planar coordinates, finite, one row per zone. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Probability model, compile-time. `'poisson'` (default): `baseline` is required. `'bernoulli'`:
   * `controls` is required, `baseline` is ignored. `'space-time-permutation'`: neither is used, the
   * expectation comes from the margins of `cases` (needs `timeBuckets > 1` to differ from a
   * purely spatial Poisson scan with population-proportional cases).
   */
  model?: GPUSpatialScanModel;
  /**
   * Observed case counts per (zone, time bucket), indexed `zone * timeBuckets + bucket`, the
   * layout of `GPUTemporalReduction` counts. The total must be below 2^32.
   */
  cases: GraphDataView<'uint32'>;
  /**
   * Poisson model: baseline per (zone, time bucket) in the same layout: population at risk or
   * expected counts, finite and non-negative, with any total scale. The null model is a Poisson
   * process with `E[cases_i] = baseline_i * C / sum(baseline)`, `C` the total case count.
   */
  baseline?: GraphDataView<'float32'>;
  /**
   * Bernoulli model: controls (non-cases) per (zone, time bucket) in the same layout. The
   * population of a cell is `cases + controls`; the total population must be below 2^32, and
   * window populations are accumulated in f32 (exact below 2^24).
   */
  controls?: GraphDataView<'uint32'>;
  /** Time buckets per zone: 1 for a purely spatial scan. Compile-time, at most 32. */
  timeBuckets?: number;
  /**
   * Compile-time capacity of the per-center neighbor tables: windows hold at most this many of the
   * center's nearest zones. The default is every zone while `zones * zones <= 2^24` (4096 zones),
   * so the population fraction governs the window size as in SaTScan; for more zones it is the
   * largest capacity that fits in the table budget. A center whose windows reach the capacity
   * while the population fraction would allow larger ones is counted in `summary.overflow` and
   * flags its clusters with `truncatedByZoneCap`. The per-frame parameter may lower it.
   * Selecting the neighbors costs about `zones^2 * capacity / 32` operations per frame.
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
 * Kulldorff's spatial and space-time scan statistic (SaTScan; CARTO
 * `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or points, with the Poisson,
 * Bernoulli (Kulldorff 1997) or space-time permutation (Kulldorff et al. 2005) model.
 *
 * Windows: for every center zone, the `k` nearest zones for every `k` (a circle: with the default
 * shape a window never splits zones at equal distance; ties are exact f32 squared-distance
 * equality), kept while they hold at most `maximumPopulationFraction` (default 0.5) of the total
 * population (the baseline, the cases plus controls, or, for the space-time permutation model, the
 * expected cases). The population share governs the size of a window: the per-center neighbor
 * tables hold up to `maximumWindowZones` zones, every zone by default, and a window that the
 * capacity cuts short is reported (`summary.overflow`, `truncatedByZoneCap`). With `timeBuckets > 1`
 * each window is a cylinder over every run of at most `maximumTimeBuckets` consecutive buckets,
 * 50% of the study period by default.
 *
 * Statistic: for a window with `c` cases and null expectation `e` (total cases `C`), the Poisson
 * log-likelihood ratio `c ln(c/e) + (C - c) ln((C - c)/(C - e))`. Bernoulli: with window
 * population `n` of `N`, the sum of `o ln(o/e)` over the four cells (cases, controls inside and
 * outside), `e = n C/N`, which equals Kulldorff's `c ln(c/n) + ... - [C ln(C/N) + ...]`. Space-time
 * permutation: the Poisson form with `e = (sum of the zone totals of the window) * (sum of the
 * bucket totals of the window) / C`, so a purely temporal trend or seasonal cycle shared by every
 * zone is not a cluster. The `rateDirection` parameter selects windows with `c > e` (`'high'`,
 * default), `c < e` (`'low'`) or both; other windows score 0. Windows with no population are
 * skipped. Evaluated in f32; the terms cancel for `c` close to `e`, so LLRs below about 1e-3 are
 * approximate.
 *
 * Clusters: the most likely cluster is the window with the largest LLR. Secondary clusters are
 * found greedily: the best window sharing no zone with any earlier cluster (time is ignored for
 * overlap), until `maximumClusters` or no positive LLR remains. Ties pick the lowest center zone.
 *
 * Inference: `P` Monte Carlo replicates redistribute the cases and record the largest LLR of
 * each replicate (Philox 4x32-10 streams from `permutation-inference`). Poisson: a multinomial over
 * the zone-buckets with probabilities proportional to the baseline (each case an independent
 * draw from an integer cumulative table). Bernoulli: the case labels are permuted over the fixed
 * population of every cell (conditioning on the total cases), by one invocation per replicate
 * drawing individuals without replacement. Space-time permutation: the case times are permuted
 * over the case locations, so the zone totals and the bucket totals are fixed, again one
 * invocation per replicate (cost about `C * timeBuckets` per replicate). The Bernoulli and
 * space-time permutation draws are sequential per replicate: about `C` draws (at most half the
 * population for Bernoulli). A cluster's p-value is `(1 + #{replicate maxima >= LLR}) /
 * (P + 1)`; secondary clusters use the same maxima distribution, as SaTScan does.
 *
 * Reductions run in fixed order: integer atomic adds for the draws, serial block sums, and
 * argmax with the lowest index winning ties. Results are a pure function of the inputs and seed.
 * Capacity: one `(P + 1) * zones * timeBuckets` word replicate table (at most 2^25 words), two
 * neighbor tables of `zones * (maximumWindowZones + 1)` words (at most 2^24 each) and
 * `(P + 1) * zones` window maxima; work is about `(P + 1) * zones * windowZones *
 * timeBuckets^2`, `windowZones` the zones of the largest window.
 *
 * P-values come from the Monte Carlo maximum over every window, so they already account for the
 * search over windows; no further multiple-testing correction applies (see
 * {@link GPUSpatialScanStatistic.getInferenceMetadata}).
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
  /** Probability model. */
  readonly model: GPUSpatialScanModel;
  /** Largest window in zones (the compile-time capacity). */
  readonly usableZones: number;
  /** Rows per center in the neighbor tables: `usableZones` plus one look-ahead row when truncating. */
  readonly listLength: number;
  /** Cluster capacity. */
  readonly maximumClusters: number;

  constructor(props: GPUSpatialScanStatisticProps) {
    const id = props.id ?? 'scan-statistic';
    this.id = id;
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedUint32View(props.cases, `${id} cases`);
    this.model = props.model ?? 'poisson';
    if (!['poisson', 'bernoulli', 'space-time-permutation'].includes(this.model)) {
      throw new Error(`${id} model must be 'poisson', 'bernoulli' or 'space-time-permutation'`);
    }
    if (this.model === 'poisson') {
      if (!props.baseline) throw new Error(`${id} the poisson model needs a baseline`);
      validatePackedView(props.baseline, ['float32'], `${id} baseline`);
    }
    if (this.model === 'bernoulli') {
      if (!props.controls) throw new Error(`${id} the bernoulli model needs controls`);
      validatePackedUint32View(props.controls, `${id} controls`);
    }
    validatePackedUint32View(props.parameters, `${id} parameters`);
    validatePackedUint32View(props.clusterIndices, `${id} clusterIndices`);
    validatePackedView(props.clusterStatistics, ['float32'], `${id} clusterStatistics`);
    validatePackedView(props.statistics, ['float32'], `${id} statistics`);
    validatePackedUint32View(props.summary, `${id} summary`);
    this.zones = props.positions.length;
    this.timeBuckets = props.timeBuckets ?? 1;
    this.maximumClusters = props.maximumClusters ?? 8;
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
    const layout = getGPUSpatialScanLayout(this.zones, props.maximumWindowZones);
    this.usableZones = layout.usableZones;
    this.listLength = layout.listLength;
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
    if (
      props.cases.length !== cells ||
      (this.model === 'poisson' && props.baseline!.length !== cells) ||
      (this.model === 'bernoulli' && props.controls!.length !== cells)
    ) {
      throw new Error(`${id} cases and baseline or controls must hold zones * timeBuckets rows`);
    }
    if ((maximumPermutations + 1) * cells > MAXIMUM_REPLICATE_WORDS) {
      throw new Error(
        `${id} (maximumPermutations + 1) * zones * timeBuckets must not exceed ${MAXIMUM_REPLICATE_WORDS}`
      );
    }
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
      ...(props.baseline ? [props.baseline] : []),
      ...(props.controls ? [props.controls] : []),
      props.parameters
    ]);
    if (new Set(outputs.map(output => output.buffer)).size !== outputs.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * CPU-side description of how the p-values are produced, for legends. P-values are Monte Carlo
   * maximum-statistic: each is `(1 + #{replicate maxima >= LLR}) / (P + 1)` against the maximum of
   * the LLR over every window, so the search over windows is already accounted for and
   * `multipleTesting` is `'none'`.
   *
   * @param parameters The per-frame parameters; `rateDirection` selects the alternative
   * (`'high'`: `'greater'`, `'low'`: `'lesser'`, `'both'`: `'two-sided'`; windows of both
   * directions compete in one maximum) and `permutations` the replicate count. Without them the
   * metadata describes the default high-rate scan with the compile-time `maximumPermutations`
   * (the per-frame count may be lower).
   */
  getInferenceMetadata(
    parameters?: Pick<GPUSpatialScanStatisticParameters, 'seed' | 'permutations' | 'rateDirection'>
  ): GPUInferenceMetadata {
    const direction = parameters?.rateDirection ?? 'high';
    return {
      alternative: direction === 'high' ? 'greater' : direction === 'low' ? 'lesser' : 'two-sided',
      permutationCount: parameters?.permutations ?? this.props.maximumPermutations,
      ...(parameters ? {seed: parameters.seed} : {}),
      multipleTesting: 'none',
      testedRowsDefinition:
        'every window: a center zone, its nearest zones within maximumPopulationFraction of the population, and a run of consecutive time buckets within maximumTimeBuckets; p-values are Monte Carlo maxima over all windows'
    };
  }

  /**
   * Returns the nodes in order: `nearest`, the model preparation (`population` for Bernoulli,
   * `margins` and `weights` for the space-time permutation model), `block-totals`, `totals`, for
   * Poisson and Bernoulli the cumulative table (`weight-sums`, `weight-scan`, `cdf`),
   * `init-counts`, the replicate draws (`scatter`, `draw-labels` or `draw-times`), `window-limit`,
   * `evaluate`, `select-init`, then per cluster `round-members`, `round-mask`, `round-partials` and
   * `round-select`, and `summary`. `evaluate` reduces the per-replicate maxima itself (workgroup
   * max, one `atomicMax` per workgroup), so there is no per-center candidate table and no separate
   * maximum pass.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zones, timeBuckets, listLength, usableZones, maximumClusters, model} = this;
    const {positions, cases, parameters, maximumPermutations} = props;
    const {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      cases,
      ...(props.baseline ? [props.baseline] : []),
      ...(props.controls ? [props.controls] : []),
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
    const members = u32('members', zones);
    // Per-cell population of the window sums: the Poisson baseline, cases + controls (Bernoulli) or
    // the expectation z_s * m_b / C of the space-time permutation model.
    const weights: GraphDataView<'float32'> =
      model === 'poisson' ? props.baseline! : f32('weights', cells);
    const population = model === 'bernoulli' ? u32('population', cells) : null;
    const zoneTotals = model === 'space-time-permutation' ? u32('zone-totals', zones) : null;
    const bucketTotals =
      model === 'space-time-permutation' ? u32('bucket-totals', timeBuckets) : null;
    const blockBaseline = f32('block-baseline', blocks);
    const blockCases = u32('block-cases', blocks);
    const blockWeights = u32('block-weights', blocks);
    const blockOffsets = u32('block-offsets', blocks);
    const totals = u32('totals', 4);
    const cumulative = u32('cumulative', cells);
    const counts = u32('counts', slotCount * cells);
    const evaluateGroups = Math.ceil(zones / EVALUATE_WORKGROUP_SIZE);
    // Per-center best-window records, then (from WINDOW_INFO_START) two words per center written by
    // `window-limit` for `evaluate`: the window slot count and the capacity-truncation flag. They
    // share one buffer to keep `evaluate` within the 8 storage buffers per stage of core devices.
    const centerBest = u32('center-best', zones * CENTER_WORDS + zones * 2);
    const alive = u32('alive', zones);
    const partials = u32('partials', zoneBlocks * 2);
    const state = u32('state', 8);

    const common = /* wgsl */ `
const ZONES: u32 = ${zones}u;
const LIST: u32 = ${listLength}u;
const USABLE: u32 = ${usableZones}u;
const BUCKETS: u32 = ${timeBuckets}u;
const CELLS: u32 = ${cells}u;
const BLOCKS: u32 = ${blocks}u;
const CELLS_PER_BLOCK: u32 = ${cellsPerBlock}u;
const ZONE_BLOCKS: u32 = ${zoneBlocks}u;
const ZONES_PER_BLOCK: u32 = ${zonesPerBlock}u;
const MAXIMUM_PERMUTATIONS: u32 = ${maximumPermutations}u;
const CENTER_WORDS: u32 = ${CENTER_WORDS}u;
const WINDOW_INFO_START: u32 = ${zones * CENTER_WORDS}u;
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
  return clamp(parameters[parametersOffset + 4u], 1u, USABLE);
}
// 0 selects the default, 50% of the study period.
fn readMaximumBuckets() -> u32 {
  let requested = parameters[parametersOffset + 5u];
  if (requested == 0u) {
    return max(1u, BUCKETS / 2u);
  }
  return clamp(requested, 1u, BUCKETS);
}
fn readWindowShape() -> u32 {
  return parameters[parametersOffset + 6u];
}
fn readRateDirection() -> u32 {
  return parameters[parametersOffset + 7u];
}`;
    const withParameters = `${common}${readers}`;
    const direction = /* wgsl */ `
fn getObservedTerm(observed: f32, expected: f32) -> f32 {
  if (observed > 0.0 && expected > 0.0) {
    return observed * log(observed / expected);
  }
  return 0.0;
}
fn isDirected(observed: f32, expected: f32, direction: u32) -> bool {
  if (direction == 0u) {
    return observed > expected;
  }
  if (direction == 1u) {
    return observed < expected;
  }
  return observed != expected;
}`;
    const likelihood =
      model === 'bernoulli'
        ? /* wgsl */ `${direction}
// Kulldorff (1997) Bernoulli LLR as the sum of o ln(o/e) over the 2x2 table of
// (cases, controls) x (inside, outside), with e = row total * column total / N.
fn getLogLikelihoodRatio(observed: f32, windowWeight: f32, totalCases: f32, totalWeight: f32, direction: u32) -> f32 {
  let outside = totalWeight - windowWeight;
  if (!(windowWeight > 0.0) || !(outside > 0.0)) {
    return 0.0;
  }
  let expectedInside = windowWeight * totalCases / totalWeight;
  if (!isDirected(observed, expectedInside, direction)) {
    return 0.0;
  }
  let totalControls = totalWeight - totalCases;
  let ratio = getObservedTerm(observed, expectedInside) +
    getObservedTerm(max(windowWeight - observed, 0.0), windowWeight * totalControls / totalWeight) +
    getObservedTerm(totalCases - observed, outside * totalCases / totalWeight) +
    getObservedTerm(max(totalControls - windowWeight + observed, 0.0), outside * totalControls / totalWeight);
  return max(ratio, 0.0);
}`
        : /* wgsl */ `${direction}
// Poisson LLR c ln(c/e) + (C - c) ln((C - c)/(C - e)); the space-time permutation model passes the
// window's e = (zone totals * bucket totals / C) as the window weight of a total weight C.
fn getLogLikelihoodRatio(observed: f32, windowWeight: f32, totalCases: f32, totalWeight: f32, direction: u32) -> f32 {
  let expected = windowWeight * totalCases / totalWeight;
  if (!(expected > 0.0) || !isDirected(observed, expected, direction)) {
    return 0.0;
  }
  var ratio = getObservedTerm(observed, expected);
  let remaining = totalCases - observed;
  if (remaining > 0.0 && totalCases - expected > 0.0) {
    ratio += remaining * log(remaining / (totalCases - expected));
  }
  return max(ratio, 0.0);
}`;
    const cellWeight = (cell: string) =>
      model === 'bernoulli'
        ? `population[populationOffset + ${cell}]`
        : `getCellWeight(baseline[baselineOffset + ${cell}], totalBaseline)`;
    const cellWeightBinding = population
      ? {name: 'population', view: population, type: 'u32' as const, access: 'read' as const}
      : {name: 'baseline', view: weights, type: 'f32' as const, access: 'read' as const};

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
const LIST: u32 = ${listLength}u;
const TILE: u32 = ${NEIGHBOR_TILE}u;`,
        // The sorted list is selected TILE zones per pass: each pass scans every zone for the next
        // TILE entries after the previous tile in (distance, zone index) order, so ties are broken
        // by ascending zone index and the private arrays stay small.
        body: `var tileIndex: array<u32, TILE>;
  var tileDistance: array<f32, TILE>;
  var lastDistance = 0.0;
  var lastZone = 0u;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  for (var base = 0u; base < LIST; base += TILE) {
    let wanted = min(TILE, LIST - base);
    var filled = 0u;
    for (var zone = 0u; zone < ZONES; zone++) {
      let dx = positions[positionsOffset + zone * 2u] - centerX;
      let dy = positions[positionsOffset + zone * 2u + 1u] - centerY;
      let distance = dx * dx + dy * dy;
      if (base > 0u && (distance < lastDistance || (distance == lastDistance && zone <= lastZone))) {
        continue;
      }
      if (filled < wanted || distance < tileDistance[wanted - 1u]) {
        var slot = select(wanted - 1u, filled, filled < wanted);
        filled = min(filled + 1u, wanted);
        while (slot > 0u && tileDistance[slot - 1u] > distance) {
          tileDistance[slot] = tileDistance[slot - 1u];
          tileIndex[slot] = tileIndex[slot - 1u];
          slot--;
        }
        tileDistance[slot] = distance;
        tileIndex[slot] = zone;
      }
    }
    for (var slot = 0u; slot < wanted; slot++) {
      neighbors[neighborsOffset + index * LIST + base + slot] = tileIndex[slot];
      distances[distancesOffset + index * LIST + base + slot] = tileDistance[slot];
    }
    lastDistance = tileDistance[wanted - 1u];
    lastZone = tileIndex[wanted - 1u];
  }`
      })
    ];

    if (model === 'bernoulli') {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-population`,
          operation: OPERATION,
          variant: 'population',
          bindings: [
            {name: 'cases', view: cases, type: 'u32', access: 'read'},
            {name: 'controls', view: props.controls!, type: 'u32', access: 'read'},
            {name: 'population', view: population!, type: 'u32', access: 'read_write'},
            {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cells,
          declarations: common,
          body: `let cellPopulation = cases[casesOffset + index] + controls[controlsOffset + index];
  population[populationOffset + index] = cellPopulation;
  weights[weightsOffset + index] = f32(cellPopulation);`
        })
      );
    }
    if (model === 'space-time-permutation') {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-margins`,
          operation: OPERATION,
          variant: 'margins',
          bindings: [
            {name: 'cases', view: cases, type: 'u32', access: 'read'},
            {name: 'zoneTotals', view: zoneTotals!, type: 'u32', access: 'read_write'},
            {name: 'bucketTotals', view: bucketTotals!, type: 'u32', access: 'read_write'}
          ],
          invocationCount: Math.max(zones, timeBuckets),
          declarations: common,
          body: `if (index < ZONES) {
    var zoneSum = 0u;
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      zoneSum += cases[casesOffset + index * BUCKETS + bucket];
    }
    zoneTotals[zoneTotalsOffset + index] = zoneSum;
  }
  if (index < BUCKETS) {
    var bucketSum = 0u;
    for (var zone = 0u; zone < ZONES; zone++) {
      bucketSum += cases[casesOffset + zone * BUCKETS + index];
    }
    bucketTotals[bucketTotalsOffset + index] = bucketSum;
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-weights`,
          operation: OPERATION,
          variant: 'weights',
          bindings: [
            {name: 'zoneTotals', view: zoneTotals!, type: 'u32', access: 'read'},
            {name: 'bucketTotals', view: bucketTotals!, type: 'u32', access: 'read'},
            {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cells,
          declarations: common,
          // Expected cases of the cell under fixed margins: zone total * bucket total / C.
          body: `let zone = index / BUCKETS;
  let bucket = index - zone * BUCKETS;
  var total = 0u;
  for (var other = 0u; other < BUCKETS; other++) {
    total += bucketTotals[bucketTotalsOffset + other];
  }
  weights[weightsOffset + index] = select(
    0.0,
    f32(zoneTotals[zoneTotalsOffset + zone]) * f32(bucketTotals[bucketTotalsOffset + bucket]) / f32(total),
    total > 0u
  );`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-totals`,
        operation: OPERATION,
        variant: 'block-totals',
        bindings: [
          {name: 'baseline', view: weights, type: 'f32', access: 'read'},
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
      })
    );

    if (model !== 'space-time-permutation') {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-weight-sums`,
          operation: OPERATION,
          variant: 'weight-sums',
          bindings: [
            cellWeightBinding,
            {name: 'totals', view: totals, type: 'u32', access: 'read'},
            {name: 'blockWeights', view: blockWeights, type: 'u32', access: 'read_write'}
          ],
          invocationCount: blocks,
          declarations: common,
          body: `let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  var weightSum = 0u;
  let first = index * CELLS_PER_BLOCK;
  for (var cell = first; cell < min(first + CELLS_PER_BLOCK, CELLS); cell++) {
    weightSum += ${cellWeight('cell')};
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
            cellWeightBinding,
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
    running += ${cellWeight('cell')};
    cumulative[cumulativeOffset + cell] = running;
  }`
        })
      );
    }

    // Slot 0 of the count table is the observed data, slots 1..P the replicates.
    nodes.push(
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
      })
    );

    if (model === 'poisson') {
      nodes.push(
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
        })
      );
    } else if (model === 'bernoulli') {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-draw-labels`,
          operation: OPERATION,
          variant: 'draw-labels',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read'},
            {name: 'cumulative', view: cumulative, type: 'u32', access: 'read'},
            {name: 'population', view: population!, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: maximumPermutations,
          declarations: `${PERMUTATION_RANDOM_WGSL}
${withParameters}
const SCAN_STREAM: u32 = ${SCAN_STREAM}u;`,
          // One replicate per invocation: a random permutation of the case labels over the fixed
          // population is a uniform draw of C individuals without replacement. A draw picks a cell
          // in proportion to its population and accepts it with the share of its individuals not
          // yet chosen, which makes the cell probabilities proportional to the remaining
          // individuals (multivariate hypergeometric). When more than half the individuals are
          // cases the controls are drawn instead and the counts complemented, so at least half of
          // the draws are accepted.
          body: `let replicate = index;
  let totalCases = totals[totalsOffset];
  let totalPopulation = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalPopulation == 0u) {
    return;
  }
  let complement = totalCases > totalPopulation - totalCases;
  let drawCount = select(totalCases, totalPopulation - totalCases, complement);
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (replicate + 1u) * CELLS;
  var drawn = 0u;
  while (drawn < drawCount) {
    let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), totalPopulation);
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
    let cellPopulation = population[populationOffset + low];
    let unchosen = cellPopulation - counts[row + low];
    if (getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), cellPopulation) < unchosen) {
      counts[row + low] += 1u;
      drawn++;
    }
  }
  if (complement) {
    for (var cell = 0u; cell < CELLS; cell++) {
      counts[row + cell] = population[populationOffset + cell] - counts[row + cell];
    }
  }`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-draw-times`,
          operation: OPERATION,
          variant: 'draw-times',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read'},
            {name: 'zoneTotals', view: zoneTotals!, type: 'u32', access: 'read'},
            {name: 'bucketTotals', view: bucketTotals!, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: maximumPermutations,
          declarations: `${PERMUTATION_RANDOM_WGSL}
${withParameters}
const SCAN_STREAM: u32 = ${SCAN_STREAM}u;`,
          // One replicate per invocation: the case times are permuted over the case locations. Each
          // zone in index order takes its cases' times without replacement from the times left, so
          // both margins stay fixed and the table is a multivariate hypergeometric draw.
          body: `let replicate = index;
  if (replicate >= readPermutations()) {
    return;
  }
  var remaining: array<u32, BUCKETS>;
  for (var bucket = 0u; bucket < BUCKETS; bucket++) {
    remaining[bucket] = bucketTotals[bucketTotalsOffset + bucket];
  }
  var left = totals[totalsOffset];
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (replicate + 1u) * CELLS;
  for (var zone = 0u; zone < ZONES; zone++) {
    let zoneCases = zoneTotals[zoneTotalsOffset + zone];
    for (var draw = 0u; draw < zoneCases; draw++) {
      let drawTarget = getPhiloxMultiplyHigh(nextPhiloxUint32(&stream), left);
      var accumulated = 0u;
      var chosen = BUCKETS - 1u;
      for (var bucket = 0u; bucket < BUCKETS; bucket++) {
        accumulated += remaining[bucket];
        if (drawTarget < accumulated) {
          chosen = bucket;
          break;
        }
      }
      counts[row + zone * BUCKETS + chosen] += 1u;
      remaining[chosen] -= 1u;
      left -= 1u;
    }
  }`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-window-limit`,
        operation: OPERATION,
        variant: 'window-limit',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'baseline', view: weights, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read_write'}
        ],
        invocationCount: zones,
        declarations: withParameters,
        // Word 0: the number of nearest zones whose cumulative population stays within the limit
        // (the population share governs; the per-frame zone limit may lower it). Word 1: 1 when the
        // compile-time capacity ended the list while the next zone would still have fit.
        body: `let totalBaseline = bitcast<f32>(totals[totalsOffset + 1u]);
  let limit = readMaximumFraction() * totalBaseline;
  let maximumZones = readMaximumZones();
  var populated = 0.0;
  var slots = 0u;
  var exceeded = false;
  for (var slot = 0u; slot < maximumZones; slot++) {
    let zone = neighbors[neighborsOffset + index * LIST + slot];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    if (populated > limit) {
      exceeded = true;
      break;
    }
    slots = slot + 1u;
  }
  var truncated = 0u;
  if (!exceeded && USABLE < ZONES && slots == USABLE && maximumZones == USABLE) {
    let zone = neighbors[neighborsOffset + index * LIST + USABLE];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      populated += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    truncated = select(0u, 1u, populated <= limit);
  }
  centerBest[centerBestOffset + WINDOW_INFO_START + index * 2u] = slots;
  centerBest[centerBestOffset + WINDOW_INFO_START + index * 2u + 1u] = truncated;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-evaluate`,
        operation: OPERATION,
        variant: 'evaluate',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'baseline', view: weights, type: 'f32', access: 'read'},
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
  let direction = readRateDirection();
  let nearestShape = readWindowShape() == 1u;
  let windowSlots = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u];
  let slotStart = slot * CELLS;
  var caseSums: array<u32, BUCKETS>;
  var baselineSums: array<f32, BUCKETS>;
  var bestZones = 0u;
  var bestFirst = 0u;
  var bestLast = 0u;
  var bestCases = 0.0;
  var bestExpected = 0.0;
  for (var slotIndex = 0u; slotIndex < windowSlots; slotIndex++) {
    let zone = neighbors[neighborsOffset + center * LIST + slotIndex];
    for (var bucket = 0u; bucket < BUCKETS; bucket++) {
      caseSums[bucket] += counts[countsOffset + slotStart + zone * BUCKETS + bucket];
      baselineSums[bucket] += baseline[baselineOffset + zone * BUCKETS + bucket];
    }
    // A circle never ends between two zones at the same distance.
    let separated = nearestShape || slotIndex + 1u >= ZONES ||
      distances[distancesOffset + center * LIST + slotIndex] < distances[distancesOffset + center * LIST + slotIndex + 1u];
    if (!separated) {
      continue;
    }
    for (var first = 0u; first < BUCKETS; first++) {
      var windowCases = 0u;
      var windowBaseline = 0.0;
      for (var last = first; last < min(first + maximumBuckets, BUCKETS); last++) {
        windowCases += caseSums[last];
        windowBaseline += baselineSums[last];
        let ratio = getLogLikelihoodRatio(f32(windowCases), windowBaseline, totalCases, totalBaseline, direction);
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
    centerBest[record + 6u] = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u + 1u];
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
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select-init`,
        operation: OPERATION,
        variant: 'select-init',
        bindings: [
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
          {name: 'alive', view: alive, type: 'u32', access: 'read_write'},
          {name: 'members', view: members, type: 'u32', access: 'read_write'},
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
    members[membersOffset + index] = 0u;
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
          id: `${id}-round-members-${round}`,
          operation: OPERATION,
          variant: 'round-members',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
            {name: 'members', view: members, type: 'u32', access: 'read_write'}
          ],
          invocationCount: listLength,
          declarations: common,
          // Marks the zones of the cluster selected last (clusters never overlap, so the marks of
          // earlier clusters stay valid).
          body: `if (state[stateOffset + 3u] == 0u || index >= state[stateOffset + 2u]) {
    return;
  }
  let zone = neighbors[neighborsOffset + state[stateOffset + 1u] * LIST + index];
  members[membersOffset + zone] = 1u;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-round-mask-${round}`,
          operation: OPERATION,
          variant: 'round-mask',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
            {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
            {name: 'members', view: members, type: 'u32', access: 'read'},
            {name: 'alive', view: alive, type: 'u32', access: 'read_write'}
          ],
          invocationCount: zones,
          declarations: common,
          // A cluster blocks every later window sharing a zone with it.
          body: `if (state[stateOffset + 3u] == 0u || alive[aliveOffset + index] == 0u) {
    return;
  }
  let ownZones = centerBest[centerBestOffset + index * CENTER_WORDS + 1u];
  for (var own = 0u; own < ownZones; own++) {
    if (members[membersOffset + neighbors[neighborsOffset + index * LIST + own]] == 1u) {
      alive[aliveOffset + index] = 0u;
      return;
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
  clusterStatistics[statisticStart + 6u] = f32(centerBest[record + 6u]);
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
          {name: 'centerBest', view: centerBest, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: withParameters,
        body: `summary[summaryOffset] = state[stateOffset];
  summary[summaryOffset + 1u] = totals[totalsOffset];
  summary[summaryOffset + 2u] = readPermutations();
  var overflow = 0u;
  for (var center = 0u; center < ZONES; center++) {
    overflow += centerBest[centerBestOffset + center * CENTER_WORDS + 6u];
  }
  summary[summaryOffset + 3u] = overflow;`
      })
    );
    return nodes;
  }
}
