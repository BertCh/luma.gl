// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  getBoundedDispatchLayout,
  type GPUCommandGraphNodeCondition,
  type GraphBufferUse,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize
} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getSeparationWGSL,
  type InterpolationSeparation
} from '../spatial-interpolation/separation-wgsl';
import {PERMUTATION_RANDOM_WGSL} from '../permutation-inference/permutation-random';
import type {GPUInferenceMetadata} from '../../analysis-core/contracts/index';
import {
  getGPUSpatialScanStatisticLayout,
  getGPUSpatialScanStatisticParameterValues,
  GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_CLUSTERS,
  GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS,
  GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS,
  GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH,
  type GPUSpatialScanModel,
  type GPUSpatialScanStatisticSettings
} from './scan-statistic-parameters';
import {PHILOX_PURPOSES} from '../../utils/philox-random';

const OPERATION = 'GPUSpatialScanStatistic';
/** Scan windows measure planar or great-circle distances. */
const SCAN_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);
/** Fixed-order reduction blocks (at most 256 invocations walk the cells serially). */
const MAXIMUM_BLOCKS = 256;
/** Case-draw blocks per replicate: each owns a Philox stream, so results ignore the dispatch. */
const CASE_BLOCKS = 1024;
/** Philox counter tag of the multinomial case draws, apart from the permutation streams. */
const SCAN_STREAM = PHILOX_PURPOSES.spatialScanStatistic.id;
/** Bins of one 8-bit digit of the neighbor radix select and sort (a 1 KiB private histogram). */
const NEIGHBOR_RADIX_BINS = 256;
/**
 * Words of one per-center record: LLR bits, zones, first bucket, last bucket, cases, expected,
 * truncated, squared radius bits in the spatial context's squared length unit.
 */
const CENTER_WORDS = 8;
/** Invocations of the evaluate workgroups: each reduces the window maxima of 256 centers of one replicate. */
const EVALUATE_WORKGROUP_SIZE = 256;
/**
 * Largest batch count table, in words: the observed slot plus one batch of replicates (the default
 * storage binding limit is 128 MiB).
 */
const MAXIMUM_REPLICATE_WORDS = 2 ** 25;
/** Largest replicates per batch: more adds no occupancy and only grows the count table. */
const MAXIMUM_BATCH_REPLICATES = 4096;
/** Largest unrolled batch count: each batch adds four graph nodes. */
const MAXIMUM_BATCHES = 1024;
/** Words of the `totals` record: cases, baseline bits, weight total, next batch base, batch base. */
const TOTALS_WORDS = 5;

/** Caller-owned outputs of {@link GPUSpatialScanStatistic}. */
export type GPUSpatialScanStatisticOutput = {
  /**
   * Caller-owned cluster geometry, `maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS` uint32
   * words, laid out as `GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX`. Records past `clusterCount` are 0.
   */
  clusterIndices: GraphDataView<'uint32'>;
  /**
   * Caller-owned cluster statistics, `maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS`
   * float32 words, laid out as `GPU_SPATIAL_SCAN_STATISTIC_CLUSTER`.
   */
  clusterStatistics: GraphDataView<'float32'>;
  /**
   * Caller-owned float32, `maximumPermutations + 1` rows: element 0 is the largest observed LLR
   * (the most likely cluster), elements `1..P` the largest LLR of each Monte Carlo replicate
   * (rows past `P` are 0). Each batch of replicates folds its window maxima into these rows, so
   * the rows are the running maxima and no per-replicate table is kept.
   */
  statistics: GraphDataView<'float32'>;
  /** Caller-owned `GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH` uint32 words, see `GPU_SPATIAL_SCAN_STATISTIC_SUMMARY`. */
  summary: GraphDataView<'uint32'>;
  /** Optional caller-owned float32, one row per zone: the best observed LLR of windows centered there. */
  zoneStatistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialScanStatistic}.
 *
 * Per-frame (no recompile): the contents of `cases`, `baseline`, `controls`, `positions` and
 * `parameters` (seed, replicate count, window limits, window shape, rate direction). Compile-time:
 * the zone count, `model`, `timeBuckets`, `maximumWindowZones`, `maximumPermutations`,
 * `maximumClusters`, and `spatialContext`.
 */
export type GPUSpatialScanStatisticProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'scan-statistic'`. */
  id?: string;
  /** Finite zone centroids in the spatial context coordinates, one row per zone. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Omitted means native planar: Euclidean distances in position units, for example meters in a
   * `createGPULocalFrame` azimuthal-equidistant frame. Longitude/latitude degrees require
   * `metric: 'great-circle'`: distances and radii are great-circle in the context's length unit
   * (meters, or kilometers with `units: 'kilometers'`), on `sphereRadius` or the mean Earth radius.
   * Other longitude/latitude metrics throw.
   */
  spatialContext?: GPUSpatialContext;
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
   * Selecting and sorting the neighbors costs about `zones * (5 * zones + 12 * capacity)`
   * operations per frame (`zones * 13 * zones` at the default capacity): a radix select and sort
   * per center, so it no longer grows with `zones^2 * capacity`.
   */
  maximumWindowZones?: number;
  /**
   * Compile-time upper bound on the per-frame replicate count, at most 99999. Replicates run in
   * batches of `batchReplicates` (see {@link GPUSpatialScanStatistic}), so the bound no longer
   * depends on `zones * timeBuckets`, only on the unrolled batch count: at most 1024 batches.
   */
  maximumPermutations: number;
  /** Compile-time capacity of the cluster list, at most 64. Defaults to 8. */
  maximumClusters?: number;
  /**
   * Per-frame parameters: `GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH` uint32 words written with
   * `getGPUSpatialScanStatisticParameterValues(settings)`.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUSpatialScanStatisticOutput;
  /**
   * Lowers the replicates per batch, for tests of batch independence.
   * @internal
   */
  batchReplicates?: number;
};

/**
 * Kulldorff's spatial and space-time scan statistic (SaTScan; CARTO
 * `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or points, with the Poisson,
 * Bernoulli (Kulldorff 1997) or space-time permutation (Kulldorff et al. 2005) model.
 *
 * Windows: for every center zone, the `k` nearest zones for every `k` (a circle: with the default
 * shape a window never splits zones at equal distance; ties are exact f32 separation-key
 * equality: squared distance for planar, haversine term for great-circle), kept while they hold
 * at most `maximumPopulationFraction` (default 0.5) of the total population (the baseline,
 * the cases plus controls, or, for the space-time permutation model, the
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
 * skipped. Evaluated in f32, each cell term as `e ((1 + d) ln(1 + d) - d)` with `d = (o - e) / e`
 * (a series for small `d`): the linear parts `o - e` cancel exactly over the cells, so a window with
 * `c` close to `e` keeps its precision instead of losing it to two large opposite logarithm terms;
 * the absolute LLR error is about `|c - e|` times the relative rounding of `e`.
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
 * Batches: the observed data is evaluated once; the replicates run in `batchCount` batches of
 * `batchReplicates` (`R`) each. A batch draws replicates `b * R + 1 .. b * R + R` into an `R`-slot
 * count table, evaluates them and folds each replicate's largest LLR into its row of `statistics`.
 * `R` is the largest count for which `(R + 1) * zones * timeBuckets` words fit 2^25 words (the
 * 128 MiB binding limit), at most 4096 and at most `maximumPermutations`. Philox streams are keyed
 * by the global replicate index, so results do not depend on `R`; a batch whose first replicate
 * is past the per-frame `P` is disarmed on the GPU (indirect dispatch gate) and launches nothing.
 * Each batch adds four graph nodes (`batch-control`, `batch-clear`, the draw kernel and
 * `batch-evaluate`); the pipeline count does not grow with the batch count.
 *
 * Capacity: one `(R + 1) * zones * timeBuckets` word count table (at most 2^25 words), two
 * neighbor tables of `zones * (maximumWindowZones + 1)` words (at most 2^24 each) plus two sort
 * scratch tables of the same size that are only live while the neighbor tables are built, and
 * `P + 1` window maxima; work is about `(P + 1) * zones * windowZones * timeBuckets^2`,
 * `windowZones` the zones of the largest window.
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
  /** Resolved coordinate space, distance metric and length unit of the scan windows. */
  readonly spatialContext: GPUSpatialContext;
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
  /** Replicates drawn and evaluated per batch (`R`), derived from the count table budget. */
  readonly batchReplicates: number;
  /** Unrolled batches: `ceil(maximumPermutations / batchReplicates)`. */
  readonly batchCount: number;

  constructor(props: GPUSpatialScanStatisticProps) {
    const id = props.id ?? 'scan-statistic';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SCAN_SPATIAL_CONTEXT_SUPPORT
    );
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.cases, ['uint32'], `${id} cases`);
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
      validatePackedView(props.controls, ['uint32'], `${id} controls`);
    }
    validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
    const {output} = props;
    validatePackedView(output.clusterIndices, ['uint32'], `${id} output.clusterIndices`);
    validatePackedView(output.clusterStatistics, ['float32'], `${id} output.clusterStatistics`);
    validatePackedView(output.statistics, ['float32'], `${id} output.statistics`);
    validatePackedView(output.summary, ['uint32'], `${id} output.summary`);
    this.zones = props.positions.length;
    this.timeBuckets = props.timeBuckets ?? 1;
    this.maximumClusters = props.maximumClusters ?? 8;
    const {maximumPermutations} = props;
    if (
      props.batchReplicates !== undefined &&
      (!Number.isInteger(props.batchReplicates) || props.batchReplicates < 1)
    ) {
      throw new Error(`${id} batchReplicates must be a positive integer`);
    }
    if (this.zones < 1) {
      throw new Error(`${id} needs at least one zone`);
    }
    validateCapacity(
      id,
      'timeBuckets',
      this.timeBuckets,
      1,
      GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS
    );
    const layout = getGPUSpatialScanStatisticLayout(this.zones, props.maximumWindowZones);
    this.usableZones = layout.usableZones;
    this.listLength = layout.listLength;
    validateCapacity(
      id,
      'maximumClusters',
      this.maximumClusters,
      1,
      GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_CLUSTERS
    );
    validateCapacity(
      id,
      'maximumPermutations',
      maximumPermutations,
      1,
      GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS
    );
    const cells = this.zones * this.timeBuckets;
    if (
      props.cases.length !== cells ||
      (this.model === 'poisson' && props.baseline!.length !== cells) ||
      (this.model === 'bernoulli' && props.controls!.length !== cells)
    ) {
      throw new Error(`${id} cases, baseline and controls need zones * timeBuckets rows`);
    }
    // The count table holds the observed slot and one batch of replicates.
    const budgetReplicates = Math.floor(MAXIMUM_REPLICATE_WORDS / cells) - 1;
    if (budgetReplicates < 1) {
      throw new Error(`${id} zones * timeBuckets must be <= ${MAXIMUM_REPLICATE_WORDS / 2}`);
    }
    this.batchReplicates = Math.min(
      maximumPermutations,
      MAXIMUM_BATCH_REPLICATES,
      budgetReplicates,
      props.batchReplicates ?? Infinity
    );
    this.batchCount = Math.ceil(maximumPermutations / this.batchReplicates);
    if (this.batchCount > MAXIMUM_BATCHES) {
      throw new Error(
        `${id} maximumPermutations needs ${this.batchCount} batches of ${this.batchReplicates} replicates, more than ${MAXIMUM_BATCHES}`
      );
    }
    if (props.parameters.length < GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters needs ${GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH} words`
      );
    }
    if (
      output.clusterIndices.length <
      this.maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS
    ) {
      throw new Error(`${id} output.clusterIndices needs maximumClusters * 4 rows`);
    }
    if (
      output.clusterStatistics.length <
      this.maximumClusters * GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS
    ) {
      throw new Error(`${id} output.clusterStatistics needs maximumClusters * 8 rows`);
    }
    if (output.statistics.length < maximumPermutations + 1) {
      throw new Error(`${id} output.statistics needs maximumPermutations + 1 rows`);
    }
    if (output.summary.length < GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary needs ${GPU_SPATIAL_SCAN_STATISTIC_SUMMARY_LENGTH} rows`
      );
    }
    if (output.zoneStatistics) {
      validatePackedView(output.zoneStatistics, ['float32'], `${id} output.zoneStatistics`);
      if (output.zoneStatistics.length < this.zones) {
        throw new Error(`${id} output.zoneStatistics needs one row per zone`);
      }
    }
    const outputs = [
      output.clusterIndices,
      output.clusterStatistics,
      output.statistics,
      output.summary,
      ...(output.zoneStatistics ? [output.zoneStatistics] : [])
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
   * @param settings The per-frame settings packed into `parameters` (validated as the packer
   * does): `rateDirection` selects the alternative (`'high'`: `'greater'`, `'low'`: `'lesser'`,
   * `'both'`: `'two-sided'`; windows of both directions compete in one maximum),
   * `permutationCount` the replicate count (clamped to `maximumPermutations`, as the kernels do)
   * and `seed` the stream seed.
   */
  getInferenceMetadata(settings: GPUSpatialScanStatisticSettings): GPUInferenceMetadata {
    // Same validation as the packed values, so the metadata never describes an unpackable frame.
    getGPUSpatialScanStatisticParameterValues(settings);
    const direction = settings.rateDirection ?? 'high';
    return {
      alternative: direction === 'high' ? 'greater' : direction === 'low' ? 'lesser' : 'two-sided',
      // The kernels clamp the packed count to the compile-time maximum.
      permutationCount: Math.min(settings.permutationCount, this.props.maximumPermutations),
      seed: settings.seed,
      multipleTesting: 'none',
      testedRowsDefinition:
        'every window: a center zone, its nearest zones within maximumPopulationFraction of the population, and a run of consecutive time buckets within maximumTimeBuckets; p-values are Monte Carlo maxima over all windows'
    };
  }

  /**
   * Returns the nodes in order: `nearest`, the model preparation (`population` for Bernoulli,
   * `margins` and `weights` for the space-time permutation model), `block-totals`, `totals`, for
   * Poisson and Bernoulli the cumulative table (`weight-sums`, `weight-scan`, `cdf`),
   * `init-counts`, `window-limit`, `evaluate` (the observed data), then per replicate batch
   * `batch-control`, `batch-clear`, the draw kernel (`scatter-<batch>`, `draw-labels-<batch>` or
   * `draw-times-<batch>`) and `batch-evaluate`, then `select-init`, then per cluster
   * `round-members`, `round-mask`, `round-partials` and `round-select`, and `summary`. `evaluate`
   * and `batch-evaluate` reduce the per-replicate maxima themselves (workgroup max, one
   * `atomicMax` per workgroup), so there is no per-center candidate table and no separate maximum
   * pass. `batch-control` advances a GPU cursor and arms or disarms the indirect dispatch of its
   * batch's kernels, so batches past the per-frame replicate count launch nothing. The graph has
   * `4 * batchCount + 1` replicate nodes. The round kernels other than `round-select` are gated by
   * a GPU-written indirect dispatch, so once no positive window is left the remaining rounds
   * launch no workgroups.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zones, timeBuckets, listLength, usableZones, maximumClusters, model} = this;
    const separation: InterpolationSeparation = {
      context: this.spatialContext,
      geographic: this.spatialContext.coordinateSpace === 'longitude-latitude',
      sphereRadius: getGPUSphereRadius(this.spatialContext)
    };
    const separationWGSL = getSeparationWGSL(separation);
    const {positions, cases, parameters, maximumPermutations} = props;
    const {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics} = props.output;
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
    const {batchReplicates, batchCount} = this;
    // Table slot 0 holds the observed data, slots 1..R the replicates of the running batch.
    const tableSlots = batchReplicates + 1;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const neighbors = u32('neighbors', zones * listLength);
    const distances = f32('distances', zones * listLength);
    // Ping-pong rows of the per-center radix sort in `nearest`; dead once the tables are built.
    const sortKeys = u32('sort-keys', zones * listLength);
    const sortZones = u32('sort-zones', zones * listLength);
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
    // Words: cases, baseline bits, weight total, next batch base, running batch base (replicates
    // done before the running batch). The batch control kernels advance the two base words.
    const totals = u32('totals', TOTALS_WORDS);
    const cumulative = u32('cumulative', cells);
    const counts = u32('counts', tableSlots * cells);
    const evaluateGroups = Math.ceil(zones / EVALUATE_WORKGROUP_SIZE);
    // Per-center best-window records, then (from WINDOW_INFO_START) two words per center written by
    // `window-limit` for `evaluate`: the window slot count and the capacity-truncation flag. They
    // share one buffer to keep `evaluate` within the 8 storage buffers per stage of core devices.
    const centerBest = u32('center-best', zones * CENTER_WORDS + zones * 2);
    const alive = u32('alive', zones);
    const partials = u32('partials', zoneBlocks * 2);
    const state = u32('state', 8);
    // Rule 13: once no positive window is left, the remaining cluster rounds launch nothing.
    // `select-init` arms one indirect dispatch per gated round kernel (`round-members`,
    // `round-mask`, `round-partials`) and `round-select` disarms them all; `round-select` itself
    // stays ungated because it writes the gate (a buffer is never both an indirect argument and
    // storage-written in one dispatch).
    const gatedInvocationCounts = [listLength, zones, zoneBlocks];
    const gatedWorkgroupSizes = gatedInvocationCounts.map(getAutomaticWGSLKernelWorkgroupSize);
    const gateLayouts = gatedInvocationCounts.map((count, slot) =>
      getBoundedDispatchLayout(
        OPERATION,
        Math.max(count, 1),
        gatedWorkgroupSizes[slot],
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const roundGate = createTransientView(
      graph,
      `${id}-round-dispatch`,
      'uint32',
      3 * gateLayouts.length,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    const getRoundGate = (
      slot: number
    ): {
      workgroupSize: number;
      condition: GPUCommandGraphNodeCondition<Parameters>;
      extraResources: GraphBufferUse[];
    } => ({
      workgroupSize: gatedWorkgroupSizes[slot],
      condition: {
        id: `${id}-round-gate-${slot}`,
        source: 'gpu',
        mode: 'indirect',
        buffer: roundGate.buffer,
        byteOffset: roundGate.byteOffset + slot * 12
      },
      extraResources: [{buffer: roundGate, usage: 'indirect'}]
    });
    const armRoundGateWGSL = gateLayouts
      .map(
        (layout, slot) => `roundDispatch[roundDispatchOffset + ${3 * slot}u] = ${layout.x}u;
    roundDispatch[roundDispatchOffset + ${3 * slot + 1}u] = ${layout.y}u;
    roundDispatch[roundDispatchOffset + ${3 * slot + 2}u] = ${layout.z}u;`
      )
      .join('\n    ');
    const disarmRoundGateWGSL = gateLayouts
      .map((_, slot) => `roundDispatch[roundDispatchOffset + ${3 * slot}u] = 0u;`)
      .join(' ');

    // Rule 13: a batch whose first replicate is past the per-frame count launches nothing. The
    // cursor is the GPU word `totals[3]`, so the batch index never reaches the WGSL text (rule 12).
    // `batch-control` is ungated because it writes the gate.
    const drawInvocationCount =
      model === 'poisson' ? batchReplicates * CASE_BLOCKS : batchReplicates;
    const batchGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-batch-dispatch`,
      operation: OPERATION,
      shapes: [
        [batchReplicates * cells],
        [drawInvocationCount],
        [batchReplicates * evaluateGroups * EVALUATE_WORKGROUP_SIZE, EVALUATE_WORKGROUP_SIZE]
      ]
    });

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
const BATCH_REPLICATES: u32 = ${batchReplicates}u;
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
// (1 + d) ln(1 + d) - d for d >= -1. Below |d| = 1/4 the Taylor series
// d^2 * sum_k>=2 (-d)^(k - 2) / (k (k - 1)), terms to k = 14 (truncation below 1e-9 relative),
// avoids the cancellation of the direct form.
fn getEntropyExcess(delta: f32) -> f32 {
  if (abs(delta) < 0.25) {
    var series = 1.0 / 182.0;
    series = -1.0 / 156.0 + delta * series;
    series = 1.0 / 132.0 + delta * series;
    series = -1.0 / 110.0 + delta * series;
    series = 1.0 / 90.0 + delta * series;
    series = -1.0 / 72.0 + delta * series;
    series = 1.0 / 56.0 + delta * series;
    series = -1.0 / 42.0 + delta * series;
    series = 1.0 / 30.0 + delta * series;
    series = -1.0 / 20.0 + delta * series;
    series = 1.0 / 12.0 + delta * series;
    series = -1.0 / 6.0 + delta * series;
    series = 1.0 / 2.0 + delta * series;
    return delta * delta * series;
  }
  return (1.0 + delta) * log(1.0 + delta) - delta;
}
// Kulldorff term o ln(o/e) minus its linear part o - e, which sums to 0 over the cells of a full
// partition of the cases, as e * ((1 + d) ln(1 + d) - d) with d = (o - e) / e: near o = e the
// LLR no longer comes from the difference of two large logarithm terms.
fn getObservedTerm(observed: f32, expected: f32) -> f32 {
  if (!(expected > 0.0)) {
    return 0.0;
  }
  if (!(observed > 0.0)) {
    return expected;
  }
  return expected * getEntropyExcess((observed - expected) / expected);
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
  // A window holding the whole expectation leaves no outside cell; its LLR is never positive.
  if (!(totalCases - expected > 0.0)) {
    return 0.0;
  }
  let ratio = getObservedTerm(observed, expected) +
    getObservedTerm(max(totalCases - observed, 0.0), totalCases - expected);
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
          {name: 'distances', view: distances, type: 'u32', access: 'read_write'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortZones', view: sortZones, type: 'u32', access: 'read_write'}
        ],
        invocationCount: zones,
        declarations: `${separationWGSL}
const ZONES: u32 = ${zones}u;
const LIST: u32 = ${listLength}u;
const RADIX_BINS: u32 = ${NEIGHBOR_RADIX_BINS}u;
// Separation keys are non-negative f32, so their bits order as u32 keys.
fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {
  let zoneX = positions[positionsOffset + zone * 2u];
  let zoneY = positions[positionsOffset + zone * 2u + 1u];
  return bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)));
}`,
        // The row of a center lists its LIST nearest zones in (separation key, zone index) order.
        // When LIST < ZONES a radix select over the key bits (8-bit digits, most significant
        // first, one pass over the zones per digit) finds the key of the LIST-th nearest zone and
        // how many zones at that key still fit. One more pass gathers the selected zones in zone
        // order, and an LSD radix sort (four stable 8-bit passes, ping-ponging with the sort
        // scratch rows) orders them by key, so equal keys keep ascending zone order. Work per
        // center is O(ZONES + LIST), not the O(ZONES * LIST) of a selection by repeated scans.
        body: `var histogram: array<u32, RADIX_BINS>;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  let row = index * LIST;
  var threshold = 0xffffffffu;
  var equalCount = LIST;
  if (LIST < ZONES) {
    var prefix = 0u;
    var prefixMask = 0u;
    var rank = LIST - 1u;
    for (var level = 0u; level < 4u; level++) {
      let shift = 24u - 8u * level;
      for (var bin = 0u; bin < RADIX_BINS; bin++) {
        histogram[bin] = 0u;
      }
      for (var zone = 0u; zone < ZONES; zone++) {
        let key = getZoneSeparationKey(zone, centerX, centerY);
        if ((key & prefixMask) == prefix) {
          let bin = (key >> shift) & 255u;
          histogram[bin] = histogram[bin] + 1u;
        }
      }
      var digit = 0u;
      loop {
        if (digit == RADIX_BINS - 1u || rank < histogram[digit]) {
          break;
        }
        rank -= histogram[digit];
        digit++;
      }
      prefix = prefix | (digit << shift);
      prefixMask = prefixMask | (255u << shift);
    }
    threshold = prefix;
    equalCount = rank + 1u;
  }
  var filled = 0u;
  var equalTaken = 0u;
  for (var zone = 0u; zone < ZONES; zone++) {
    if (filled >= LIST) {
      break;
    }
    let key = getZoneSeparationKey(zone, centerX, centerY);
    var take = key < threshold;
    if (!take && key == threshold && equalTaken < equalCount) {
      take = true;
      equalTaken++;
    }
    if (take) {
      distances[distancesOffset + row + filled] = key;
      neighbors[neighborsOffset + row + filled] = zone;
      filled++;
    }
  }
  for (var shift = 0u; shift < 32u; shift += 16u) {
    // Digit shift: the table rows to the sort scratch.
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (distances[distancesOffset + row + slot] >> shift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    var running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = distances[distancesOffset + row + slot];
      let bin = (key >> shift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      sortKeys[sortKeysOffset + row + destination] = key;
      sortZones[sortZonesOffset + row + destination] = neighbors[neighborsOffset + row + slot];
    }
    // Digit shift + 8: the sort scratch back to the table rows.
    let nextShift = shift + 8u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      histogram[bin] = 0u;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let bin = (sortKeys[sortKeysOffset + row + slot] >> nextShift) & 255u;
      histogram[bin] = histogram[bin] + 1u;
    }
    running = 0u;
    for (var bin = 0u; bin < RADIX_BINS; bin++) {
      let count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    for (var slot = 0u; slot < LIST; slot++) {
      let key = sortKeys[sortKeysOffset + row + slot];
      let bin = (key >> nextShift) & 255u;
      let destination = histogram[bin];
      histogram[bin] = destination + 1u;
      distances[distancesOffset + row + destination] = key;
      neighbors[neighborsOffset + row + destination] = sortZones[sortZonesOffset + row + slot];
    }
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
  for (var blockIndex = 0u; blockIndex < BLOCKS; blockIndex++) {
    baselineSum += blockBaseline[blockBaselineOffset + blockIndex];
    caseSum += blockCases[blockCasesOffset + blockIndex];
  }
  totals[totalsOffset] = caseSum;
  totals[totalsOffset + 1u] = bitcast<u32>(baselineSum);
  totals[totalsOffset + 2u] = 0u;
  totals[totalsOffset + 3u] = 0u;
  totals[totalsOffset + 4u] = 0u;`
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
  for (var blockIndex = 0u; blockIndex < BLOCKS; blockIndex++) {
    blockOffsets[blockOffsetsOffset + blockIndex] = running;
    running += blockWeights[blockWeightsOffset + blockIndex];
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

    // Slot 0 of the count table is the observed data; each batch clears and refills slots 1..R.
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
        invocationCount: Math.max(cells, statistics.length),
        declarations: `${common}
const STATISTIC_WORDS: u32 = ${statistics.length}u;`,
        // The same pass zeroes `statistics` (0 bits are 0.0), the target of the evaluate maxima.
        body: `if (index < CELLS) {
    counts[countsOffset + index] = cases[casesOffset + index];
  }
  if (index < STATISTIC_WORDS) {
    statistics[statisticsOffset + index] = 0u;
  }`
      })
    );

    // Draws the replicates of the running batch into table slots 1..R. A replicate is identified by
    // its global index `totals[4] + local`, which keys its Philox stream, so the draws do not depend
    // on the batch size.
    let createDrawNode: (batch: number) => GPUCommandNode<Parameters>;
    if (model === 'poisson') {
      createDrawNode = batch =>
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-scatter-${batch}`,
          ...batchGate.getGate(1),
          operation: OPERATION,
          variant: 'scatter',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read'},
            {name: 'cumulative', view: cumulative, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: drawInvocationCount,
          declarations: `${PERMUTATION_RANDOM_WGSL}
${withParameters}
const CASE_BLOCKS: u32 = ${CASE_BLOCKS}u;
const SCAN_STREAM: u32 = ${SCAN_STREAM}u;`,
          body: `let localReplicate = index / CASE_BLOCKS;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let blockIndex = index - localReplicate * CASE_BLOCKS;
  let totalCases = totals[totalsOffset];
  let totalWeight = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalWeight == 0u) {
    return;
  }
  let chunk = (totalCases + CASE_BLOCKS - 1u) / CASE_BLOCKS;
  let first = blockIndex * chunk;
  let stop = min(first + chunk, totalCases);
  var stream = createPhiloxStream(readSeedKey(), replicate, blockIndex, SCAN_STREAM);
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
    atomicAdd(&counts[countsOffset + (localReplicate + 1u) * CELLS + low], 1u);
  }`
        });
    } else if (model === 'bernoulli') {
      createDrawNode = batch =>
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-draw-labels-${batch}`,
          ...batchGate.getGate(1),
          operation: OPERATION,
          variant: 'draw-labels',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read'},
            {name: 'cumulative', view: cumulative, type: 'u32', access: 'read'},
            {name: 'population', view: population!, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: drawInvocationCount,
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
          body: `let localReplicate = index;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  let totalCases = totals[totalsOffset];
  let totalPopulation = totals[totalsOffset + 2u];
  if (replicate >= readPermutations() || totalPopulation == 0u) {
    return;
  }
  let complement = totalCases > totalPopulation - totalCases;
  let drawCount = select(totalCases, totalPopulation - totalCases, complement);
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (localReplicate + 1u) * CELLS;
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
        });
    } else {
      createDrawNode = batch =>
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-draw-times-${batch}`,
          ...batchGate.getGate(1),
          operation: OPERATION,
          variant: 'draw-times',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read'},
            {name: 'zoneTotals', view: zoneTotals!, type: 'u32', access: 'read'},
            {name: 'bucketTotals', view: bucketTotals!, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: drawInvocationCount,
          declarations: `${PERMUTATION_RANDOM_WGSL}
${withParameters}
const SCAN_STREAM: u32 = ${SCAN_STREAM}u;`,
          // One replicate per invocation: the case times are permuted over the case locations. Each
          // zone in index order takes its cases' times without replacement from the times left, so
          // both margins stay fixed and the table is a multivariate hypergeometric draw.
          body: `let localReplicate = index;
  let replicate = totals[totalsOffset + 4u] + localReplicate;
  if (replicate >= readPermutations()) {
    return;
  }
  var remaining: array<u32, BUCKETS>;
  for (var bucket = 0u; bucket < BUCKETS; bucket++) {
    remaining[bucket] = bucketTotals[bucketTotalsOffset + bucket];
  }
  var left = totals[totalsOffset];
  var stream = createPhiloxStream(readSeedKey(), replicate, 0u, SCAN_STREAM);
  let row = countsOffset + (localReplicate + 1u) * CELLS;
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
        });
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
      })
    );

    // The observed slot is evaluated once and records each center's best window; a batch evaluates
    // its replicates (global index `totals[4] + local`) and folds each replicate's maximum into
    // its row of `statistics`. The window loops and the maximum are the same for both, and the
    // maximum is order independent, so a replicate's value does not depend on the batch layout.
    const createEvaluateNode = (observed: boolean, batch: number) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: observed ? `${id}-evaluate` : `${id}-batch-evaluate-${batch}`,
        operation: OPERATION,
        variant: observed ? 'evaluate' : 'batch-evaluate',
        ...(observed ? {} : batchGate.getGate(2)),
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'baseline', view: weights, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'atomic<u32>', access: 'read_write'},
          {
            name: 'centerBest',
            view: centerBest,
            type: 'u32',
            access: observed ? 'read_write' : 'read'
          }
        ],
        // One workgroup per (replicate, 256 consecutive centers); every invocation reaches the
        // barriers, so the body is unguarded.
        invocationCount:
          (observed ? 1 : batchReplicates) * evaluateGroups * EVALUATE_WORKGROUP_SIZE,
        workgroupSize: EVALUATE_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${withParameters}
${likelihood}
${observed ? separationWGSL : ''}
const EVALUATE_GROUPS: u32 = ${evaluateGroups}u;
const EVALUATE_SIZE: u32 = ${EVALUATE_WORKGROUP_SIZE}u;
var<workgroup> maxima: array<f32, ${EVALUATE_WORKGROUP_SIZE}>;`,
        body: `let group = index / EVALUATE_SIZE;
  let localReplicate = group / EVALUATE_GROUPS;
  let center = (group - localReplicate * EVALUATE_GROUPS) * EVALUATE_SIZE + localInvocationIndex;
  ${
    observed
      ? `let slot = 0u;
  let statisticRow = 0u;
  let isLive = true;`
      : `let replicate = totals[totalsOffset + 4u] + localReplicate;
  let slot = localReplicate + 1u;
  let statisticRow = replicate + 1u;
  let isLive = replicate < readPermutations();`
  }
  var best = 0.0;
  if (isLive && center < ZONES) {
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
  ${
    observed
      ? `  if (slot == 0u) {
    let record = centerBestOffset + center * CENTER_WORDS;
    centerBest[record] = bitcast<u32>(best);
    centerBest[record + 1u] = bestZones;
    centerBest[record + 2u] = bestFirst;
    centerBest[record + 3u] = bestLast;
    centerBest[record + 4u] = bitcast<u32>(bestCases);
    centerBest[record + 5u] = bitcast<u32>(bestExpected);
    centerBest[record + 6u] = centerBest[centerBestOffset + WINDOW_INFO_START + center * 2u + 1u];
    // Squared radius in context units: decode the key of the farthest zone.
    centerBest[record + 7u] = select(
      0u,
      bitcast<u32>(getSquaredMeters(distances[distancesOffset + center * LIST + max(bestZones, 1u) - 1u])),
      bestZones > 0u
    );
  }
`
      : ''
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
    atomicMax(&statistics[statisticsOffset + statisticRow], bitcast<u32>(maxima[0]));
  }`
      });

    nodes.push(createEvaluateNode(true, 0));

    for (let batch = 0; batch < batchCount; batch++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-control-${batch}`,
          operation: OPERATION,
          variant: 'batch-control',
          bindings: [
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read_write'},
            batchGate.binding
          ],
          invocationCount: 1,
          declarations: withParameters,
          // Replicates before this batch: the cursor advances on the GPU, so every batch shares one
          // pipeline. A batch starting at or past the per-frame count is disarmed.
          body: `let base = totals[totalsOffset + 3u];
  totals[totalsOffset + 4u] = base;
  totals[totalsOffset + 3u] = base + BATCH_REPLICATES;
  if (base < readPermutations()) {
    ${batchGate.armWGSL}
  } else {
    ${batchGate.disarmWGSL}
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-clear-${batch}`,
          operation: OPERATION,
          variant: 'batch-clear',
          ...batchGate.getGate(0),
          bindings: [{name: 'counts', view: counts, type: 'u32', access: 'read_write'}],
          invocationCount: batchReplicates * cells,
          declarations: common,
          // Slots 1..R; slot 0 keeps the observed counts.
          body: `counts[countsOffset + CELLS + index] = 0u;`
        }),
        createDrawNode(batch),
        createEvaluateNode(false, batch)
      );
    }

    nodes.push(
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
          {name: 'roundDispatch', view: roundGate, type: 'u32', access: 'read_write'},
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
  if (index == 0u) {
    ${armRoundGateWGSL}
  }
  if (index < CLUSTERS) {
    for (var word = 0u; word < ${GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS}u; word++) {
      clusterIndices[clusterIndicesOffset + index * ${GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS}u + word] = 0u;
    }
    for (var word = 0u; word < ${GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS}u; word++) {
      clusterStatistics[clusterStatisticsOffset + index * ${GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS}u + word] = 0.0;
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
          ...getRoundGate(0),
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
          ...getRoundGate(1),
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
          ...getRoundGate(2),
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
            {name: 'roundDispatch', view: roundGate, type: 'u32', access: 'read_write'},
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
  for (var blockIndex = 0u; blockIndex < ZONE_BLOCKS; blockIndex++) {
    let ratio = bitcast<f32>(partials[partialsOffset + blockIndex * 2u]);
    let center = partials[partialsOffset + blockIndex * 2u + 1u];
    if (center != NO_CENTER && ratio > best) {
      best = ratio;
      bestCenter = center;
    }
  }
  if (bestCenter == NO_CENTER) {
    state[stateOffset + 3u] = 0u;
    state[stateOffset + 4u] = 1u;
    ${disarmRoundGateWGSL}
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
  clusterIndices[clusterIndicesOffset + count * ${GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS}u] = bestCenter;
  clusterIndices[clusterIndicesOffset + count * ${GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS}u + 1u] = zoneCount;
  clusterIndices[clusterIndicesOffset + count * ${GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS}u + 2u] = centerBest[record + 2u];
  clusterIndices[clusterIndicesOffset + count * ${GPU_SPATIAL_SCAN_STATISTIC_INDEX_WORDS}u + 3u] = centerBest[record + 3u];
  let statisticStart = clusterStatisticsOffset + count * ${GPU_SPATIAL_SCAN_STATISTIC_STATISTIC_WORDS}u;
  clusterStatistics[statisticStart] = best;
  clusterStatistics[statisticStart + 1u] = observed;
  clusterStatistics[statisticStart + 2u] = expected;
  clusterStatistics[statisticStart + 3u] = f32(exceeding + 1u) / f32(permutations + 1u);
  // Word 7 already holds squared distance in the context unit, not a separation key.
  clusterStatistics[statisticStart + 4u] = sqrt(bitcast<f32>(centerBest[record + 7u]));
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
