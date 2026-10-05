// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createTransientUint32Rows} from '../../utils/wgsl-kernel-nodes';
import type {GPUFloat32Positions, GPUUint32Rows} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGraphViewChunks,
  validateGraphViewsBelongToGraph,
  validateGraphViewTopology
} from '../../utils/gpu-contributor-utils';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import {validateDisjointOutputs} from '../spatial-join/spatial-join-passes';
import {
  createTransientFloat32Rows,
  createZonalStatisticsAreaNode,
  createZonalStatisticsDensityNode,
  createZonalStatisticsMeanNode,
  createZonalStatisticsPrepareNodes,
  getZonalStatisticsExtentNodes,
  getZonalStatisticsPlan,
  type ZonalStatisticsPlan
} from './zonal-statistics-passes';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';

const OPERATION = 'GPUZonalStatistics';

/**
 * Feature membership from polygon features: points are joined to polygons with
 * `GPUPointInPolygonJoin` every encoding. A point belongs to the containing feature with the
 * smallest row.
 *
 * Polygon geometry uses the GeoArrow-style layout of `GPUPointInPolygonJoin`; see that recipe for
 * the offset conventions. The polygon buffers' contents are per-frame; everything else here is
 * compile-time topology.
 */
export type GPUZonalStatisticsPolygons = {
  /** Feature source discriminator. */
  kind: 'polygons';
  /** Flattened polygon vertices. Per-frame contents. */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. Per-frame contents. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. Ring 0 is the shell. Per-frame contents. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. Per-frame contents. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Maximum `(point, feature)` bounding-box candidates per encoding. Compile-time. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. Compile-time. */
  leafCapacity?: number;
  /** Count points on a ring boundary as contained. Defaults to `true`. Compile-time. */
  includeBoundary?: boolean;
};

/**
 * Feature membership computed by the caller (for example by a join with different semantics, a
 * lookup table or a previous frame's result). No geometry is read.
 */
export type GPUZonalStatisticsFeatureRows = {
  /** Feature source discriminator. */
  kind: 'feature-rows';
  /**
   * Per-point feature row in `[0, featureCount)`. `GPU_SPATIAL_JOIN_NO_FEATURE` (or any value
   * `>= featureCount`) marks an unassigned point. Per-frame contents; the chunk topology defines
   * the point count, and `values` and `weights` must have the same total length (their chunk
   * boundaries may differ in the `'atomic'` sum order).
   */
  pointFeatureRows: GPUUint32Rows;
  /** Number of features, which is the row count of every per-feature output. Compile-time. */
  featureCount: number;
};

/** Feature membership source for {@link GPUZonalStatistics}. */
export type GPUZonalStatisticsFeatures = GPUZonalStatisticsPolygons | GPUZonalStatisticsFeatureRows;

/**
 * How per-feature sums and weight sums are accumulated. Compile-time.
 *
 * - `'atomic'`: compare-exchange float atomics through `GPUGroupAggregation`. Fast, but the
 *   addition order depends on GPU scheduling, so sums can differ in the last bits between runs and
 *   devices.
 * - `'sorted'`: points are stably sorted by feature row, then reduced per feature in a fixed tree
 *   order, which is bitwise reproducible across runs on the same device. The tree order differs
 *   from a sequential CPU sum, so results match a CPU oracle only within a float32 tolerance. The
 *   sort costs O(N) work per radix pass with `ceil(log2(featureCount + 1) / 4)` passes, plus a
 *   scan and gather, and the segmented reduction uses one 256-thread workgroup per feature.
 */
export type GPUZonalStatisticsSumOrder = 'atomic' | 'sorted';

/** Statistic whose `[min, max]` extent over features is published to `output.extent`. */
export type GPUZonalStatisticsExtentStatistic =
  | 'count'
  | 'sum'
  | 'mean'
  | 'minimum'
  | 'maximum'
  | 'density';

/**
 * Caller-owned outputs of {@link GPUZonalStatistics}. Which views are present is compile-time.
 *
 * Every per-feature output has exactly `featureCount` rows and is rewritten on every encoding.
 */
export type GPUZonalStatisticsOutput = {
  /** Points assigned to the feature, regardless of value validity. */
  counts?: GraphDataView<'uint32'>;
  /** Assigned points with a finite value (and a finite weight when weights are given). */
  valueCounts?: GraphDataView<'uint32'>;
  /** `sum(v)`, or `sum(w * v)` with weights, over value-valid rows. 0 for empty features. */
  sums?: GraphDataView<'float32'>;
  /** `sum(w)` over value-valid rows. Requires `weights`. 0 for empty features. */
  weightSums?: GraphDataView<'float32'>;
  /**
   * `sum(v) / valueCount`, or `sum(w * v) / sum(w)` with weights. Quiet NaN (`0x7fc00000`) when the
   * feature has no valid rows or the weight sum is 0.
   */
  means?: GraphDataView<'float32'>;
  /** Minimum of the valid values, never weighted. NaN for features without valid rows. */
  minima?: GraphDataView<'float32'>;
  /** Maximum of the valid values, never weighted. NaN for features without valid rows. */
  maxima?: GraphDataView<'float32'>;
  /**
   * `counts / area` when the area is finite and positive, otherwise NaN. The area is `areas` when
   * given, otherwise the GPU polygon area. Weighted density (`weightSums / area`) is not computed.
   */
  densities?: GraphDataView<'float32'>;
  /**
   * GPU-computed polygon areas. Only valid for `kind: 'polygons'` without `areas`.
   * Per feature and polygon: `|area(shell)| - sum |area(hole)|`, summed over polygons, with each
   * ring's shoelace sum taken relative to the ring's first finite vertex. Non-finite vertices are
   * skipped (their neighbors are joined), and ranges that do not fit the offset arrays contribute
   * 0. One invocation handles a whole feature, so the cost per thread is proportional to the
   * vertex count of the largest feature. Overlapping polygons of one feature are summed, not
   * unioned.
   */
  featureAreas?: GraphDataView<'float32'>;
  /**
   * `kind: 'polygons'` only: receives the join's per-point feature row (or
   * `GPU_SPATIAL_JOIN_NO_FEATURE`), with the chunk topology of `points`.
   */
  pointFeatureRows?: GPUUint32Rows;
  /**
   * Two rows `[minimum, maximum]` of `extentStatistic` over features whose count is positive and
   * whose statistic is finite. `[0, 0]` when no feature qualifies. Counts and extrema are exact;
   * counts above 2^24 lose precision as float32.
   */
  extent?: GraphDataView<'float32'>;
  /**
   * Statistic that `extent` summarizes. Required with `extent`. The statistic is computed into a
   * graph transient when its own output view is absent. Compile-time.
   */
  extentStatistic?: GPUZonalStatisticsExtentStatistic;
  /**
   * One row: 1 when the join overflowed a BVH leaf or candidate capacity, otherwise 0. Always 0
   * for `kind: 'feature-rows'`.
   */
  overflow?: GraphDataView<'uint32'>;
  /**
   * One row: the number of (point, feature) candidate pairs whose containment could not be proven
   * (non-finite input, malformed offsets, or a predicate outside the double-single arithmetic
   * envelope). Uncertain pairs are never assigned: a point whose only candidates were uncertain
   * stays unassigned, so a nonzero value means `counts` and the other statistics may be low. Always
   * 0 for `kind: 'feature-rows'`.
   */
  uncertainCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUZonalStatistics}.
 *
 * Per-frame: the contents of every input buffer (points, values, weights, areas, geometry, rows).
 * Compile-time: view lengths and chunking, which outputs exist, `features` capacities and flags,
 * `sumOrder`, and `extentStatistic`.
 */
export type GPUZonalStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'zonal-statistics'`. */
  id?: string;
  /** Feature membership source. */
  features: GPUZonalStatisticsFeatures;
  /** Planar points. Required for `kind: 'polygons'`, ignored for `kind: 'feature-rows'`. */
  points?: GPUFloat32Positions;
  /**
   * Per-point values with the same total length as the points. Required by every value statistic
   * (`valueCounts`, `sums`, `weightSums`, `means`, `minima`, `maxima` and extents of them).
   * Non-finite values never contribute.
   */
  values?: GraphDataView<'float32'> | import('@luma.gl/gpgpu/gpu-core').GraphVectorView<'float32'>;
  /**
   * Per-point weights with the topology of `values`. Requires `values`. With weights, `sums` is
   * `sum(w * v)` and `means` is `sum(w * v) / sum(w)`; a row with a non-finite weight is invalid
   * for every value statistic. Products that overflow float32 are dropped from sums.
   */
  weights?: GraphDataView<'float32'> | import('@luma.gl/gpgpu/gpu-core').GraphVectorView<'float32'>;
  /**
   * Optional per-feature areas (`featureCount` rows). They replace the GPU-computed polygon area
   * and are required for densities with `kind: 'feature-rows'`. Per-frame contents.
   */
  areas?: GraphDataView<'float32'>;
  /** Accumulation order of sums and weight sums. Defaults to `'atomic'`. Compile-time. */
  sumOrder?: GPUZonalStatisticsSumOrder;
  /** Caller-owned outputs. At least one must be present. */
  output: GPUZonalStatisticsOutput;
};

const PER_FEATURE_FLOAT_OUTPUTS = [
  'sums',
  'weightSums',
  'means',
  'minima',
  'maxima',
  'densities',
  'featureAreas'
] as const;
const PER_FEATURE_UINT32_OUTPUTS = ['counts', 'valueCounts'] as const;
const OUTPUT_NAMES = [
  ...PER_FEATURE_UINT32_OUTPUTS,
  ...PER_FEATURE_FLOAT_OUTPUTS,
  'pointFeatureRows',
  'extent',
  'overflow',
  'uncertainCount'
] as const;

/**
 * Choropleth aggregation: joins points to features and reduces optional per-point values and
 * weights into per-feature statistics, all on the GPU.
 *
 * Membership comes from `GPUPointInPolygonJoin` (`kind: 'polygons'`) or from caller-provided
 * feature rows. The statistics computed are chosen at compile time by which output views exist;
 * statistics that other outputs depend on (for example sums and value counts for means) are
 * computed into graph transients.
 *
 * ## Semantics
 *
 * - A row is valid for value statistics when it is assigned to a feature row `< featureCount`,
 *   its value is finite, and, when weights are given, its weight is finite. Rows with non-finite
 *   values are still counted in `counts`.
 * - Empty features (no valid rows) have `sums` 0, `weightSums` 0, `valueCounts` 0, and NaN
 *   `means`, `minima` and `maxima` (quiet NaN `0x7fc00000`, as `GPUGroupAggregation`). A weighted
 *   mean with a weight sum of 0 is NaN.
 * - Density is `counts[f] / area[f]` when the area is finite and positive, otherwise NaN.
 * - The extent covers features with `counts > 0` and a finite statistic, and is `[0, 0]` when none
 *   qualifies.
 *
 * ## Determinism
 *
 * Counts, value counts, minima, maxima and extents of those are exact and independent of
 * scheduling. With `sumOrder: 'atomic'`, sums and weight sums use compare-exchange float
 * addition whose order depends on GPU scheduling: results can differ in the last bits between
 * runs and devices (and from a sequential CPU sum). With `sumOrder: 'sorted'` they are bitwise
 * reproducible on one device, at the cost of a stable sort of the points by feature row.
 *
 * ## Non-goals
 *
 * No weighted density, no median or quantiles, no variance, no weighted minima or maxima, no
 * geographic (spherical) areas, and no union of overlapping polygons within one feature. The
 * recipe never compiles, encodes, submits, or reads back.
 */
export class GPUZonalStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUZonalStatisticsProps;
  /** Number of features, the row count of every per-feature output. */
  readonly featureCount: number;
  /** Accumulation order of sums and weight sums. */
  readonly sumOrder: GPUZonalStatisticsSumOrder;
  /** Number of points. */
  readonly pointCount: number;
  /** Which statistics are computed, including transient ones. */
  readonly plan: ZonalStatisticsPlan;

  constructor(props: GPUZonalStatisticsProps) {
    this.id = props.id ?? 'zonal-statistics';
    this.props = props;
    this.sumOrder = props.sumOrder ?? 'atomic';
    const {id} = this;
    const {features, output, values, weights, areas} = props;
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }

    if (features.kind === 'polygons') {
      if (!props.points) {
        throw new Error(`${id} points are required for polygon features`);
      }
      for (const chunk of getGraphViewChunks(props.points)) {
        validatePackedView(chunk, ['float32x2'], `${id} points`);
      }
      validatePackedView(features.polygonPositions, ['float32x2'], `${id} polygonPositions`);
      for (const [name, view] of [
        ['featureOffsets', features.featureOffsets],
        ['polygonOffsets', features.polygonOffsets],
        ['ringOffsets', features.ringOffsets]
      ] as const) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} requires a terminal entry`);
        }
      }
      this.featureCount = features.featureOffsets.length - 1;
      this.pointCount = props.points.length;
    } else {
      for (const chunk of getGraphViewChunks(features.pointFeatureRows)) {
        validatePackedUint32View(chunk, `${id} pointFeatureRows`);
      }
      if (!Number.isSafeInteger(features.featureCount)) {
        throw new Error(`${id} featureCount must be an integer`);
      }
      if (output.pointFeatureRows) {
        throw new Error(`${id} output.pointFeatureRows requires polygon features`);
      }
      if (output.featureAreas) {
        throw new Error(`${id} output.featureAreas requires polygon features`);
      }
      this.featureCount = features.featureCount;
      this.pointCount = features.pointFeatureRows.length;
    }
    if (this.featureCount < 1) {
      throw new Error(`${id} requires at least one feature`);
    }

    if (!OUTPUT_NAMES.some(name => output[name])) {
      throw new Error(`${id} requires at least one output`);
    }
    if (output.extent && !output.extentStatistic) {
      throw new Error(`${id} output.extent requires output.extentStatistic`);
    }
    if (
      output.extentStatistic &&
      !['count', 'sum', 'mean', 'minimum', 'maximum', 'density'].includes(output.extentStatistic)
    ) {
      throw new Error(`${id} output.extentStatistic is not a supported statistic`);
    }
    if (output.featureAreas && areas) {
      throw new Error(`${id} output.featureAreas cannot be requested together with areas`);
    }

    if (weights && !values) {
      throw new Error(`${id} weights require values`);
    }
    if (output.weightSums && !weights) {
      throw new Error(`${id} output.weightSums requires weights`);
    }
    for (const [name, view] of [
      ['values', values],
      ['weights', weights]
    ] as const) {
      if (!view) {
        continue;
      }
      for (const chunk of getGraphViewChunks(view)) {
        validatePackedView(chunk, ['float32'], `${id} ${name}`);
      }
      if (view.length !== this.pointCount) {
        throw new Error(`${id} ${name} length must equal the point count`);
      }
    }
    if (values && weights) {
      validateGraphViewTopology(id, 'weights', values, weights);
    }
    if (areas) {
      validatePackedView(areas, ['float32'], `${id} areas`);
      if (areas.length !== this.featureCount) {
        throw new Error(`${id} areas length must equal the feature count`);
      }
    }

    for (const name of PER_FEATURE_UINT32_OUTPUTS) {
      const view = output[name];
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} output.${name} length must equal the feature count`);
        }
      }
    }
    for (const name of PER_FEATURE_FLOAT_OUTPUTS) {
      const view = output[name];
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} output.${name} length must equal the feature count`);
        }
      }
    }
    if (output.extent) {
      validatePackedView(output.extent, ['float32'], `${id} output.extent`);
      if (output.extent.length !== 2) {
        throw new Error(`${id} output.extent must contain two float32 rows`);
      }
    }
    if (output.overflow) {
      validatePackedUint32View(output.overflow, `${id} output.overflow`);
      if (output.overflow.length < 1) {
        throw new Error(`${id} output.overflow must contain one uint32 row`);
      }
    }
    if (output.uncertainCount) {
      validatePackedUint32View(output.uncertainCount, `${id} output.uncertainCount`);
      if (output.uncertainCount.length < 1) {
        throw new Error(`${id} output.uncertainCount must contain one uint32 row`);
      }
    }
    if (output.pointFeatureRows) {
      for (const chunk of getGraphViewChunks(output.pointFeatureRows)) {
        validatePackedUint32View(chunk, `${id} output.pointFeatureRows`);
      }
      validateGraphViewTopology(
        id,
        'output.pointFeatureRows',
        props.points!,
        output.pointFeatureRows
      );
    }

    this.plan = getZonalStatisticsPlan({
      outputs: {
        counts: Boolean(output.counts),
        valueCounts: Boolean(output.valueCounts),
        sums: Boolean(output.sums),
        weightSums: Boolean(output.weightSums),
        means: Boolean(output.means),
        minima: Boolean(output.minima),
        maxima: Boolean(output.maxima),
        densities: Boolean(output.densities),
        featureAreas: Boolean(output.featureAreas),
        extent: Boolean(output.extent)
      },
      extentStatistic: output.extentStatistic,
      hasWeights: Boolean(weights),
      hasAreas: Boolean(areas),
      sorted: this.sumOrder === 'sorted'
    });
    if (this.plan.needValueStatistics && !values) {
      throw new Error(`${id} value statistics (sums, means, minima, maxima, ...) require values`);
    }
    if (features.kind === 'feature-rows' && this.plan.needDensities && !areas) {
      throw new Error(`${id} densities require areas with feature-rows features`);
    }

    if (this.sumOrder === 'sorted') {
      const rows =
        features.kind === 'polygons' ? output.pointFeatureRows : features.pointFeatureRows;
      for (const [name, view] of [
        ['points', features.kind === 'polygons' ? props.points : undefined],
        ['values', values],
        ['weights', weights],
        ['pointFeatureRows', rows]
      ] as const) {
        if (view instanceof GraphVectorView) {
          throw new Error(`${id} sorted sum order requires packed ${name}`);
        }
      }
    }

    validateDisjointOutputs(
      id,
      [
        props.points,
        values,
        weights,
        areas,
        features.kind === 'polygons' ? features.polygonPositions : undefined,
        features.kind === 'polygons' ? features.featureOffsets : undefined,
        features.kind === 'polygons' ? features.polygonOffsets : undefined,
        features.kind === 'polygons' ? features.ringOffsets : undefined,
        features.kind === 'feature-rows' ? features.pointFeatureRows : undefined
      ],
      OUTPUT_NAMES.map(name => output[name])
    );
  }

  /**
   * Returns membership, preparation, area, aggregation, finalize and extent nodes in order.
   *
   * @throws If any view belongs to another graph.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, plan, sumOrder} = this;
    const {features, output, values, weights, areas} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.points,
      values,
      weights,
      areas,
      features.kind === 'polygons' ? features.polygonPositions : undefined,
      features.kind === 'polygons' ? features.featureOffsets : undefined,
      features.kind === 'polygons' ? features.polygonOffsets : undefined,
      features.kind === 'polygons' ? features.ringOffsets : undefined,
      features.kind === 'feature-rows' ? features.pointFeatureRows : undefined,
      ...OUTPUT_NAMES.map(name => output[name])
    ]);
    const sortedSums = sumOrder === 'sorted' && (plan.needSums || plan.needWeightSums);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const uint32Rows = (name: string, view: GraphDataView<'uint32'> | undefined, needed: boolean) =>
      view ??
      (needed ? createTransientView(graph, `${id}-${name}`, 'uint32', featureCount) : undefined);
    const float32Rows = (
      name: string,
      view: GraphDataView<'float32'> | undefined,
      needed: boolean
    ) =>
      view ??
      (needed ? createTransientView(graph, `${id}-${name}`, 'float32', featureCount) : undefined);

    const counts = uint32Rows('feature-counts', output.counts, plan.needCounts);
    const valueCounts = uint32Rows(
      'feature-value-counts',
      output.valueCounts,
      plan.needValueCounts
    );
    const sums = float32Rows('feature-sums', output.sums, plan.needSums);
    const weightSums = float32Rows('feature-weight-sums', output.weightSums, plan.needWeightSums);
    const means = float32Rows('feature-means', output.means, plan.needMeans);
    const minima = float32Rows('feature-minima', output.minima, plan.needMinima);
    const maxima = float32Rows('feature-maxima', output.maxima, plan.needMaxima);
    const densities = float32Rows('feature-densities', output.densities, plan.needDensities);

    // 1. Membership.
    let pointFeatureRows: GPUUint32Rows;
    if (features.kind === 'polygons') {
      pointFeatureRows =
        output.pointFeatureRows ??
        createTransientUint32Rows(graph, `${id}-point-feature-rows`, props.points!);
      const overflow = output.overflow ?? createTransientView(graph, `${id}-overflow`, 'uint32', 1);
      nodes.push(
        ...new GPUPointInPolygonJoin({
          id: `${id}-join`,
          points: props.points!,
          polygonPositions: features.polygonPositions,
          featureOffsets: features.featureOffsets,
          polygonOffsets: features.polygonOffsets,
          ringOffsets: features.ringOffsets,
          candidateCapacity: features.candidateCapacity,
          leafCapacity: features.leafCapacity,
          includeBoundary: features.includeBoundary,
          pointFeatureIds: pointFeatureRows,
          featureCounts: counts,
          overflow,
          uncertainCount: output.uncertainCount
        }).getCommandNodes(graph)
      );
    } else {
      pointFeatureRows = features.pointFeatureRows;
      if (counts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-count-aggregation`,
            keys: pointFeatureRows,
            output: counts
          }).getCommandNodes(graph)
        );
      }
      if (output.overflow) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-overflow`,
            operation: OPERATION,
            view: output.overflow,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
      if (output.uncertainCount) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-uncertain-count`,
            operation: OPERATION,
            view: output.uncertainCount,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
    }

    // 2. Per-point validity and contributions.
    let valueMask: GPUUint32Rows | undefined;
    let sumContributions: ReturnType<typeof createTransientFloat32Rows> | undefined;
    let weightContributions: ReturnType<typeof createTransientFloat32Rows> | undefined;
    if (plan.needValueStatistics && values) {
      valueMask = createTransientUint32Rows(graph, `${id}-value-mask`, values);
      if (plan.needSums && (weights || sumOrder === 'sorted')) {
        sumContributions = createTransientFloat32Rows(graph, `${id}-sum-contributions`, values);
      }
      if (plan.needWeightSums && sumOrder === 'sorted') {
        weightContributions = createTransientFloat32Rows(
          graph,
          `${id}-weight-contributions`,
          values
        );
      }
      nodes.push(
        ...createZonalStatisticsPrepareNodes<Parameters>(graph, {
          id,
          values,
          weights,
          valueMask,
          sumContributions,
          weightContributions
        })
      );
    }

    // 3. Polygon areas.
    let featureAreas = areas;
    if (features.kind === 'polygons' && plan.needGPUAreas) {
      featureAreas = float32Rows('feature-areas', output.featureAreas, true)!;
      nodes.push(
        createZonalStatisticsAreaNode<Parameters>(graph, {
          id: `${id}-area`,
          featureCount,
          polygonPositions: features.polygonPositions,
          featureOffsets: features.featureOffsets,
          polygonOffsets: features.polygonOffsets,
          ringOffsets: features.ringOffsets,
          areas: featureAreas
        })
      );
    }

    // 4. Aggregations over valid rows.
    if (valueMask && values) {
      if (valueCounts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-value-count-aggregation`,
            keys: pointFeatureRows,
            mask: valueMask,
            output: valueCounts
          }).getCommandNodes(graph)
        );
      }
      const statistics: [
        string,
        'sum' | 'min' | 'max',
        typeof values | undefined,
        GraphDataView<'float32'> | undefined
      ][] = [
        [
          'sum',
          'sum',
          weights ? sumContributions : values,
          sumOrder === 'atomic' ? sums : undefined
        ],
        ['weight-sum', 'sum', weights, sumOrder === 'atomic' ? weightSums : undefined],
        ['minimum', 'min', values, minima],
        ['maximum', 'max', values, maxima]
      ];
      for (const [name, operation, source, destination] of statistics) {
        if (source && destination) {
          nodes.push(
            ...new GPUGroupAggregation({
              id: `${id}-${name}-aggregation`,
              keys: pointFeatureRows,
              values: source,
              mask: valueMask,
              output: destination,
              operation
            }).getCommandNodes(graph)
          );
        }
      }
    }
    if (sortedSums) {
      nodes.push(
        ...getSortedSegmentSumNodes<Parameters>(graph, {
          id,
          operation: OPERATION,
          segmentCount: featureCount,
          segmentKeys: pointFeatureRows as GraphDataView<'uint32'>,
          segmentCounts: counts!,
          sumContributions: sumContributions as GraphDataView<'float32'> | undefined,
          weightContributions: weightContributions as GraphDataView<'float32'> | undefined,
          sums: plan.needSums ? sums : undefined,
          weightSums: plan.needWeightSums ? weightSums : undefined
        })
      );
    }

    // 5. Finalize derived statistics.
    if (plan.needMeans) {
      nodes.push(
        createZonalStatisticsMeanNode<Parameters>(graph, {
          id: `${id}-means`,
          featureCount,
          sums: sums!,
          valueCounts: weights ? undefined : valueCounts,
          weightSums: weights ? weightSums : undefined,
          means: means!
        })
      );
    }
    if (plan.needDensities) {
      nodes.push(
        createZonalStatisticsDensityNode<Parameters>(graph, {
          id: `${id}-densities`,
          featureCount,
          counts: counts!,
          areas: featureAreas!,
          densities: densities!
        })
      );
    }

    // 6. Extent.
    if (output.extent) {
      const statisticViews = {
        count: undefined,
        sum: sums,
        mean: means,
        minimum: minima,
        maximum: maxima,
        density: densities
      };
      nodes.push(
        ...getZonalStatisticsExtentNodes<Parameters>(graph, {
          id,
          featureCount,
          statistic: output.extentStatistic!,
          counts: counts!,
          statisticView: statisticViews[output.extentStatistic!],
          extent: output.extent
        })
      );
    }
    return nodes;
  }
}
