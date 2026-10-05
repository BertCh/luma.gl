// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {createSegmentSumNode, getSortKeyBits} from '../../utils/sorted-segment-sums';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  GEOGRAPHIC_DISTRIBUTION_MAXIMUM_ROWS,
  GEOGRAPHIC_DISTRIBUTION_WGSL,
  WEISZFELD_COINCIDENT_DISTANCE_SQUARED
} from './geographic-distribution-kernels';
import {GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} from './geographic-distribution-parameters';

const OPERATION = 'GPUGeographicDistribution';
const MAXIMUM_MEDIAN_ITERATIONS = 256;
const DEFAULT_MEDIAN_ITERATIONS = 24;
const DEFAULT_POLYGON_VERTEX_COUNT = 64;

/**
 * Caller-owned outputs of {@link GPUGeographicDistribution}. Which views are present is
 * compile-time; at least one is required. Every output is rewritten on every encoding and holds
 * at least `groupCount * components` elements.
 *
 * An empty group (no included row, so a weight sum of 0) has `counts` 0, `weightSums` 0 and NaN in
 * every other output.
 */
export type GPUGeographicDistributionOutput = {
  /** Included rows per group. */
  counts?: GraphDataView<'uint32'>;
  /** Sum of the weights of the included rows per group. */
  weightSums?: GraphDataView<'float32'>;
  /** Weighted mean centre `[x, y]` per group, in the caller's planar coordinates. */
  meanCenters?: GraphDataView<'float32x2'>;
  /** Weighted geometric median `[x, y]` per group (Weiszfeld iteration). */
  medianCenters?: GraphDataView<'float32x2'>;
  /** `1` when the last Weiszfeld step moved at most `medianTolerance`, otherwise `0`. */
  medianConverged?: GraphDataView<'uint32'>;
  /**
   * Standard distance per group: `standardDeviations * sqrt(sum(w * |p - mean|^2) / sum(w))`,
   * the radius of the ArcGIS standard distance circle.
   */
  standardDistances?: GraphDataView<'float32'>;
  /**
   * Standard deviational ellipse per group, 3 floats `[angle, sigmaX, sigmaY]`. See
   * {@link GPUGeographicDistribution} for the exact definition. `sigmaY >= sigmaX`.
   */
  ellipses?: GraphDataView<'float32'>;
  /**
   * Linear directional mean per group, 3 floats `[meanAngle, circularVariance, meanLength]`.
   * Requires `lineEnds`.
   */
  directionalMeans?: GraphDataView<'float32'>;
  /**
   * Counter-clockwise ellipse ring of `polygonVertexCount` vertices per group (the ring is not
   * closed: the first vertex is not repeated). Empty groups are NaN.
   */
  ellipseVertices?: GraphDataView<'float32x2'>;
  /** Counter-clockwise standard-distance circle ring, laid out like `ellipseVertices`. */
  circleVertices?: GraphDataView<'float32x2'>;
};

/**
 * Properties for {@link GPUGeographicDistribution}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Topology
 * (needs a new graph): view lengths, `groupCount`, which inputs and outputs are present,
 * `medianIterations` and `polygonVertexCount`.
 */
export type GPUGeographicDistributionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geographic-distribution'`. */
  id?: string;
  /**
   * Planar points. The caller projects longitude/latitude (for example to Web Mercator or a local
   * equal-distance projection) because every statistic is Euclidean. Fewer than `2^24` rows.
   * Rows with a non-finite coordinate are excluded.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional non-negative weights. A row with a non-finite, negative or zero weight is excluded.
   * Without weights every row has weight 1.
   */
  weights?: GraphDataView<'float32'>;
  /** Optional group id per row. Ids `>= groupCount` exclude the row. Defaults to one group. */
  groupIds?: GraphDataView<'uint32'>;
  /** Number of groups (rows of every per-group output). Defaults to 1; must be 1 without `groupIds`. */
  groupCount?: number;
  /** Optional row mask; a zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional line ends. Row `i` is then also the line from `positions[i]` to `lineEnds[i]`, which
   * `directionalMeans` averages. Zero-length or non-finite lines are skipped by the directional
   * mean only.
   */
  lineEnds?: GraphDataView<'float32x2'>;
  /** Per-frame parameters packed by `getGPUGeographicDistributionParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Weiszfeld iterations of the median centre. Compile-time, 1 to 256. Defaults to 24. */
  medianIterations?: number;
  /** Vertices per ring of `ellipseVertices` and `circleVertices`, at least 3. Defaults to 64. */
  polygonVertexCount?: number;
  /** Caller-owned outputs. At least one must be present. */
  output: GPUGeographicDistributionOutput;
};

/**
 * Geographic distribution statistics (the ArcGIS "Measuring Geographic Distributions" family) for
 * planar points, per group or per masked selection, deterministic on the GPU.
 *
 * ## Definitions
 *
 * With included rows `i`, weights `w_i` (1 when unweighted), `W = sum(w_i)` and positions taken
 * relative to the local `origin`:
 *
 * - Mean centre `M = sum(w p) / W`.
 * - Standard distance `SD = standardDeviations * sqrt(sum(w |p - M|^2) / W)`. The divisor is `W`
 *   (the row count when unweighted), as ArcGIS divides by `n`, not `n - 1`.
 * - Covariance about the mean centre `Sxx = sum(w dx^2) / W`, `Syy = sum(w dy^2) / W`,
 *   `Sxy = sum(w dx dy) / W` with `d = p - M`, from two passes (no `E[x^2] - E[x]^2`).
 * - Standard deviational ellipse: principal variances `lambdaMax/Min = (Sxx + Syy) / 2 +/-
 *   sqrt(((Sxx - Syy) / 2)^2 + Sxy^2)`; semi-axes `standardDeviations * scale * sqrt(lambda)`
 *   with `scale = sqrt(2)` for the `'arcgis'` convention (ArcGIS Directional Distribution) and
 *   `1` for `'standard'`.
 * - Axis naming follows ArcGIS: the ellipse "y" axis is the long axis and the "x" axis is the
 *   short axis, so `sigmaY >= sigmaX`. ArcGIS reports the clockwise-from-north angle `theta` of
 *   the y axis; `ellipses[g * 3]` instead stores the counter-clockwise angle from `+x` of the x
 *   (short) axis, `angle = -theta` wrapped to `(-pi/2, pi/2]`. The long axis points at
 *   `angle + pi/2`. A vertex is `M + R(angle) * (sigmaX cos t, sigmaY sin t)`.
 * - Median centre: Weiszfeld iteration `m <- m + sum(w d / |d|) / sum(w / |d|)` with `d = p - m`
 *   started at the mean centre, `medianIterations` fixed steps; a point within `1e-6` of the
 *   iterate is skipped in that step.
 * - Linear directional mean: line angle `a = atan2(dy, dx)` counter-clockwise from `+x`;
 *   `R = |sum(w cos a, w sin a)| / Nd` over non-degenerate lines (`Nd` is their weight sum),
 *   mean angle `atan2(sum w sin a, sum w cos a)` in `(-pi, pi]`, circular variance `1 - R`, mean
 *   length `sum(w |line|) / Nd`. With `orientationOnly` the angles are doubled before averaging and
 *   the mean angle is halved, giving `(-pi/2, pi/2]`. The angle is NaN when `R` is exactly 0.
 *
 * ## Determinism
 *
 * No float atomics. Rows are stably sorted by group once; every sum (including each Weiszfeld
 * iteration) is a fixed-order segmented tree sum over the sorted rows, so results are bitwise
 * reproducible across encodings on one device. Sums differ from a sequential CPU sum only by
 * float32 rounding.
 *
 * ## Non-goals
 *
 * No central feature (argmin of summed distances), no geodesic distances, no per-feature
 * confidence ellipse beyond the standard deviation multiplier. The contributor never compiles, encodes,
 * submits, or reads back.
 */
export class GPUGeographicDistribution implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeographicDistributionProps;
  /** Number of groups, the row count of every per-group output. */
  readonly groupCount: number;
  /** Weiszfeld iteration count. */
  readonly medianIterations: number;
  /** Vertices per ring. */
  readonly polygonVertexCount: number;

  constructor(props: GPUGeographicDistributionProps) {
    this.id = props.id ?? 'geographic-distribution';
    this.props = props;
    const {id} = this;
    const {output} = props;
    for (const [name, view] of [
      ['positions', props.positions],
      ['weights', props.weights],
      ['groupIds', props.groupIds],
      ['mask', props.mask],
      ['lineEnds', props.lineEnds],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (rows > GEOGRAPHIC_DISTRIBUTION_MAXIMUM_ROWS) {
      throw new Error(`${id} supports fewer than 2^24 rows`);
    }
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
    }
    if (props.groupIds) {
      validatePackedUint32View(props.groupIds, `${id} groupIds`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
    }
    if (props.lineEnds) {
      validatePackedView(props.lineEnds, ['float32x2'], `${id} lineEnds`);
    }
    for (const [name, view] of [
      ['weights', props.weights],
      ['groupIds', props.groupIds],
      ['mask', props.mask],
      ['lineEnds', props.lineEnds]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal positions length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} float32 values`
      );
    }
    this.groupCount = props.groupCount ?? 1;
    if (
      !Number.isInteger(this.groupCount) ||
      this.groupCount < 1 ||
      this.groupCount > GEOGRAPHIC_DISTRIBUTION_MAXIMUM_ROWS
    ) {
      throw new Error(`${id} groupCount must be a positive integer below 2^24`);
    }
    if (!props.groupIds && this.groupCount !== 1) {
      throw new Error(`${id} groupCount must be 1 without groupIds`);
    }
    this.medianIterations = props.medianIterations ?? DEFAULT_MEDIAN_ITERATIONS;
    if (
      !Number.isInteger(this.medianIterations) ||
      this.medianIterations < 1 ||
      this.medianIterations > MAXIMUM_MEDIAN_ITERATIONS
    ) {
      throw new Error(
        `${id} medianIterations must be an integer in [1, ${MAXIMUM_MEDIAN_ITERATIONS}]`
      );
    }
    this.polygonVertexCount = props.polygonVertexCount ?? DEFAULT_POLYGON_VERTEX_COUNT;
    if (!Number.isInteger(this.polygonVertexCount) || this.polygonVertexCount < 3) {
      throw new Error(`${id} polygonVertexCount must be an integer of at least 3`);
    }

    const groups = this.groupCount;
    const vertices = groups * this.polygonVertexCount;
    const checks = [
      ['counts', output.counts, ['uint32'], groups],
      ['weightSums', output.weightSums, ['float32'], groups],
      ['meanCenters', output.meanCenters, ['float32x2'], groups],
      ['medianCenters', output.medianCenters, ['float32x2'], groups],
      ['medianConverged', output.medianConverged, ['uint32'], groups],
      ['standardDistances', output.standardDistances, ['float32'], groups],
      ['ellipses', output.ellipses, ['float32'], groups * 3],
      ['directionalMeans', output.directionalMeans, ['float32'], groups * 3],
      ['ellipseVertices', output.ellipseVertices, ['float32x2'], vertices],
      ['circleVertices', output.circleVertices, ['float32x2'], vertices]
    ] as const;
    let outputCount = 0;
    for (const [name, view, formats, length] of checks) {
      if (!view) {
        continue;
      }
      outputCount++;
      validatePackedView(view, formats, `${id} output.${name}`);
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} rows`);
      }
    }
    if (outputCount === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    if (output.directionalMeans && !props.lineEnds) {
      throw new Error(`${id} output.directionalMeans requires lineEnds`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      checks.map(([, view]) => view),
      [props.positions, props.weights, props.groupIds, props.mask, props.lineEnds, props.parameters]
    );
  }

  /**
   * Returns prepare, count, sort, scan, gather, moment, median, directional and publish nodes in
   * order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, groupCount, polygonVertexCount} = this;
    const {output, positions, weights, groupIds, mask, lineEnds, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      weights,
      groupIds,
      mask,
      lineEnds,
      parameters,
      ...Object.values(output)
    ]);
    const rows = positions.length;
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32' | 'atomic<u32>' = 'f32'
    ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
    const kernel = (
      step: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount,
        declarations: `${GEOGRAPHIC_DISTRIBUTION_WGSL}\n${declarations}`,
        body
      });
    const groupConstants = `const GROUP_COUNT: u32 = ${groupCount}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Row validity, sort keys and weights, then per-group counts.
    const sortKeys = u32('sort-keys', rows);
    const sortIndices = u32('sort-indices', rows);
    const rowWeights = f32('row-weights', rows);
    const groupCounts = u32('group-counts', groupCount);
    const prepareBindings = [read('positions', positions, 'f32')];
    if (weights) {
      prepareBindings.push(read('weights', weights, 'f32'));
    }
    if (groupIds) {
      prepareBindings.push(read('groupIds', groupIds, 'u32'));
    }
    if (mask) {
      prepareBindings.push(read('mask', mask, 'u32'));
    }
    prepareBindings.push(
      write('sortKeys', sortKeys, 'u32'),
      write('sortIndices', sortIndices, 'u32'),
      write('rowWeights', rowWeights)
    );
    nodes.push(
      kernel(
        'prepare',
        rows,
        prepareBindings,
        `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var valid = isFiniteValue(x) && isFiniteValue(y);
  var weight = 1.0;
  ${weights ? 'weight = weights[weightsOffset + index]; valid = valid && isFiniteValue(weight) && weight > 0.0;' : ''}
  var group = 0u;
  ${groupIds ? 'group = groupIds[groupIdsOffset + index]; valid = valid && group < GROUP_COUNT;' : ''}
  ${mask ? 'valid = valid && mask[maskOffset + index] != 0u;' : ''}
  sortKeys[sortKeysOffset + index] = select(GROUP_COUNT, group, valid);
  sortIndices[sortIndicesOffset + index] = index;
  rowWeights[rowWeightsOffset + index] = select(0.0, weight, valid);`,
        groupConstants
      ),
      createFillNode<Parameters>(graph, {
        id: `${id}-zero-counts`,
        operation: OPERATION,
        view: groupCounts,
        type: 'u32',
        value: '0u'
      }),
      kernel(
        'count',
        rows,
        [read('sortKeys', sortKeys, 'u32'), write('groupCounts', groupCounts, 'atomic<u32>')],
        `let group = sortKeys[sortKeysOffset + index];
  if (group < GROUP_COUNT) {
    atomicAdd(&groupCounts[groupCountsOffset + group], 1u);
  }`,
        groupConstants
      )
    );

    // 2. Stable sort by group and fixed segment offsets.
    const sortedKeys = u32('sorted-keys', rows);
    const sortedIndices = u32('sorted-indices', rows);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(groupCount)
      }).getCommandNodes(graph)
    );
    const segmentOffsets = u32('segment-offsets', groupCount + 1);
    nodes.push(
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: groupCounts,
        output: segmentOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      kernel(
        'segment-total',
        1,
        [read('counts', groupCounts, 'u32'), write('segmentOffsets', segmentOffsets, 'u32')],
        `segmentOffsets[segmentOffsetsOffset + ${groupCount}u] =
    segmentOffsets[segmentOffsetsOffset + ${groupCount - 1}u] + counts[countsOffset + ${groupCount - 1}u];`
      )
    );
    const sum = (name: string, input: GraphDataView<'float32'>, result: GraphDataView<'float32'>) =>
      createSegmentSumNode<Parameters>(graph, {
        id: `${id}-sum-${name}`,
        operation: OPERATION,
        segmentCount: groupCount,
        input,
        segmentOffsets,
        output: result
      });

    // 3. Sorted local coordinates, then the weighted mean centre.
    const sortedX = f32('sorted-x', rows);
    const sortedY = f32('sorted-y', rows);
    const sortedWeights = f32('sorted-weights', rows);
    const weightedX = f32('weighted-x', rows);
    const weightedY = f32('weighted-y', rows);
    const weightSums = f32('weight-sums', groupCount);
    const sumX = f32('sum-x', groupCount);
    const sumY = f32('sum-y', groupCount);
    const meanCenters = createTransientView(graph, `${id}-mean`, 'float32x2', groupCount);
    nodes.push(
      kernel(
        'gather',
        rows,
        [
          read('sortedIndices', sortedIndices, 'u32'),
          read('positions', positions, 'f32'),
          read('rowWeights', rowWeights, 'f32'),
          read('parameters', parameters, 'f32'),
          write('sortedX', sortedX),
          write('sortedY', sortedY),
          write('sortedWeights', sortedWeights)
        ],
        `let row = sortedIndices[sortedIndicesOffset + index];
  let weight = rowWeights[rowWeightsOffset + row];
  let isIncluded = weight > 0.0;
  sortedX[sortedXOffset + index] = select(0.0, positions[positionsOffset + row * 2u] - parameters[parametersOffset], isIncluded);
  sortedY[sortedYOffset + index] = select(0.0, positions[positionsOffset + row * 2u + 1u] - parameters[parametersOffset + 1u], isIncluded);
  sortedWeights[sortedWeightsOffset + index] = weight;`
      ),
      kernel(
        'weighted-coordinates',
        rows,
        [
          read('sortedX', sortedX, 'f32'),
          read('sortedY', sortedY, 'f32'),
          read('sortedWeights', sortedWeights, 'f32'),
          write('weightedX', weightedX),
          write('weightedY', weightedY)
        ],
        `let weight = sortedWeights[sortedWeightsOffset + index];
  weightedX[weightedXOffset + index] = weight * sortedX[sortedXOffset + index];
  weightedY[weightedYOffset + index] = weight * sortedY[sortedYOffset + index];`
      ),
      sum('weight', sortedWeights, weightSums),
      sum('x', weightedX, sumX),
      sum('y', weightedY, sumY),
      kernel(
        'mean',
        groupCount,
        [
          read('weightSums', weightSums, 'f32'),
          read('sumX', sumX, 'f32'),
          read('sumY', sumY, 'f32'),
          write('mean', meanCenters)
        ],
        `let total = weightSums[weightSumsOffset + index];
  let isEmpty = !(total > 0.0);
  mean[meanOffset + index * 2u] = select(sumX[sumXOffset + index] / total, getQuietNaN(index), isEmpty);
  mean[meanOffset + index * 2u + 1u] = select(sumY[sumYOffset + index] / total, getQuietNaN(index), isEmpty);`
      )
    );

    // 4. Central second moments about the mean centre (two passes).
    const needSpread =
      output.standardDistances ||
      output.ellipses ||
      output.ellipseVertices ||
      output.circleVertices;
    const standardDistances =
      output.standardDistances ?? (needSpread ? f32('standard-distances', groupCount) : undefined);
    const ellipses =
      output.ellipses ?? (output.ellipseVertices ? f32('ellipses', groupCount * 3) : undefined);
    if (needSpread) {
      const contributionXX = f32('contribution-xx', rows);
      const contributionYY = f32('contribution-yy', rows);
      const contributionXY = f32('contribution-xy', rows);
      const sumXX = f32('sum-xx', groupCount);
      const sumYY = f32('sum-yy', groupCount);
      const sumXY = f32('sum-xy', groupCount);
      nodes.push(
        kernel(
          'moments-diagonal',
          rows,
          [
            read('sortedKeys', sortedKeys, 'u32'),
            read('sortedX', sortedX, 'f32'),
            read('sortedY', sortedY, 'f32'),
            read('sortedWeights', sortedWeights, 'f32'),
            read('mean', meanCenters, 'f32'),
            write('contributionXX', contributionXX),
            write('contributionYY', contributionYY)
          ],
          `let group = sortedKeys[sortedKeysOffset + index];
  var xx = 0.0;
  var yy = 0.0;
  if (group < GROUP_COUNT) {
    let weight = sortedWeights[sortedWeightsOffset + index];
    let dx = sortedX[sortedXOffset + index] - mean[meanOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - mean[meanOffset + group * 2u + 1u];
    xx = weight * dx * dx;
    yy = weight * dy * dy;
  }
  contributionXX[contributionXXOffset + index] = xx;
  contributionYY[contributionYYOffset + index] = yy;`,
          groupConstants
        ),
        kernel(
          'moments-cross',
          rows,
          [
            read('sortedKeys', sortedKeys, 'u32'),
            read('sortedX', sortedX, 'f32'),
            read('sortedY', sortedY, 'f32'),
            read('sortedWeights', sortedWeights, 'f32'),
            read('mean', meanCenters, 'f32'),
            write('contributionXY', contributionXY)
          ],
          `let group = sortedKeys[sortedKeysOffset + index];
  var xy = 0.0;
  if (group < GROUP_COUNT) {
    let weight = sortedWeights[sortedWeightsOffset + index];
    let dx = sortedX[sortedXOffset + index] - mean[meanOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - mean[meanOffset + group * 2u + 1u];
    xy = weight * dx * dy;
  }
  contributionXY[contributionXYOffset + index] = xy;`,
          groupConstants
        ),
        sum('xx', contributionXX, sumXX),
        sum('yy', contributionYY, sumYY),
        sum('xy', contributionXY, sumXY)
      );
      const finalizeBindings = [
        read('weightSums', weightSums, 'f32'),
        read('sumXX', sumXX, 'f32'),
        read('sumYY', sumYY, 'f32'),
        read('sumXY', sumXY, 'f32'),
        read('parameters', parameters, 'f32')
      ];
      if (standardDistances) {
        finalizeBindings.push(write('standardDistances', standardDistances));
      }
      if (ellipses) {
        finalizeBindings.push(write('ellipses', ellipses));
      }
      nodes.push(
        kernel(
          'spread',
          groupCount,
          finalizeBindings,
          `let total = weightSums[weightSumsOffset + index];
  let isEmpty = !(total > 0.0);
  let varianceX = sumXX[sumXXOffset + index] / total;
  let varianceY = sumYY[sumYYOffset + index] / total;
  let covariance = sumXY[sumXYOffset + index] / total;
  let multiplier = parameters[parametersOffset + 2u];
  let scale = multiplier * parameters[parametersOffset + 3u];
  ${
    standardDistances
      ? 'standardDistances[standardDistancesOffset + index] = select(multiplier * sqrt(max(varianceX + varianceY, 0.0)), getQuietNaN(index), isEmpty);'
      : ''
  }
  ${
    ellipses
      ? `let halfSum = 0.5 * (varianceX + varianceY);
  let halfDifference = 0.5 * (varianceX - varianceY);
  let radius = sqrt(halfDifference * halfDifference + covariance * covariance);
  var majorAngle = 0.0;
  if (radius > 0.0) {
    majorAngle = 0.5 * atan2(covariance, halfDifference);
  }
  var angle = majorAngle + HALF_PI;
  if (angle > HALF_PI) {
    angle = angle - PI;
  }
  let nan = getQuietNaN(index);
  ellipses[ellipsesOffset + index * 3u] = select(angle, nan, isEmpty);
  ellipses[ellipsesOffset + index * 3u + 1u] = select(scale * sqrt(max(halfSum - radius, 0.0)), nan, isEmpty);
  ellipses[ellipsesOffset + index * 3u + 2u] = select(scale * sqrt(max(halfSum + radius, 0.0)), nan, isEmpty);`
      : ''
  }`
        )
      );
    }

    // 5. Weiszfeld median centre, reusing the sorted order every iteration.
    if (output.medianCenters || output.medianConverged) {
      const median = createTransientView(graph, `${id}-median`, 'float32x2', groupCount);
      const lastMoves = f32('median-moves', groupCount);
      const contributionWeight = f32('median-contribution-weight', rows);
      const contributionX = f32('median-contribution-x', rows);
      const contributionY = f32('median-contribution-y', rows);
      const sumMedianWeight = f32('median-sum-weight', groupCount);
      const sumMedianX = f32('median-sum-x', groupCount);
      const sumMedianY = f32('median-sum-y', groupCount);
      const coincidentDistanceSquared = getWGSLFloatLiteral(WEISZFELD_COINCIDENT_DISTANCE_SQUARED);
      nodes.push(
        kernel(
          'median-start',
          groupCount,
          [
            read('mean', meanCenters, 'f32'),
            write('median', median),
            write('lastMoves', lastMoves)
          ],
          `median[medianOffset + index * 2u] = mean[meanOffset + index * 2u];
  median[medianOffset + index * 2u + 1u] = mean[meanOffset + index * 2u + 1u];
  lastMoves[lastMovesOffset + index] = 0.0;`
        )
      );
      for (let iteration = 0; iteration < this.medianIterations; iteration++) {
        const step = `median-${iteration}`;
        nodes.push(
          kernel(
            `${step}-x`,
            rows,
            [
              read('sortedKeys', sortedKeys, 'u32'),
              read('sortedX', sortedX, 'f32'),
              read('sortedY', sortedY, 'f32'),
              read('sortedWeights', sortedWeights, 'f32'),
              read('median', median, 'f32'),
              write('contributionWeight', contributionWeight),
              write('contributionX', contributionX)
            ],
            `let group = sortedKeys[sortedKeysOffset + index];
  var inverseWeight = 0.0;
  var deltaX = 0.0;
  if (group < GROUP_COUNT) {
    let dx = sortedX[sortedXOffset + index] - median[medianOffset + group * 2u];
    let dy = sortedY[sortedYOffset + index] - median[medianOffset + group * 2u + 1u];
    let distanceSquared = dx * dx + dy * dy;
    if (distanceSquared > ${coincidentDistanceSquared}) {
      inverseWeight = sortedWeights[sortedWeightsOffset + index] / sqrt(distanceSquared);
      deltaX = inverseWeight * dx;
    }
  }
  contributionWeight[contributionWeightOffset + index] = inverseWeight;
  contributionX[contributionXOffset + index] = deltaX;`,
            groupConstants
          ),
          kernel(
            `${step}-y`,
            rows,
            [
              read('sortedKeys', sortedKeys, 'u32'),
              read('sortedY', sortedY, 'f32'),
              read('contributionWeight', contributionWeight, 'f32'),
              read('median', median, 'f32'),
              write('contributionY', contributionY)
            ],
            `let group = sortedKeys[sortedKeysOffset + index];
  var deltaY = 0.0;
  if (group < GROUP_COUNT) {
    deltaY = contributionWeight[contributionWeightOffset + index] * (sortedY[sortedYOffset + index] - median[medianOffset + group * 2u + 1u]);
  }
  contributionY[contributionYOffset + index] = deltaY;`,
            groupConstants
          ),
          sum(`${step}-weight`, contributionWeight, sumMedianWeight),
          sum(`${step}-x`, contributionX, sumMedianX),
          sum(`${step}-y`, contributionY, sumMedianY),
          kernel(
            `${step}-update`,
            groupCount,
            [
              read('sumWeight', sumMedianWeight, 'f32'),
              read('sumX', sumMedianX, 'f32'),
              read('sumY', sumMedianY, 'f32'),
              write('median', median),
              write('lastMoves', lastMoves)
            ],
            `let total = sumWeight[sumWeightOffset + index];
  var stepX = 0.0;
  var stepY = 0.0;
  if (total > 0.0) {
    stepX = sumX[sumXOffset + index] / total;
    stepY = sumY[sumYOffset + index] / total;
  }
  median[medianOffset + index * 2u] = median[medianOffset + index * 2u] + stepX;
  median[medianOffset + index * 2u + 1u] = median[medianOffset + index * 2u + 1u] + stepY;
  lastMoves[lastMovesOffset + index] = sqrt(stepX * stepX + stepY * stepY);`
          )
        );
      }
      const medianBindings = [
        read('median', median, 'f32'),
        read('lastMoves', lastMoves, 'f32'),
        read('parameters', parameters, 'f32')
      ];
      if (output.medianCenters) {
        medianBindings.push(write('medianCenters', output.medianCenters));
      }
      if (output.medianConverged) {
        medianBindings.push(write('medianConverged', output.medianConverged, 'u32'));
      }
      nodes.push(
        kernel(
          'median-publish',
          groupCount,
          medianBindings,
          `${
            output.medianCenters
              ? `medianCenters[medianCentersOffset + index * 2u] = median[medianOffset + index * 2u] + parameters[parametersOffset];
  medianCenters[medianCentersOffset + index * 2u + 1u] = median[medianOffset + index * 2u + 1u] + parameters[parametersOffset + 1u];`
              : ''
          }
  ${
    output.medianConverged
      ? `let isValid = isFiniteValue(median[medianOffset + index * 2u]);
  medianConverged[medianConvergedOffset + index] = select(0u, 1u, isValid && lastMoves[lastMovesOffset + index] <= parameters[parametersOffset + 5u]);`
      : ''
  }`
        )
      );
    }

    // 6. Linear directional mean.
    if (output.directionalMeans && lineEnds) {
      const contributionCosine = f32('direction-contribution-cosine', rows);
      const contributionSine = f32('direction-contribution-sine', rows);
      const contributionLength = f32('direction-contribution-length', rows);
      const contributionWeight = f32('direction-contribution-weight', rows);
      const sumCosine = f32('direction-sum-cosine', groupCount);
      const sumSine = f32('direction-sum-sine', groupCount);
      const sumLength = f32('direction-sum-length', groupCount);
      const sumWeight = f32('direction-sum-weight', groupCount);
      const lineHelpers = `
fn getLineWeight(weight: f32, dx: f32, dy: f32) -> f32 {
  let lengthSquared = dx * dx + dy * dy;
  let isValid = weight > 0.0 && isFiniteValue(dx) && isFiniteValue(dy) && isFiniteValue(lengthSquared) && lengthSquared > 0.0;
  return select(0.0, weight, isValid);
}`;
      const lineBindings = (outputs: WGSLKernelBinding[]) => [
        read('sortedIndices', sortedIndices, 'u32'),
        read('positions', positions, 'f32'),
        read('lineEnds', lineEnds, 'f32'),
        read('rowWeights', rowWeights, 'f32'),
        ...outputs
      ];
      const lineSetup = `let row = sortedIndices[sortedIndicesOffset + index];
  let dx = lineEnds[lineEndsOffset + row * 2u] - positions[positionsOffset + row * 2u];
  let dy = lineEnds[lineEndsOffset + row * 2u + 1u] - positions[positionsOffset + row * 2u + 1u];
  let weight = getLineWeight(rowWeights[rowWeightsOffset + row], dx, dy);`;
      nodes.push(
        kernel(
          'direction-angles',
          rows,
          lineBindings([
            read('parameters', parameters, 'f32'),
            write('contributionCosine', contributionCosine),
            write('contributionSine', contributionSine)
          ]),
          `${lineSetup}
  var cosine = 0.0;
  var sine = 0.0;
  if (weight > 0.0) {
    let lengthSquared = dx * dx + dy * dy;
    if (parameters[parametersOffset + 4u] > 0.5) {
      cosine = (dx * dx - dy * dy) / lengthSquared;
      sine = 2.0 * dx * dy / lengthSquared;
    } else {
      let lineLength = sqrt(lengthSquared);
      cosine = dx / lineLength;
      sine = dy / lineLength;
    }
  }
  contributionCosine[contributionCosineOffset + index] = weight * cosine;
  contributionSine[contributionSineOffset + index] = weight * sine;`,
          lineHelpers
        ),
        kernel(
          'direction-lengths',
          rows,
          lineBindings([
            write('contributionLength', contributionLength),
            write('contributionWeight', contributionWeight)
          ]),
          `${lineSetup}
  contributionLength[contributionLengthOffset + index] = weight * sqrt(dx * dx + dy * dy) * select(0.0, 1.0, weight > 0.0);
  contributionWeight[contributionWeightOffset + index] = weight;`,
          lineHelpers
        ),
        sum('direction-cosine', contributionCosine, sumCosine),
        sum('direction-sine', contributionSine, sumSine),
        sum('direction-length', contributionLength, sumLength),
        sum('direction-weight', contributionWeight, sumWeight),
        kernel(
          'direction-publish',
          groupCount,
          [
            read('sumCosine', sumCosine, 'f32'),
            read('sumSine', sumSine, 'f32'),
            read('sumLength', sumLength, 'f32'),
            read('sumWeight', sumWeight, 'f32'),
            read('parameters', parameters, 'f32'),
            write('directionalMeans', output.directionalMeans)
          ],
          `let total = sumWeight[sumWeightOffset + index];
  let isEmpty = !(total > 0.0);
  let cosine = sumCosine[sumCosineOffset + index];
  let sine = sumSine[sumSineOffset + index];
  let resultant = min(sqrt(cosine * cosine + sine * sine) / total, 1.0);
  let hasAngle = cosine != 0.0 || sine != 0.0;
  var angle = atan2(sine, cosine);
  if (parameters[parametersOffset + 4u] > 0.5) {
    angle = 0.5 * angle;
  }
  let nan = getQuietNaN(index);
  directionalMeans[directionalMeansOffset + index * 3u] = select(angle, nan, isEmpty || !hasAngle);
  directionalMeans[directionalMeansOffset + index * 3u + 1u] = select(1.0 - resultant, nan, isEmpty);
  directionalMeans[directionalMeansOffset + index * 3u + 2u] = select(sumLength[sumLengthOffset + index] / total, nan, isEmpty);`
        )
      );
    }

    // 7. Published per-group columns and polygon rings.
    if (output.counts || output.weightSums || output.meanCenters) {
      const bindings = [
        read('groupCounts', groupCounts, 'u32'),
        read('weightSums', weightSums, 'f32'),
        read('mean', meanCenters, 'f32'),
        read('parameters', parameters, 'f32')
      ];
      if (output.counts) {
        bindings.push(write('countsOut', output.counts, 'u32'));
      }
      if (output.weightSums) {
        bindings.push(write('weightSumsOut', output.weightSums));
      }
      if (output.meanCenters) {
        bindings.push(write('meanCentersOut', output.meanCenters));
      }
      nodes.push(
        kernel(
          'publish',
          groupCount,
          bindings,
          `${output.counts ? 'countsOut[countsOutOffset + index] = groupCounts[groupCountsOffset + index];' : ''}
  ${output.weightSums ? 'weightSumsOut[weightSumsOutOffset + index] = weightSums[weightSumsOffset + index];' : ''}
  ${
    output.meanCenters
      ? `meanCentersOut[meanCentersOutOffset + index * 2u] = mean[meanOffset + index * 2u] + parameters[parametersOffset];
  meanCentersOut[meanCentersOutOffset + index * 2u + 1u] = mean[meanOffset + index * 2u + 1u] + parameters[parametersOffset + 1u];`
      : ''
  }`
        )
      );
    }
    const vertexConstants = `const VERTEX_COUNT: u32 = ${polygonVertexCount}u;`;
    if (output.ellipseVertices && ellipses) {
      nodes.push(
        kernel(
          'ellipse-vertices',
          groupCount * polygonVertexCount,
          [
            read('ellipses', ellipses, 'f32'),
            read('mean', meanCenters, 'f32'),
            read('parameters', parameters, 'f32'),
            write('vertices', output.ellipseVertices)
          ],
          `let group = index / VERTEX_COUNT;
  let t = TWO_PI * f32(index % VERTEX_COUNT) / f32(VERTEX_COUNT);
  let angle = ellipses[ellipsesOffset + group * 3u];
  let localX = ellipses[ellipsesOffset + group * 3u + 1u] * cos(t);
  let localY = ellipses[ellipsesOffset + group * 3u + 2u] * sin(t);
  vertices[verticesOffset + index * 2u] = mean[meanOffset + group * 2u] + parameters[parametersOffset] + localX * cos(angle) - localY * sin(angle);
  vertices[verticesOffset + index * 2u + 1u] = mean[meanOffset + group * 2u + 1u] + parameters[parametersOffset + 1u] + localX * sin(angle) + localY * cos(angle);`,
          vertexConstants
        )
      );
    }
    if (output.circleVertices && standardDistances) {
      nodes.push(
        kernel(
          'circle-vertices',
          groupCount * polygonVertexCount,
          [
            read('standardDistances', standardDistances, 'f32'),
            read('mean', meanCenters, 'f32'),
            read('parameters', parameters, 'f32'),
            write('vertices', output.circleVertices)
          ],
          `let group = index / VERTEX_COUNT;
  let t = TWO_PI * f32(index % VERTEX_COUNT) / f32(VERTEX_COUNT);
  let radius = standardDistances[standardDistancesOffset + group];
  vertices[verticesOffset + index * 2u] = mean[meanOffset + group * 2u] + parameters[parametersOffset] + radius * cos(t);
  vertices[verticesOffset + index * 2u + 1u] = mean[meanOffset + group * 2u + 1u] + parameters[parametersOffset + 1u] + radius * sin(t);`,
          vertexConstants
        )
      );
    }
    return nodes;
  }
}
