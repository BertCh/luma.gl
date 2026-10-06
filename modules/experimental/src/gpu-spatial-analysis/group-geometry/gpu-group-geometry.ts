// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSegmentedReduction,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPUGeographicDistribution} from '../geographic-distribution/index';
import {GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} from '../geographic-distribution/index';

const OPERATION = 'GPUGroupGeometry';
const MAXIMUM_ROWS = 2 ** 24 - 1;
const DEFAULT_MEDOID_MAXIMUM_GROUP_SIZE = 4096;

/** Value written to `medoidIndices` for an empty group or a group over the medoid size cap. */
export const GPU_GROUP_GEOMETRY_NO_MEDOID = 0xffffffff;

/**
 * Caller-owned outputs of {@link GPUGroupGeometry}. Which views are present is compile-time; at
 * least one is required. Every output is rewritten on every encoding and holds one row per group.
 * An empty group has `counts` 0, NaN in every float output and
 * {@link GPU_GROUP_GEOMETRY_NO_MEDOID} in `medoidIndices`.
 */
export type GPUGroupGeometryOutput = {
  /** Included rows per group (finite position and a label in `[0, groupCount)`). */
  counts?: GraphDataView<'uint32'>;
  /** Bounding box `[minX, minY, maxX, maxY]` per group, exact, in the caller's coordinates. */
  bounds?: GraphDataView<'float32x4'>;
  /** Unweighted mean center `[x, y]` per group. */
  meanCenters?: GraphDataView<'float32x2'>;
  /** Weighted mean center `[x, y]` per group. Requires `weights`. */
  weightedCenters?: GraphDataView<'float32x2'>;
  /** Sum of the positive finite weights per group. Requires `weights`. */
  weightSums?: GraphDataView<'float32'>;
  /**
   * Original row index of the medoid per group: the included row minimizing the summed Euclidean
   * distance to every included row of the group, lowest row index on ties. A group larger than
   * `medoidMaximumGroupSize` gets {@link GPU_GROUP_GEOMETRY_NO_MEDOID} and sets `overflow`.
   */
  medoidIndices?: GraphDataView<'uint32'>;
  /**
   * Standard deviational ellipse per group, 3 floats `[angle, sigmaX, sigmaY]` with the
   * `GPUGeographicDistribution` definition. Weighted when `weights` is present.
   */
  ellipses?: GraphDataView<'float32'>;
  /** Standard distance per group; weighted when `weights` is present. */
  standardDistances?: GraphDataView<'float32'>;
  /** One row. `1` when some group exceeded `medoidMaximumGroupSize` and has no medoid, else `0`. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGroupGeometry}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Topology
 * (needs a new graph): view lengths, `groupCount`, `noiseLabel`, `medoidMaximumGroupSize`, and
 * which inputs and outputs are present.
 */
export type GPUGroupGeometryProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'group-geometry'`. */
  id?: string;
  /**
   * Planar points (fewer than `2^24` rows). Rows with a non-finite coordinate are excluded.
   * Statistics are Euclidean, so project longitude and latitude first.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Label per row. Dense labels `0..groupCount-1` select a group. A label equal to `noiseLabel`
   * or `>= groupCount` (for example DBSCAN's `0xffffffff`) excludes the row from every output.
   */
  labels: GraphDataView<'uint32'>;
  /** Number of groups, the row count of every output. */
  groupCount: number;
  /**
   * Optional label that marks noise. It only matters when it is below `groupCount`; larger values
   * (such as `0xffffffff`) are already excluded.
   */
  noiseLabel?: number;
  /** Optional weights. A row with a non-finite or non-positive weight is skipped by weighted outputs. */
  weights?: GraphDataView<'float32'>;
  /**
   * Per-frame parameters in the `getGPUGeographicDistributionParameterValues` layout (local
   * origin, standard deviations and ellipse convention).
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Largest group for which the quadratic medoid search runs. Compile-time. Defaults to 4096.
   */
  medoidMaximumGroupSize?: number;
  /** Caller-owned outputs. At least one must be present. */
  output: GPUGroupGeometryOutput;
};

/**
 * Per-label geometry summaries over a `uint32` label column: bounds, mean and weighted center,
 * medoid, standard deviational ellipse, count. It is the "cluster, then describe it" step after
 * `GPUSpatialClustering` or any classifier.
 *
 * ## Composition
 *
 * Labels are validated into one group key per row (noise and out-of-range labels become the
 * excluded key `groupCount`). Rows are stably sorted by that key with `GPUSort`, offsets come from
 * `GPUScan`, bounds use `GPUSegmentedReduction`, and the centers and the ellipse come from
 * {@link GPUGeographicDistribution} run on the same keys (two instances when weights are given).
 *
 * ## Medoid
 *
 * One invocation per row sums the distance to every row of its group in sorted (ascending row)
 * order, then one invocation per group takes the first minimum, so ties go to the lowest row
 * index. Cost is quadratic in the group size, which is why groups above `medoidMaximumGroupSize`
 * are skipped and flagged rather than silently slow. Costs are float32; two rows whose exact costs
 * differ by less than float32 rounding may swap.
 *
 * ## Determinism
 *
 * No float atomics. Every sum is a fixed-order sorted segmented sum, bitwise reproducible on one
 * device. The contributor never compiles, encodes, submits or reads back.
 */
export class GPUGroupGeometry implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGroupGeometryProps;
  /** Number of groups. */
  readonly groupCount: number;
  /** Medoid group size cap. */
  readonly medoidMaximumGroupSize: number;

  constructor(props: GPUGroupGeometryProps) {
    this.id = props.id ?? 'group-geometry';
    this.props = props;
    const {id} = this;
    const {output} = props;
    for (const [name, view] of [
      ['positions', props.positions],
      ['labels', props.labels],
      ['weights', props.weights],
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
    if (rows > MAXIMUM_ROWS) {
      throw new Error(`${id} supports fewer than 2^24 rows`);
    }
    validatePackedUint32View(props.labels, `${id} labels`);
    if (props.labels.length !== rows) {
      throw new Error(`${id} labels length must equal positions length`);
    }
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
      if (props.weights.length !== rows) {
        throw new Error(`${id} weights length must equal positions length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} float32 values`
      );
    }
    this.groupCount = props.groupCount;
    if (
      !Number.isInteger(this.groupCount) ||
      this.groupCount < 1 ||
      this.groupCount > MAXIMUM_ROWS
    ) {
      throw new Error(`${id} groupCount must be a positive integer below 2^24`);
    }
    if (
      props.noiseLabel !== undefined &&
      (!Number.isInteger(props.noiseLabel) || props.noiseLabel < 0 || props.noiseLabel > 0xffffffff)
    ) {
      throw new Error(`${id} noiseLabel must be a uint32`);
    }
    this.medoidMaximumGroupSize = props.medoidMaximumGroupSize ?? DEFAULT_MEDOID_MAXIMUM_GROUP_SIZE;
    if (!Number.isInteger(this.medoidMaximumGroupSize) || this.medoidMaximumGroupSize < 1) {
      throw new Error(`${id} medoidMaximumGroupSize must be a positive integer`);
    }
    const groups = this.groupCount;
    const checks = [
      ['counts', output.counts, ['uint32'], groups],
      ['bounds', output.bounds, ['float32x4'], groups],
      ['meanCenters', output.meanCenters, ['float32x2'], groups],
      ['weightedCenters', output.weightedCenters, ['float32x2'], groups],
      ['weightSums', output.weightSums, ['float32'], groups],
      ['medoidIndices', output.medoidIndices, ['uint32'], groups],
      ['ellipses', output.ellipses, ['float32'], groups * 3],
      ['standardDistances', output.standardDistances, ['float32'], groups],
      ['overflow', output.overflow, ['uint32'], 1]
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
    if ((output.weightedCenters || output.weightSums) && !props.weights) {
      throw new Error(`${id} output.weightedCenters and output.weightSums require weights`);
    }
    if (output.overflow && !output.medoidIndices) {
      throw new Error(`${id} output.overflow requires output.medoidIndices`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      checks.map(([, view]) => view),
      [props.positions, props.labels, props.weights, props.parameters]
    );
  }

  /** Returns key, sort, bounds, geographic-distribution, medoid and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, groupCount, medoidMaximumGroupSize} = this;
    const {output, positions, labels, weights, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      labels,
      weights,
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
        declarations: `const GROUP_COUNT: u32 = ${groupCount}u;\n${declarations}`,
        body
      });
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Group keys: invalid rows get the excluded key GROUP_COUNT.
    const groupKeys = u32('group-keys', rows);
    const rowIndices = u32('row-indices', rows);
    const groupCounts = u32('group-counts', groupCount);
    const noiseCheck =
      props.noiseLabel !== undefined && props.noiseLabel < groupCount
        ? `valid = valid && label != ${props.noiseLabel}u;`
        : '';
    nodes.push(
      kernel(
        'keys',
        rows,
        [
          read('positions', positions, 'f32'),
          read('labels', labels, 'u32'),
          write('groupKeys', groupKeys, 'u32'),
          write('rowIndices', rowIndices, 'u32')
        ],
        `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let label = labels[labelsOffset + index];
  var valid = (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(y) & 0x7f800000u) != 0x7f800000u;
  valid = valid && label < GROUP_COUNT;
  ${noiseCheck}
  groupKeys[groupKeysOffset + index] = select(GROUP_COUNT, label, valid);
  rowIndices[rowIndicesOffset + index] = index;`
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
        [read('groupKeys', groupKeys, 'u32'), write('groupCounts', groupCounts, 'atomic<u32>')],
        `let group = groupKeys[groupKeysOffset + index];
  if (group < GROUP_COUNT) {
    atomicAdd(&groupCounts[groupCountsOffset + group], 1u);
  }`
      )
    );

    // 2. Stable sort by group, offsets.
    const sortedKeys = u32('sorted-keys', rows);
    const sortedIndices = u32('sorted-indices', rows);
    const segmentOffsets = u32('segment-offsets', groupCount + 1);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys: groupKeys,
        values: rowIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(groupCount)
      }).getCommandNodes(graph),
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

    // 3. Bounds from sorted coordinates.
    const needSorted = Boolean(output.bounds || output.medoidIndices);
    const sortedX = needSorted ? f32('sorted-x', rows) : undefined;
    const sortedY = needSorted ? f32('sorted-y', rows) : undefined;
    if (sortedX && sortedY) {
      nodes.push(
        kernel(
          'gather',
          rows,
          [
            read('sortedIndices', sortedIndices, 'u32'),
            read('positions', positions, 'f32'),
            write('sortedX', sortedX),
            write('sortedY', sortedY)
          ],
          `let row = sortedIndices[sortedIndicesOffset + index];
  sortedX[sortedXOffset + index] = positions[positionsOffset + row * 2u];
  sortedY[sortedYOffset + index] = positions[positionsOffset + row * 2u + 1u];`
        )
      );
    }
    if (output.bounds && sortedX && sortedY) {
      const reduce = (name: string, input: GraphDataView<'float32'>, operation: 'min' | 'max') => {
        const result = f32(`bounds-${name}`, groupCount);
        nodes.push(
          ...new GPUSegmentedReduction({
            id: `${id}-bounds-${name}`,
            input,
            segmentOffsets,
            output: result,
            operation
          }).getCommandNodes(graph)
        );
        return result;
      };
      const minX = reduce('min-x', sortedX, 'min');
      const minY = reduce('min-y', sortedY, 'min');
      const maxX = reduce('max-x', sortedX, 'max');
      const maxY = reduce('max-y', sortedY, 'max');
      nodes.push(
        kernel(
          'publish-bounds',
          groupCount,
          [
            read('counts', groupCounts, 'u32'),
            read('minX', minX, 'f32'),
            read('minY', minY, 'f32'),
            read('maxX', maxX, 'f32'),
            read('maxY', maxY, 'f32'),
            write('bounds', output.bounds)
          ],
          `let isEmpty = counts[countsOffset + index] == 0u;
  let nan = getQuietNaN(index);
  bounds[boundsOffset + index * 4u] = select(minX[minXOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 1u] = select(minY[minYOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 2u] = select(maxX[maxXOffset + index], nan, isEmpty);
  bounds[boundsOffset + index * 4u + 3u] = select(maxY[maxYOffset + index], nan, isEmpty);`,
          'fn getQuietNaN(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }'
        )
      );
    }
    if (output.counts) {
      nodes.push(
        kernel(
          'publish-counts',
          groupCount,
          [read('counts', groupCounts, 'u32'), write('countsOut', output.counts, 'u32')],
          'countsOut[countsOutOffset + index] = counts[countsOffset + index];'
        )
      );
    }

    // 4. Centers and ellipse through GPUGeographicDistribution on the same keys.
    const wantsEllipse = Boolean(output.ellipses || output.standardDistances);
    const needUnweightedGeography = Boolean(output.meanCenters || (wantsEllipse && !weights));
    if (needUnweightedGeography) {
      const meanCenters =
        output.meanCenters ??
        createTransientView(graph, `${id}-mean-centers`, 'float32x2', groupCount);
      nodes.push(
        ...new GPUGeographicDistribution({
          id: `${id}-unweighted`,
          positions,
          groupIds: groupKeys,
          groupCount,
          parameters,
          output: {
            meanCenters,
            ellipses: weights ? undefined : output.ellipses,
            standardDistances: weights ? undefined : output.standardDistances
          }
        }).getCommandNodes(graph)
      );
    }
    if (weights && (output.weightedCenters || output.weightSums || wantsEllipse)) {
      nodes.push(
        ...new GPUGeographicDistribution({
          id: `${id}-weighted`,
          positions,
          weights,
          groupIds: groupKeys,
          groupCount,
          parameters,
          output: {
            meanCenters:
              output.weightedCenters ??
              createTransientView(graph, `${id}-weighted-centers`, 'float32x2', groupCount),
            weightSums: output.weightSums,
            ellipses: output.ellipses,
            standardDistances: output.standardDistances
          }
        }).getCommandNodes(graph)
      );
    }

    // 5. Medoid: per-row summed distance, then per-group first minimum.
    if (output.medoidIndices && sortedX && sortedY) {
      const costs = f32('medoid-costs', rows);
      nodes.push(
        kernel(
          'medoid-cost',
          rows,
          [
            read('sortedKeys', sortedKeys, 'u32'),
            read('segmentOffsets', segmentOffsets, 'u32'),
            read('sortedX', sortedX, 'f32'),
            read('sortedY', sortedY, 'f32'),
            write('costs', costs)
          ],
          `let group = sortedKeys[sortedKeysOffset + index];
  var cost = 0.0;
  if (group < GROUP_COUNT) {
    let begin = segmentOffsets[segmentOffsetsOffset + group];
    let end = segmentOffsets[segmentOffsetsOffset + group + 1u];
    if (end - begin <= MEDOID_MAXIMUM_GROUP_SIZE) {
      let x = sortedX[sortedXOffset + index];
      let y = sortedY[sortedYOffset + index];
      for (var other = begin; other < end; other++) {
        let dx = sortedX[sortedXOffset + other] - x;
        let dy = sortedY[sortedYOffset + other] - y;
        cost += sqrt(dx * dx + dy * dy);
      }
    }
  }
  costs[costsOffset + index] = cost;`,
          `const MEDOID_MAXIMUM_GROUP_SIZE: u32 = ${medoidMaximumGroupSize}u;`
        )
      );
      if (output.overflow) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-zero-overflow`,
            operation: OPERATION,
            view: output.overflow,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
      const selectBindings = [
        read('segmentOffsets', segmentOffsets, 'u32'),
        read('costs', costs, 'f32'),
        read('sortedIndices', sortedIndices, 'u32'),
        write('medoids', output.medoidIndices, 'u32')
      ];
      if (output.overflow) {
        selectBindings.push(write('overflow', output.overflow, 'atomic<u32>'));
      }
      nodes.push(
        kernel(
          'medoid-select',
          groupCount,
          selectBindings,
          `let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  var best = NO_MEDOID;
  if (end - begin > MEDOID_MAXIMUM_GROUP_SIZE) {
    ${output.overflow ? 'atomicStore(&overflow[overflowOffset], 1u);' : ''}
  } else if (end > begin) {
    var bestCost = costs[costsOffset + begin];
    best = sortedIndices[sortedIndicesOffset + begin];
    for (var row = begin + 1u; row < end; row++) {
      let cost = costs[costsOffset + row];
      if (cost < bestCost) {
        bestCost = cost;
        best = sortedIndices[sortedIndicesOffset + row];
      }
    }
  }
  medoids[medoidsOffset + index] = best;`,
          `const MEDOID_MAXIMUM_GROUP_SIZE: u32 = ${medoidMaximumGroupSize}u;
const NO_MEDOID: u32 = ${GPU_GROUP_GEOMETRY_NO_MEDOID}u;`
        )
      );
    }
    return nodes;
  }
}
