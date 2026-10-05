// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode} from './map-graph-kernels';

const SEGMENT_WORKGROUP_SIZE = 256;

/** Returns the number of key bits that hold values in `[0, maximumKey]` (at least one). @internal */
export function getMapGraphSortKeyBits(maximumKey: number): number {
  return Math.max(1, Math.floor(maximumKey).toString(2).length);
}

/**
 * One deterministic per-segment sum computed by {@link getMapGraphSortedSumNodes}.
 *
 * @internal
 */
export type MapGraphSortedReduction = {
  /** Short name used in generated node and transient IDs. Must be unique per call. */
  name: string;
  /** Per-row contributions aligned with `segmentKeys`; invalid rows contribute 0. */
  contributions: GraphDataView<'float32'>;
  /** Per-segment sums (`segmentCount` rows). */
  output: GraphDataView<'float32'>;
};

/**
 * Shared sorted segmented reduction used by zonal statistics, raster zonal statistics, flow
 * aggregation and spatial clustering. Reduces per-segment sums in a fixed order: stable sort by
 * segment key, exclusive scan of the counts into segment offsets, gather of contributions in sorted
 * order, and one segmented sum, so the results are bitwise reproducible.
 *
 * `segmentKeys` holds a segment key per row; keys at or above `segmentCount` sort last and never
 * contribute. `segmentCounts[s]` must equal the number of rows with key `s`, so that the scanned
 * offsets delimit each segment in the sorted order.
 *
 * @internal
 */
export function getMapGraphSortedSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Operation name reported in workload estimates. */
    operation: string;
    segmentCount: number;
    segmentKeys: GraphDataView<'uint32'>;
    segmentCounts: GraphDataView<'uint32'>;
    sumContributions?: GraphDataView<'float32'>;
    weightContributions?: GraphDataView<'float32'>;
    sums?: GraphDataView<'float32'>;
    weightSums?: GraphDataView<'float32'>;
    /** Additional named reductions sharing the same sort and segment offsets. */
    reductions?: readonly MapGraphSortedReduction[];
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, segmentCount, segmentKeys, segmentCounts} = props;
  const pointCount = segmentKeys.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', pointCount);
  const sortIndices = createTransientView(graph, `${id}-sort-indices`, 'uint32', pointCount);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', pointCount);
  const sortedIndices = createTransientView(graph, `${id}-sorted-indices`, 'uint32', pointCount);
  // Unassigned and out-of-range rows share the key `segmentCount`, which sorts after every segment.
  nodes.push(
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-sort-prepare`,
      operation,
      variant: 'sort-prepare',
      bindings: [
        {name: 'pointFeatureRows', view: segmentKeys, type: 'u32', access: 'read'},
        {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
        {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
      ],
      invocationCount: pointCount,
      declarations: `const FEATURE_COUNT: u32 = ${segmentCount}u;`,
      body: `sortKeys[sortKeysOffset + index] = min(pointFeatureRows[pointFeatureRowsOffset + index], FEATURE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`
    })
  );
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort`,
      keys: sortKeys,
      values: sortIndices,
      outputKeys: sortedKeys,
      outputValues: sortedIndices,
      keyBits: getMapGraphSortKeyBits(segmentCount)
    }).getCommandNodes(graph)
  );

  const segmentOffsets = createTransientView(
    graph,
    `${id}-segment-offsets`,
    'uint32',
    segmentCount + 1
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-segment-scan`,
      input: segmentCounts,
      output: segmentOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-segment-total`,
      operation,
      variant: 'segment-total',
      bindings: [
        {name: 'counts', view: segmentCounts, type: 'u32', access: 'read'},
        {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `const LAST_FEATURE: u32 = ${segmentCount - 1}u;`,
      body: `segmentOffsets[segmentOffsetsOffset + LAST_FEATURE + 1u] =
    segmentOffsets[segmentOffsetsOffset + LAST_FEATURE] + counts[countsOffset + LAST_FEATURE];`
    })
  );

  const reductions: MapGraphSortedReduction[] = [];
  if (props.sums && props.sumContributions) {
    reductions.push({name: 'sums', contributions: props.sumContributions, output: props.sums});
  }
  if (props.weightSums && props.weightContributions) {
    reductions.push({
      name: 'weight-sums',
      contributions: props.weightContributions,
      output: props.weightSums
    });
  }
  reductions.push(...(props.reductions ?? []));
  for (const reduction of reductions) {
    const sortedContributions = createTransientView(
      graph,
      `${id}-sorted-${reduction.name}`,
      'float32',
      pointCount
    );
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-gather-${reduction.name}`,
        operation,
        variant: 'gather',
        bindings: [
          {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
          {name: 'contributions', view: reduction.contributions, type: 'f32', access: 'read'},
          {
            name: 'sortedContributions',
            view: sortedContributions,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: pointCount,
        body: `sortedContributions[sortedContributionsOffset + index] =
    contributions[contributionsOffset + sortedIndices[sortedIndicesOffset + index]];`
      })
    );
    nodes.push(
      createMapGraphSegmentSumNode<Parameters>(graph, {
        id: `${id}-reduce-${reduction.name}`,
        operation,
        segmentCount,
        input: sortedContributions,
        segmentOffsets,
        output: reduction.output
      })
    );
  }
  return nodes;
}

/**
 * Sums each segment of `input` with one 256-thread workgroup: a strided per-thread partial sum
 * followed by a fixed binary tree. The order depends only on the segment bounds, so the result is
 * bitwise reproducible. Segments are workgroups of a bounded 3D dispatch, so the segment count is
 * not limited by `maxComputeWorkgroupsPerDimension`.
 *
 * @internal
 */
export function createMapGraphSegmentSumNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    /** Operation name reported in workload estimates. */
    operation: string;
    segmentCount: number;
    input: GraphDataView<'float32'>;
    segmentOffsets: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'segment-sum',
    bindings: [
      {name: 'segmentInput', view: props.input, type: 'f32', access: 'read'},
      {name: 'segmentOffsets', view: props.segmentOffsets, type: 'u32', access: 'read'},
      {name: 'segmentOutput', view: props.output, type: 'f32', access: 'read_write'}
    ],
    workgroupSize: SEGMENT_WORKGROUP_SIZE,
    invocationCount: props.segmentCount * SEGMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const INPUT_COUNT: u32 = ${props.input.length}u;
var<workgroup> partialSums: array<f32, ${SEGMENT_WORKGROUP_SIZE}>;`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let segment = index / ${SEGMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var partialSum = 0.0;
  var begin = 0u;
  var end = 0u;
  if (isInRange) {
    begin = segmentOffsets[segmentOffsetsOffset + segment];
    end = min(segmentOffsets[segmentOffsetsOffset + segment + 1u], INPUT_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += ${SEGMENT_WORKGROUP_SIZE}u) {
      partialSum += segmentInput[segmentInputOffset + row];
    }
  }
  partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = ${SEGMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    segmentOutput[segmentOutputOffset + segment] = partialSums[0];
  }`
  });
}
