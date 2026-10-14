// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GPUGroupConvexHull} from '../group-geometry/gpu-group-convex-hull';
import {
  getPositionDeclarations,
  getRingBindings,
  getRingDeclarations,
  type FeatureRingInputs
} from './shape-descriptors-kernels';

/** `GPUGroupConvexHull` supports fewer than `2^24` rows and groups. */
const HULL_LIMIT = 2 ** 24 - 1;

/** Whether the sort-based hull can serve this input size. @internal */
export function canUseMonotoneChainConvexity(rowCount: number, featureCount: number): boolean {
  return rowCount < HULL_LIMIT && featureCount < HULL_LIMIT;
}

/**
 * Per-feature convexity through `GPUGroupConvexHull`: every vertex of a feature is labeled with the
 * feature index, hulls come from the sort-based monotone chain (`O(n log n)` work, parallel
 * prefilter, exact predicates), and a small kernel integrates each hull. This replaces the
 * `O(vertices * hull vertices)` per-feature gift wrapping when features are large or skewed.
 *
 * @internal
 */
export function getMonotoneChainConvexityNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FeatureRingInputs & {
    areas: GraphDataView<'float32'>;
    convexities: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, featureCount} = props;
  const rowCount = props.positions.length;
  const labels = createTransientView(graph, `${id}-labels`, 'uint32', rowCount);
  const hullIndices = createTransientView(graph, `${id}-hull-indices`, 'uint32', rowCount);
  const hullOffsets = createTransientView(graph, `${id}-hull-offsets`, 'uint32', featureCount + 1);
  const hullCounts = createTransientView(graph, `${id}-hull-counts`, 'uint32', featureCount);
  const hullOverflow = createTransientView(graph, `${id}-hull-overflow`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    // Label each vertex row with the feature that owns it; rows outside every feature are excluded
    // by the label `featureCount`.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-labels`,
      operation,
      variant: 'convexity-labels',
      bindings: [
        ...getRingBindings(props),
        {name: 'labels', view: labels, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rowCount,
      declarations: `${getRingDeclarations(props)}
const FEATURE_COUNT: u32 = ${featureCount}u;
fn getFeatureRowStart(feature: u32) -> u32 { return getRingRowStart(getRingStart(feature)); }
fn getFeatureRowEnd(feature: u32) -> u32 { return max(getRingRowStart(getRingEnd(feature)), getFeatureRowStart(feature)); }`,
      body: `// Last feature whose first row is at or before this row (skips empty features).
  var low = 0u;
  var high = FEATURE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (getFeatureRowStart(middle) <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let owned = getFeatureRowStart(low) <= index && index < getFeatureRowEnd(low);
  labels[labelsOffset + index] = select(FEATURE_COUNT, low, owned);`
    }),
    ...new GPUGroupConvexHull({
      id: `${id}-hull`,
      positions: props.positions,
      labels,
      groupCount: featureCount,
      // A hull never has more vertices than the rows it was built from, so neither cap can trip.
      maximumVerticesPerGroup: Math.max(3, rowCount),
      vertexCapacity: rowCount,
      output: {
        vertexIndices: hullIndices,
        offsets: hullOffsets,
        counts: hullCounts,
        status: {overflow: hullOverflow}
      }
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-hull-area`,
      operation,
      variant: 'convexity-area',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'areas', view: props.areas, type: 'f32', access: 'read'},
        {name: 'hullIndices', view: hullIndices, type: 'u32', access: 'read'},
        {name: 'hullOffsets', view: hullOffsets, type: 'u32', access: 'read'},
        {name: 'hullCounts', view: hullCounts, type: 'u32', access: 'read'},
        {name: 'convexities', view: props.convexities, type: 'f32', access: 'read_write'}
      ],
      invocationCount: featureCount,
      declarations: getPositionDeclarations(rowCount),
      // A hull of three or more vertices implies three or more rows, so the feature's row range is
      // not needed (the kernel is at the 8-binding limit with the offset views).
      body: `let area = areas[areasOffset + index];
  let hullStart = hullOffsets[hullOffsetsOffset + index];
  let hullCount = hullCounts[hullCountsOffset + index];
  var hullArea = 0.0;
  if (area > 0.0 && hullCount >= 3u) {
    // Shoelace over the counter-clockwise hull, relative to its first vertex for f32 accuracy.
    let origin = getPosition(hullIndices[hullIndicesOffset + hullStart]);
    var previous = getPosition(hullIndices[hullIndicesOffset + hullStart + hullCount - 1u]) - origin;
    var twiceArea = 0.0;
    for (var vertex = 0u; vertex < hullCount; vertex++) {
      let current = getPosition(hullIndices[hullIndicesOffset + hullStart + vertex]) - origin;
      twiceArea += previous.x * current.y - current.x * previous.y;
      previous = current;
    }
    hullArea = 0.5 * twiceArea;
  }
  convexities[convexitiesOffset + index] = select(getNan(index), min(area / hullArea, 1.0), hullArea > 0.0);`
    })
  ];
  return nodes;
}
