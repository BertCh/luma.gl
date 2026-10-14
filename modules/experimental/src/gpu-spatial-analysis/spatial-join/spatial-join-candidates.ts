// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {createScanCarryNode} from '../../utils/scan-total';
import {SPATIAL_JOIN_WGSL_HELPERS} from './spatial-join-passes';
import {NODE_OVERLAPS_WGSL} from '../../utils/wgsl-common';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';

const OPERATION = 'GPUSpatialJoinCandidates';

/** BVH facts the candidate probe needs. @internal */
export type SpatialJoinCandidateTree = {
  /** Node minima of the right-hand BVH. */
  nodeMinima: GraphDataView<'float32x2'>;
  /** Node maxima of the right-hand BVH. */
  nodeMaxima: GraphDataView<'float32x2'>;
  /** Leaf slot to right feature row. */
  leafIds: GraphDataView<'uint32'>;
  /** Number of internal nodes. */
  internalNodeCount: number;
};

/** Formats a float as a WGSL literal. @internal */
export function formatSpatialJoinFloat(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/**
 * Returns the shared bounding-box candidate stage: a counted and scanned BVH probe from every left
 * box (expanded by `margin`), writing `(left, right)` rows ordered by left row and then right row.
 *
 * `state[0]` receives the unclamped candidate total. Slots past the total keep whatever the caller
 * cleared them to; slots past `candidateCapacity` are dropped. Nodes: count, scan, total, write.
 *
 * @internal
 */
export function getSpatialJoinCandidateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    leftCount: number;
    rightCount: number;
    leftMinima: GraphDataView<'float32x2'>;
    leftMaxima: GraphDataView<'float32x2'>;
    tree: SpatialJoinCandidateTree;
    candidateCapacity: number;
    /**
     * Box expansion in coordinate units: a compile-time number, or a one-row float32 view read at
     * every encoding (a negative, NaN or infinite value counts as 0).
     */
    margin: number | GraphDataView<'float32'>;
    /**
     * When set, boxes hold longitude/latitude degrees and `margin` is a great-circle distance in
     * meters on a sphere of this radius. Each left box is then expanded by the angular radius in
     * latitude and by the widest longitude reach of that radius at the box's far latitude (the
     * whole longitude range when the expansion reaches a pole), and a box that crosses the
     * antimeridian also probes its copy shifted by 360 degrees. The expansion is conservative by
     * 0.1% plus 1e-5 degrees.
     */
    sphereRadius?: number;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, leftCount, rightCount, candidateCapacity, tree, state, candidatePairs} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const leftCounts = createTransientView(graph, `${id}-left-counts`, 'uint32', leftCount);
  const leftOffsets = createTransientView(graph, `${id}-left-offsets`, 'uint32', leftCount);
  const marginView = typeof props.margin === 'number' ? undefined : props.margin;
  const {sphereRadius} = props;
  const probeBindings = (extra: WGSLKernelBinding[]): WGSLKernelBinding[] => [
    {name: 'leftMinima', view: props.leftMinima, type: 'f32', access: 'read'},
    {name: 'leftMaxima', view: props.leftMaxima, type: 'f32', access: 'read'},
    {name: 'nodeMinima', view: tree.nodeMinima, type: 'f32', access: 'read'},
    {name: 'nodeMaxima', view: tree.nodeMaxima, type: 'f32', access: 'read'},
    {name: 'leafIds', view: tree.leafIds, type: 'u32', access: 'read'},
    ...(marginView
      ? [{name: 'marginRow', view: marginView, type: 'f32', access: 'read'} as const]
      : []),
    ...extra
  ];
  const marginWGSL = marginView
    ? `fn getMargin() -> f32 {
  let margin = marginRow[marginRowOffset];
  return select(0.0, margin, margin >= 0.0 && margin <= 3.4028234e38);
}`
    : `fn getMargin() -> f32 { return ${formatSpatialJoinFloat(Math.fround(props.margin as number))}; }`;
  const probeDeclarations = `${SPATIAL_JOIN_WGSL_HELPERS}
const INTERNAL_NODE_COUNT: u32 = ${tree.internalNodeCount}u;
const RIGHT_COUNT: u32 = ${rightCount}u;
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
${marginWGSL}
${NODE_OVERLAPS_WGSL}
${
  sphereRadius === undefined
    ? ''
    : `${GEODESIC_WGSL}
const SPHERE_RADIUS: f32 = ${formatSpatialJoinFloat(sphereRadius)};
const WRAP_SHIFT: vec2f = vec2f(360.0, 0.0);`
}`;
  const expansion =
    sphereRadius === undefined
      ? `let queryMinimum = boxMinimum - vec2f(margin);
    let queryMaximum = boxMaximum + vec2f(margin);`
      : `var queryMinimum = boxMinimum;
    var queryMaximum = boxMaximum;
    var wrapDown = false;
    var wrapUp = false;
    if (margin > 0.0) {
      let angle = margin / SPHERE_RADIUS * 1.001;
      let deltaLatitude = angle * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
      queryMinimum.y = boxMinimum.y - deltaLatitude;
      queryMaximum.y = boxMaximum.y + deltaLatitude;
      let farLatitude = max(abs(queryMinimum.y), abs(queryMaximum.y));
      var ratio = 2.0;
      if (farLatitude < 90.0) { ratio = sin(0.5 * angle) / geodesicCosLatitude(farLatitude); }
      if (ratio < 1.0) {
        let deltaLongitude = 2.0 * asin(ratio) * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
        queryMinimum.x = boxMinimum.x - deltaLongitude;
        queryMaximum.x = boxMaximum.x + deltaLongitude;
        wrapDown = queryMaximum.x > 180.0;
        wrapUp = queryMinimum.x < -180.0;
      } else {
        // The cap reaches a pole (or is wider than a hemisphere in longitude): every longitude.
        queryMinimum.x = -FLOAT32_MAXIMUM;
        queryMaximum.x = FLOAT32_MAXIMUM;
      }
    }`;
  const overlaps =
    sphereRadius === undefined
      ? 'nodeOverlaps(node, queryMinimum, queryMaximum)'
      : `(nodeOverlaps(node, queryMinimum, queryMaximum) ||
        (wrapDown && nodeOverlaps(node, queryMinimum - WRAP_SHIFT, queryMaximum - WRAP_SHIFT)) ||
        (wrapUp && nodeOverlaps(node, queryMinimum + WRAP_SHIFT, queryMaximum + WRAP_SHIFT)))`;
  const probeBody = (
    leaf: string,
    prologue: string,
    epilogue: string
  ) => `let boxMinimum = vec2f(leftMinima[leftMinimaOffset + index * 2u], leftMinima[leftMinimaOffset + index * 2u + 1u]);
  let boxMaximum = vec2f(leftMaxima[leftMaximaOffset + index * 2u], leftMaxima[leftMaximaOffset + index * 2u + 1u]);
  ${prologue}
  if (boxMinimum.x <= boxMaximum.x && boxMinimum.y <= boxMaximum.y) {
    let margin = getMargin();
    ${expansion}
    var node = 0u;
    loop {
      if (${overlaps}) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          ${leaf}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${epilogue}`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-count-candidates`,
      operation: OPERATION,
      variant: 'count-candidates',
      bindings: probeBindings([
        {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read_write'}
      ]),
      invocationCount: leftCount,
      declarations: probeDeclarations,
      body: probeBody(
        'found = found + 1u;',
        'var found = 0u;',
        'leftCounts[leftCountsOffset + index] = found;'
      )
    })
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan-candidates`,
      input: leftCounts,
      output: leftOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-total-candidates`,
      operation: OPERATION,
      variant: 'total-candidates',
      bindings: [
        {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read'},
        {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `state[stateOffset] = leftOffsets[leftOffsetsOffset + ${leftCount - 1}u] + leftCounts[leftCountsOffset + ${leftCount - 1}u];`
    })
  );
  // A candidate total of 2^32 or more wraps in the uint32 scan; saturate it (S-8).
  nodes.push(
    createScanCarryNode<Parameters>(graph, {
      id: `${id}-candidates`,
      operation: OPERATION,
      counts: leftCounts,
      offsets: leftOffsets,
      rowCount: leftCount,
      total: state
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-write-candidates`,
      operation: OPERATION,
      variant: 'write-candidates',
      bindings: probeBindings([
        {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
        {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
      ]),
      invocationCount: leftCount,
      declarations: probeDeclarations,
      body: probeBody(
        `let slot = slotStart + found;
          candidatePairs[candidatePairsOffset + slot * 2u] = index;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = rightRow;
          found = found + 1u;
          if (slot + 1u >= slotEnd) { break; }`,
        // Left features with no candidates, or whose slots all fall past the capacity, skip the
        // second traversal; the rest stop at their last counted candidate. The last row's end comes
        // from the capacity, which keeps this pass within 8 storage bindings.
        `var found = 0u;
  let slotStart = leftOffsets[leftOffsetsOffset + index];
  var slotEnd = CANDIDATE_CAPACITY;
  if (index + 1u < ${leftCount}u) {
    slotEnd = min(leftOffsets[leftOffsetsOffset + index + 1u], CANDIDATE_CAPACITY);
  }
  if (slotStart >= slotEnd) { return; }`,
        ''
      )
    })
  );
  return nodes;
}
