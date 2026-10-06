// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUBVH,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/**
 * Builds a `GPUBVH` over segment bounds, optionally after a Morton (Z-order) sort of the segments.
 *
 * With `spatialSort`, segments are reordered along a 16-bit-per-axis Morton curve of their bound
 * centers (skipped segments last) before the build, and `leafIds` map back to segment rows. This
 * keeps leaves that are close in space close in the tree; without it a source order that jumps
 * around the plane makes upper nodes cover everything and traversal visits most of the tree.
 *
 * A private copy of the technique used by the spatial join contributors, kept here so this
 * contributor does not depend on their internals. Nothing is read back.
 *
 * @internal
 */
export function getSegmentBVHNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    minima: GraphDataView<'float32x2'>;
    maxima: GraphDataView<'float32x2'>;
    leafCapacity: number;
    spatialSort: boolean;
  }
): {bvh: GPUBVH; nodes: readonly GPUCommandNode<Parameters>[]} {
  const {id, operation, minima, maxima, leafCapacity, spatialSort} = props;
  const segmentCount = minima.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const nodeCount = 2 * leafCapacity - 1;
  let bvhMinima = minima;
  let bvhMaxima = maxima;
  let sourceIds: GraphDataView<'uint32'> | undefined;

  if (spatialSort && segmentCount >= 2) {
    const sceneBounds = createTransientView(graph, `${id}-sort-scene-bounds`, 'float32', 4);
    const keys = createTransientView(graph, `${id}-sort-keys`, 'uint32', segmentCount);
    const rows = createTransientView(graph, `${id}-sort-rows`, 'uint32', segmentCount);
    const sortedKeys = createTransientView(graph, `${id}-sort-sorted-keys`, 'uint32', segmentCount);
    const sortedRows = createTransientView(graph, `${id}-sort-sorted-rows`, 'uint32', segmentCount);
    bvhMinima = createTransientView(graph, `${id}-sort-minima`, 'float32x2', segmentCount);
    bvhMaxima = createTransientView(graph, `${id}-sort-maxima`, 'float32x2', segmentCount);
    sourceIds = sortedRows;
    const boundsBindings: WGSLKernelBinding[] = [
      {name: 'segmentMinima', view: minima, type: 'f32', access: 'read'},
      {name: 'segmentMaxima', view: maxima, type: 'f32', access: 'read'}
    ];
    const declarations = `const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const SEGMENT_COUNT: u32 = ${segmentCount}u;
fn readCenter(row: u32) -> vec3f {
  // (center.x, center.y, valid). Skipped segments have inverted bounds.
  let minimum = vec2f(segmentMinima[segmentMinimaOffset + row * 2u], segmentMinima[segmentMinimaOffset + row * 2u + 1u]);
  let maximum = vec2f(segmentMaxima[segmentMaximaOffset + row * 2u], segmentMaxima[segmentMaximaOffset + row * 2u + 1u]);
  let valid = minimum.x <= maximum.x && minimum.y <= maximum.y;
  let center = minimum * 0.5 + maximum * 0.5;
  return vec3f(center, select(0.0, 1.0, valid));
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-scene-bounds`,
        operation,
        variant: 'sort-scene-bounds',
        bindings: [
          ...boundsBindings,
          {name: 'sceneBounds', view: sceneBounds, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 256,
        guardIndex: false,
        declarations: `${declarations}
var<workgroup> sharedMinima: array<vec2f, 256>;
var<workgroup> sharedMaxima: array<vec2f, 256>;`,
        body: `// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var row = localInvocationIndex; row < SEGMENT_COUNT; row += 256u) {
    let center = readCenter(row);
    if (center.z > 0.5) {
      localMinimum = min(localMinimum, center.xy);
      localMaximum = max(localMaximum, center.xy);
    }
  }
  sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    sceneBounds[sceneBoundsOffset] = select(0.0, minimum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 1u] = select(0.0, minimum.y, hasValid);
    sceneBounds[sceneBoundsOffset + 2u] = select(0.0, maximum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 3u] = select(0.0, maximum.y, hasValid);
  }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-keys`,
        operation,
        variant: 'sort-keys',
        bindings: [
          ...boundsBindings,
          {name: 'sceneBounds', view: sceneBounds, type: 'f32', access: 'read'},
          {name: 'sortKeys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'sortRows', view: rows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: segmentCount,
        declarations: `${declarations}
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || extent > FLOAT32_MAXIMUM) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(normalized * 65535.0 + 0.5), 65535u);
}`,
        body: `sortRows[sortRowsOffset + index] = index;
  let center = readCenter(index);
  var key = 0xffffffffu;
  if (center.z > 0.5) {
    let quantizedX = quantizeAxis(center.x, sceneBounds[sceneBoundsOffset], sceneBounds[sceneBoundsOffset + 2u]);
    let quantizedY = quantizeAxis(center.y, sceneBounds[sceneBoundsOffset + 1u], sceneBounds[sceneBoundsOffset + 3u]);
    key = spreadBits(quantizedX) | (spreadBits(quantizedY) << 1u);
  }
  sortKeys[sortKeysOffset + index] = key;`
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-order`,
        keys,
        values: rows,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-gather`,
        operation,
        variant: 'sort-gather',
        bindings: [
          ...boundsBindings,
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'sortedMinima', view: bvhMinima, type: 'f32', access: 'read_write'},
          {name: 'sortedMaxima', view: bvhMaxima, type: 'f32', access: 'read_write'}
        ],
        invocationCount: segmentCount,
        declarations: `const SEGMENT_COUNT: u32 = ${segmentCount}u;`,
        body: `let row = min(sortedRows[sortedRowsOffset + index], SEGMENT_COUNT - 1u);
  sortedMinima[sortedMinimaOffset + index * 2u] = segmentMinima[segmentMinimaOffset + row * 2u];
  sortedMinima[sortedMinimaOffset + index * 2u + 1u] = segmentMinima[segmentMinimaOffset + row * 2u + 1u];
  sortedMaxima[sortedMaximaOffset + index * 2u] = segmentMaxima[segmentMaximaOffset + row * 2u];
  sortedMaxima[sortedMaximaOffset + index * 2u + 1u] = segmentMaxima[segmentMaximaOffset + row * 2u + 1u];`
      })
    );
  }

  const bvh = new GPUBVH({
    id: `${id}-bvh`,
    minima: bvhMinima,
    maxima: bvhMaxima,
    sourceIds,
    leafCapacity,
    nodeMinima: createTransientView(graph, `${id}-bvh-node-minima`, 'float32x2', nodeCount),
    nodeMaxima: createTransientView(graph, `${id}-bvh-node-maxima`, 'float32x2', nodeCount),
    nodeChildren: createTransientView(graph, `${id}-bvh-node-children`, 'uint32x2', nodeCount),
    leafIds: createTransientView(graph, `${id}-bvh-leaf-ids`, 'uint32', leafCapacity),
    count: createTransientView(graph, `${id}-bvh-count`, 'uint32', 1),
    overflow: createTransientView(graph, `${id}-bvh-overflow`, 'uint32', 1)
  });
  nodes.push(...bvh.getCommandNodes(graph));
  return {bvh, nodes};
}
