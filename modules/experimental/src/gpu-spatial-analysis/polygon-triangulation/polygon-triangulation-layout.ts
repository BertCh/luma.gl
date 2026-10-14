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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';

/** GPU layout metadata sharing the existing offset bindings. @internal */
export type PolygonTriangulationLayout = {
  /** Original polygon offsets, triangle prefixes, live vertex counts, then live ring counts. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Original ring offsets followed by one non-degenerate flag per ring. */
  ringOffsets: GraphDataView<'uint32'>;
};

/**
 * Classifies rings and scans polygon triangle capacities entirely on the GPU. Original topology
 * prefixes remain at the beginning of each binding; appended metadata keeps Earcut and monotone
 * kernels within eight storage bindings. Non-degenerate layouts retain their original slices.
 * @internal
 */
export function getPolygonTriangulationLayoutNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringOffsets: GraphDataView<'uint32'>;
    triangleOffsets: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; layout: PolygonTriangulationLayout} {
  const {id, positions, polygonOffsets, ringOffsets, triangleOffsets} = props;
  const polygonCount = polygonOffsets.length - 1;
  const ringCount = ringOffsets.length - 1;
  const operation = 'GPUPolygonTriangulation';
  const layout: PolygonTriangulationLayout = {
    polygonOffsets: createTransientView(
      graph,
      `${id}-polygon-layout`,
      'uint32',
      4 * polygonCount + 2
    ),
    ringOffsets: createTransientView(graph, `${id}-ring-layout`, 'uint32', 2 * ringCount + 1)
  };
  const counts = createTransientView(
    graph,
    `${id}-triangle-capacities`,
    'uint32',
    polygonCount + 1
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-classify-rings`,
      operation,
      variant: 'classify-rings',
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
        {name: 'ringLayout', view: layout.ringOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: ringCount + 1,
      declarations: `${EXACT_ORIENTATION_WGSL}
fn readVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}`,
      body: `let first = ringOffsets[ringOffsetsOffset + index];
  ringLayout[ringLayoutOffset + index] = first;
  if (index == ${ringCount}u) { return; }
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], ${positions.length}u);
  var nonDegenerate = 0u;
  if (end >= first + 3u) {
    let origin = readVertex(first);
    for (var vertex = first + 1u; vertex + 1u < end; vertex++) {
      // Match study-window preparation: uncertified (2) is not exact zero.
      if (orientSign(origin, readVertex(vertex), readVertex(vertex + 1u)) != 0) {
        nonDegenerate = 1u;
        break;
      }
    }
  }
  ringLayout[ringLayoutOffset + ${ringCount + 1}u + index] = nonDegenerate;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-polygon-capacities`,
      operation,
      variant: 'polygon-capacities',
      bindings: [
        {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
        {name: 'ringLayout', view: layout.ringOffsets, type: 'u32', access: 'read'},
        {name: 'polygonLayout', view: layout.polygonOffsets, type: 'u32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: polygonCount + 1,
      body: `let first = polygonOffsets[polygonOffsetsOffset + index];
  polygonLayout[polygonLayoutOffset + index] = first;
  counts[countsOffset + index] = 0u;
  if (index == ${polygonCount}u) { return; }
  let end = polygonOffsets[polygonOffsetsOffset + index + 1u];
  var vertices = 0u;
  var rings = 0u;
  if (first < end && ringLayout[ringLayoutOffset + ${ringCount + 1}u + first] != 0u) {
    for (var ring = first; ring < end; ring++) {
      if (ringLayout[ringLayoutOffset + ${ringCount + 1}u + ring] == 0u) { continue; }
      vertices += ringLayout[ringLayoutOffset + ring + 1u] - ringLayout[ringLayoutOffset + ring];
      rings += 1u;
    }
  }
  if (rings > 0u) { counts[countsOffset + index] = vertices + 2u * rings - 4u; }
  polygonLayout[polygonLayoutOffset + ${2 * (polygonCount + 1)}u + index] = vertices;
  polygonLayout[polygonLayoutOffset + ${3 * polygonCount + 2}u + index] = rings;`
    }),
    ...new GPUScan({
      id: `${id}-triangle-offsets`,
      input: counts,
      output: triangleOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-publish-layout`,
      operation,
      variant: 'publish-layout',
      bindings: [
        {name: 'triangleOffsets', view: triangleOffsets, type: 'u32', access: 'read'},
        {name: 'polygonLayout', view: layout.polygonOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: polygonCount + 1,
      body: `polygonLayout[polygonLayoutOffset + ${polygonCount + 1}u + index] = triangleOffsets[triangleOffsetsOffset + index];`
    })
  ];
  return {nodes, layout};
}
