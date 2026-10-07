// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/**
 * Builds the node that writes `ringFlip`: one `uint32` per ring, `1` when the ring has its polygon
 * interior on the right of the vertex order (a counter-clockwise hole or a clockwise shell) and
 * `0` otherwise. The first ring of a polygon is its shell, the others are holes. Rings with fewer
 * than three vertices, rings outside every polygon and zero-area rings get `0`.
 *
 * The signed area is a shoelace sum relative to the ring's first vertex, walked by one thread per
 * ring, so a single very long ring costs a serial walk of its length. Only the sign is used.
 *
 * @internal
 */
export function createRingOrientationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringFlip: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {positions, ringOffsets, polygonOffsets, ringFlip} = props;
  const ringCount = ringOffsets.length - 1;
  const polygonCount = polygonOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'ring-orientation',
    bindings: [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringFlip', view: ringFlip, type: 'u32', access: 'read_write'}
    ],
    invocationCount: ringCount,
    declarations: `const VERTEX_COUNT: u32 = ${positions.length}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;`,
    body: `let begin = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  var flip = 0u;
  if (end > begin + 2u && index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    var polygon = 0u;
    var high = POLYGON_COUNT;
    while (polygon + 1u < high) {
      let middle = (polygon + high) / 2u;
      if (polygonOffsets[polygonOffsetsOffset + middle] <= index) {
        polygon = middle;
      } else {
        high = middle;
      }
    }
    let isShell = polygonOffsets[polygonOffsetsOffset + polygon] == index;
    let origin = vec2f(positions[positionsOffset + begin * 2u], positions[positionsOffset + begin * 2u + 1u]);
    var previous = vec2f(0.0);
    var twiceArea = 0.0;
    for (var vertex = begin + 1u; vertex < end; vertex++) {
      let current = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]) - origin;
      twiceArea += previous.x * current.y - previous.y * current.x;
      previous = current;
    }
    flip = select(select(0u, 1u, twiceArea > 0.0), select(0u, 1u, twiceArea < 0.0), isShell);
  }
  ringFlip[ringFlipOffset + index] = flip;`
  });
}
