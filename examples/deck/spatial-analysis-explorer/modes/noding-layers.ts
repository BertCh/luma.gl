// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Display passes for the Noding mode: expand the contributor's piece vertex runs into drawable
 * segment rows, and derive per-row and per-node values, all on the GPU. They read contributor
 * outputs directly and add nothing to the noding result.
 */

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from './mode-kernels';

/** Sentinel in `rowPieces` and `nodeDegrees` for rows that are not drawn. */
export const NODING_HIDDEN = 0xffffffff;

/** Inputs of {@link addNodingDisplayPasses}. */
export type NodingDisplayProps = {
  /** Piece offsets (`pieceCapacity + 1`). */
  pieceOffsets: GraphDataView<'uint32'>;
  /** Piece vertices (`vertexCapacity` rows). */
  piecePositions: GraphDataView<'float32x2'>;
  /** One-row piece count. */
  pieceCount: GraphDataView<'uint32'>;
  /** Edge start and end nodes per piece. */
  fromNodes: GraphDataView<'uint32'>;
  toNodes: GraphDataView<'uint32'>;
  /** CSR row offsets (`nodeCapacity + 1`). */
  csrOffsets: GraphDataView<'uint32'>;
  /** One-row node count. */
  nodeCount: GraphDataView<'uint32'>;
  /** Per-node travel cost from the reachability source, `+Infinity` when unreached. */
  costs: GraphDataView<'float32'>;
  /** Output: one `float32x4` row `x0, y0, x1, y1` per vertex slot, viewed as two `float32x2` rows. */
  segmentRows: GraphDataView<'float32x2'>;
  /** Hashed piece index per row, or {@link NODING_HIDDEN}. */
  rowPieces: GraphDataView<'uint32'>;
  /** Travel cost per row (the cheaper end of its piece), or -1 for hidden rows. */
  rowCosts: GraphDataView<'float32'>;
  /** Degree per node slot, or {@link NODING_HIDDEN} beyond the node count. */
  nodeDegrees: GraphDataView<'uint32'>;
};

/**
 * Adds the display passes to `graph`. Row `v` is the segment from vertex `v` to `v + 1` of the
 * flat piece vertex array; the last vertex of each piece, and slots beyond the written vertices,
 * are hidden.
 */
export function addNodingDisplayPasses<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NodingDisplayProps
): void {
  const vertexCapacity = props.piecePositions.length;
  const nodeCapacity = props.nodeDegrees.length;
  const pieceLookup = `
// Largest piece whose first vertex is at most vertex, among the written pieces.
fn pieceOfVertex(vertex: u32, count: u32) -> u32 {
  var low = 0u;
  var high = count - 1u;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (pieceOffsets[pieceOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle - 1u; }
  }
  return low;
}`;
  addKernelPass(graph, {
    id: 'noding-segment-rows',
    bindings: [
      {name: 'pieceOffsets', view: props.pieceOffsets, type: 'u32', access: 'read'},
      {name: 'pieceCount', view: props.pieceCount, type: 'u32', access: 'read'},
      {name: 'positions', view: props.piecePositions, type: 'f32', access: 'read'},
      {name: 'segmentRows', view: props.segmentRows, type: 'f32', access: 'read_write'},
      {name: 'rowPieces', view: props.rowPieces, type: 'u32', access: 'read_write'}
    ],
    invocationCount: vertexCapacity,
    declarations: pieceLookup,
    body: `let count = pieceCount[pieceCountOffset];
  var piece = 0u;
  var visible = false;
  if (count > 0u && index + 1u < pieceOffsets[pieceOffsetsOffset + count]) {
    piece = pieceOfVertex(index, count);
    visible = index + 1u < pieceOffsets[pieceOffsetsOffset + piece + 1u];
  }
  var segment = vec4f(0.0);
  if (visible) {
    segment = vec4f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u],
      positions[positionsOffset + index * 2u + 2u], positions[positionsOffset + index * 2u + 3u]);
  }
  // Rows are float32x4 (two vertices); vertex slots are float32x2, so a row spans slot 2*index.
  segmentRows[segmentRowsOffset + index * 4u] = segment.x;
  segmentRows[segmentRowsOffset + index * 4u + 1u] = segment.y;
  segmentRows[segmentRowsOffset + index * 4u + 2u] = segment.z;
  segmentRows[segmentRowsOffset + index * 4u + 3u] = segment.w;
  // Multiplicative hash so neighbouring pieces get unrelated palette entries.
  rowPieces[rowPiecesOffset + index] = select(0xffffffffu, (piece * 2654435761u) >> 29u, visible);`
  });
  addKernelPass(graph, {
    id: 'noding-row-costs',
    bindings: [
      {name: 'pieceOffsets', view: props.pieceOffsets, type: 'u32', access: 'read'},
      {name: 'pieceCount', view: props.pieceCount, type: 'u32', access: 'read'},
      {name: 'fromNodes', view: props.fromNodes, type: 'u32', access: 'read'},
      {name: 'toNodes', view: props.toNodes, type: 'u32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'rowCosts', view: props.rowCosts, type: 'f32', access: 'read_write'}
    ],
    invocationCount: vertexCapacity,
    declarations: `const NODE_CAPACITY: u32 = ${nodeCapacity}u;${pieceLookup}`,
    body: `let count = pieceCount[pieceCountOffset];
  var cost = -1.0;
  if (count > 0u && index + 1u < pieceOffsets[pieceOffsetsOffset + count]) {
    let piece = pieceOfVertex(index, count);
    if (index + 1u < pieceOffsets[pieceOffsetsOffset + piece + 1u]) {
      let fromNode = fromNodes[fromNodesOffset + piece];
      let toNode = toNodes[toNodesOffset + piece];
      cost = 3.4e38;
      if (fromNode < NODE_CAPACITY) { cost = min(cost, costs[costsOffset + fromNode]); }
      if (toNode < NODE_CAPACITY) { cost = min(cost, costs[costsOffset + toNode]); }
    }
  }
  rowCosts[rowCostsOffset + index] = cost;`
  });
  addKernelPass(graph, {
    id: 'noding-node-degrees',
    bindings: [
      {name: 'csrOffsets', view: props.csrOffsets, type: 'u32', access: 'read'},
      {name: 'nodeCount', view: props.nodeCount, type: 'u32', access: 'read'},
      {name: 'nodeDegrees', view: props.nodeDegrees, type: 'u32', access: 'read_write'}
    ],
    invocationCount: nodeCapacity,
    body: `var degree = 0xffffffffu;
  if (index < nodeCount[nodeCountOffset]) {
    degree = min(csrOffsets[csrOffsetsOffset + index + 1u] - csrOffsets[csrOffsetsOffset + index], 7u);
  }
  nodeDegrees[nodeDegreesOffset + index] = degree;`
  });
}
