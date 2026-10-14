// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getParallelMergeTreeNodes} from '../../utils/parallel-merge-tree';
import {getSpillCommonWGSL} from '../hydrology/terrain-flow-spill-utils';

/** Parallel spill-graph join tree and elder reduction. @internal */
export function getPeakProminenceMergeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    values: GraphDataView;
    validity: GraphDataView<'uint32'>;
    sortedKeys: GraphDataView<'uint32'>;
    sortedEdges: GraphDataView<'uint32'>;
    result: GraphDataView<'uint32'>;
  }
) {
  const count = props.width * props.height;
  const nodeCount = count * 5;
  const common = `${getSpillCommonWGSL(props.width, props.height)}
const CELL_COUNT: u32 = ${count}u;`;
  const {nodes, parents, children, createWordView, appendKernel, appendJumpNodes} =
    getParallelMergeTreeNodes(graph, {
      id: `${props.id}-tree`,
      operation: 'GPUTerrainPeakProminence',
      nodeCount,
      edgeCount: count * 4,
      edgeBindings: [
        {name: 'keys', view: props.sortedKeys, type: 'u32', access: 'read'},
        {name: 'edges', view: props.sortedEdges, type: 'u32', access: 'read'}
      ],
      declarations: common,
      edgeBody: `if (keys[keysOffset + index] == 0u) { return; }
  let edge = edges[edgesOffset + index];
  let firstEndpoint = edge / 4u;
  let secondEndpoint = getSpillNeighbor(firstEndpoint, edge % 4u);
  let node = CELL_COUNT + index;`
    });
  const elders = createWordView('elders', nodeCount);
  appendKernel(
    'elders-init',
    nodeCount,
    [
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
      {name: 'elders', view: elders, type: 'u32', access: 'read_write'}
    ],
    `var elder = NONE;
  if (index < CELL_COUNT) {
    if (validity[validityOffset + index] != 0u && isSpillFinite(values[valuesOffset + index])) {
      elder = index;
    }
  }
  elders[eldersOffset + index] = elder;`,
    common
  );
  // After round k, a node has elders from every descendant at distance < 2^(k+1).
  // Each lane scatters to its 2^k ancestor and doubles that pointer. Atomic updates may
  // arrive early within a dispatch, which is safe because max is associative/idempotent.
  // The canonical elder table is never cleared between rounds. Roots retain all updates.
  appendJumpNodes(
    'elders',
    nodeCount,
    1,
    Math.ceil(Math.log2(nodeCount)) + 1,
    parents,
    `
  var ancestor = table[tableOffset + index];
  let candidate = atomicLoad(&elders[eldersOffset + index]);
  if (ancestor != NONE) {
    if (candidate != NONE) {
      var previous = atomicLoad(&elders[eldersOffset + ancestor]);
      loop {
        if (previous != NONE) {
          let previousHeight = values[valuesOffset + previous];
          let candidateHeight = values[valuesOffset + candidate];
          if (previousHeight > candidateHeight || (previousHeight == candidateHeight && previous <= candidate)) { break; }
        }
        let exchange = atomicCompareExchangeWeak(&elders[eldersOffset + ancestor], previous, candidate);
        if (exchange.exchanged) { break; }
        previous = exchange.old_value;
      }
    }
    ancestor = table[tableOffset + ancestor];
  }
  scratch[scratchOffset + index] = ancestor;
  if (ancestor != NONE) { atomicOr(&pending[pendingOffset], 1u); }
  `,
    '',
    [
      {name: 'elders', view: elders, type: 'atomic<u32>', access: 'read_write'},
      {name: 'values', view: props.values, type: 'f32', access: 'read'}
    ]
  );
  appendKernel(
    'record',
    count * 4,
    [
      {name: 'children', view: children, type: 'u32', access: 'read'},
      {name: 'elders', view: elders, type: 'u32', access: 'read'},
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'keys', view: props.sortedKeys, type: 'u32', access: 'read'},
      {name: 'result', view: props.result, type: 'u32', access: 'read_write'}
    ],
    `let node = CELL_COUNT + index;
  let first = children[childrenOffset + node * 2u];
  let second = children[childrenOffset + node * 2u + 1u];
  // A cycle edge has one child and creates no death in the join tree.
  if (first == NONE || first == second) { return; }
  let firstPeak = elders[eldersOffset + first];
  let secondPeak = elders[eldersOffset + second];
  let firstHeight = values[valuesOffset + firstPeak];
  let secondHeight = values[valuesOffset + secondPeak];
  let firstWins = firstHeight > secondHeight || (firstHeight == secondHeight && firstPeak < secondPeak);
  let winner = select(secondPeak, firstPeak, firstWins);
  let loser = select(firstPeak, secondPeak, firstWins);
  // Each leaf loses at exactly one binary merge, so these writes are disjoint.
  result[resultOffset + loser] = getSpillValueBits(keys[keysOffset + index]);
  result[resultOffset + CELL_COUNT + loser] = winner;`,
    common
  );
  return nodes;
}
