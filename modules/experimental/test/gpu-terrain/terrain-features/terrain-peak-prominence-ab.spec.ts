// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getPeakProminenceMergeNodes} from '../../../src/gpu-terrain/terrain-features/terrain-peak-prominence-tree';
import {getSpillCommonWGSL} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-utils';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getProminenceEdges} from './terrain-peak-prominence-model';

type MergeInput = {
  width: number;
  height: number;
  values: GraphDataView<'float32'>;
  sortedKeys: GraphDataView<'uint32'>;
  sortedEdges: GraphDataView<'uint32'>;
  result: GraphDataView<'uint32'>;
};

/** Frozen pre-parallel GPU union baseline; test-only, with the original 4096-edge batches. */
function getSerialMergeNodes(graph: GPUCommandGraph, props: MergeInput): GPUCommandNode[] {
  const count = props.width * props.height;
  const forest = createTransientView(graph, 'serial-forest', 'uint32', count * 3);
  const cursor = createTransientView(graph, 'serial-cursor', 'uint32', 1);
  const gate = createWGSLDispatchGate(graph, {
    id: 'serial-dispatch',
    operation: 'ProminenceSerialBaseline',
    shapes: [[1, 64]]
  });
  const common = `${getSpillCommonWGSL(props.width, props.height)}\nconst CELL_COUNT: u32 = ${count}u;`;
  const nodes: GPUCommandNode[] = [
    createWGSLKernelNode(graph, {
      id: 'serial-init',
      operation: 'ProminenceSerialBaseline',
      invocationCount: count,
      bindings: [
        {name: 'forest', view: forest, type: 'u32', access: 'read_write'},
        {name: 'cursor', view: cursor, type: 'u32', access: 'read_write'},
        gate.binding
      ],
      declarations: common,
      body: `forest[forestOffset + index] = index;
      forest[forestOffset + CELL_COUNT + index] = 1u;
      forest[forestOffset + 2u * CELL_COUNT + index] = index;
      if (index == 0u) { cursor[cursorOffset] = 0u; ${gate.armWGSL} }`
    })
  ];
  for (let batch = 0; batch < Math.ceil((count * 4) / 4096); batch++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `serial-union-${batch}`,
        operation: 'ProminenceSerialBaseline',
        invocationCount: 1,
        workgroupSize: 64,
        ...gate.getGate(0),
        bindings: [
          {name: 'keys', view: props.sortedKeys, type: 'u32', access: 'read'},
          {name: 'edges', view: props.sortedEdges, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'forest', view: forest, type: 'u32', access: 'read_write'},
          {name: 'result', view: props.result, type: 'u32', access: 'read_write'},
          {name: 'cursor', view: cursor, type: 'u32', access: 'read_write'}
        ],
        declarations: `${common}
fn findRoot(cell: u32) -> u32 {
  var root = cell;
  for (var depth = 0u; depth < 32u; depth++) {
    let parent = forest[forestOffset + root];
    if (parent == root) { break; } root = parent;
  }
  return root;
}`,
        body: `let begin = cursor[cursorOffset];
  let end = min(begin + 4096u, CELL_COUNT * 4u);
  for (var slot = begin; slot < end; slot++) {
    if (keys[keysOffset + slot] == 0u) { cursor[cursorOffset] = CELL_COUNT * 4u; return; }
    let edge = edges[edgesOffset + slot];
    let first = edge / 4u; let second = getSpillNeighbor(first, edge % 4u);
    var firstRoot = findRoot(first); var secondRoot = findRoot(second);
    if (firstRoot == secondRoot) { continue; }
    let firstPeak = forest[forestOffset + 2u * CELL_COUNT + firstRoot];
    let secondPeak = forest[forestOffset + 2u * CELL_COUNT + secondRoot];
    let firstHeight = values[valuesOffset + firstPeak];
    let secondHeight = values[valuesOffset + secondPeak];
    let firstWins = firstHeight > secondHeight || (firstHeight == secondHeight && firstPeak < secondPeak);
    let winner = select(secondPeak, firstPeak, firstWins);
    let loser = select(firstPeak, secondPeak, firstWins);
    result[resultOffset + loser] = getSpillValueBits(keys[keysOffset + slot]);
    result[resultOffset + CELL_COUNT + loser] = winner;
    if (forest[forestOffset + CELL_COUNT + firstRoot] < forest[forestOffset + CELL_COUNT + secondRoot]) {
      let temporary = firstRoot; firstRoot = secondRoot; secondRoot = temporary;
    }
    forest[forestOffset + secondRoot] = firstRoot;
    forest[forestOffset + CELL_COUNT + firstRoot] += forest[forestOffset + CELL_COUNT + secondRoot];
    forest[forestOffset + 2u * CELL_COUNT + firstRoot] = winner;
  }
  cursor[cursorOffset] = end;`
      }),
      createWGSLKernelNode(graph, {
        id: `serial-gate-${batch}`,
        operation: 'ProminenceSerialBaseline',
        invocationCount: 1,
        bindings: [{name: 'cursor', view: cursor, type: 'u32', access: 'read'}, gate.binding],
        body: `if (cursor[cursorOffset] >= ${count * 4}u) { ${gate.disarmWGSL} }`
      })
    );
  }
  return nodes;
}

async function runMerge(
  device: Device,
  input: Parameters<typeof getProminenceEdges>[0],
  parallel: boolean
) {
  // Test-only pre-sorting isolates the changed merge stages from the unchanged radix sort.
  const {values, edges} = getProminenceEdges(input);
  const count = values.length;
  const keys = new Uint32Array(count * 4);
  const order = new Uint32Array(count * 4);
  const floatBits = new Float32Array(1);
  const uintBits = new Uint32Array(floatBits.buffer);
  edges.forEach((edge, rank) => {
    floatBits[0] = edge.saddle === 0 ? 0 : edge.saddle;
    keys[rank] = uintBits[0] & 0x80000000 ? ~uintBits[0] : uintBits[0] ^ 0x80000000;
    const columnOffset = (edge.second % input.width) - (edge.first % input.width);
    const rowOffset = Math.floor(edge.second / input.width) - Math.floor(edge.first / input.width);
    const direction = rowOffset === 0 ? 0 : columnOffset === 1 ? 1 : columnOffset === 0 ? 2 : 3;
    order[rank] = edge.first * 4 + direction;
  });
  const buffers = [
    createInputBuffer(device, Float32Array.from(values)),
    createInputBuffer(device, keys),
    createInputBuffer(device, order),
    createInputBuffer(device, new Uint32Array(count).fill(1)),
    createOutputBuffer(device, count * 2)
  ];
  const graph = new GPUCommandGraph(device);
  const props = {
    id: 'parallel',
    width: input.width,
    height: input.height,
    values: importGraphBuffer(graph, 'values', buffers[0], 'float32', count),
    sortedKeys: importGraphBuffer(graph, 'keys', buffers[1], 'uint32', count * 4),
    sortedEdges: importGraphBuffer(graph, 'edges', buffers[2], 'uint32', count * 4),
    validity: importGraphBuffer(graph, 'validity', buffers[3], 'uint32', count),
    result: importGraphBuffer(graph, 'result', buffers[4], 'uint32', count * 2)
  };
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'result-init',
      operation: 'ProminenceMergeBenchmark',
      invocationCount: count,
      bindings: [{name: 'result', view: props.result, type: 'u32', access: 'read_write'}],
      body: `result[resultOffset + index] = 0xff800000u;
      result[resultOffset + ${count}u + index] = 0xffffffffu;`
    })
  );
  for (const node of parallel
    ? getPeakProminenceMergeNodes(graph, props)
    : getSerialMergeNodes(graph, props))
    graph.add(node);
  const compiled = graph.compile();
  const times: number[] = [];
  try {
    let result: number[] = [];
    for (let repetition = 0; repetition < 4; repetition++) {
      const begin = performance.now();
      submitGraph(device, compiled, undefined);
      result = await readUint32(buffers[4], count * 2);
      if (repetition) times.push(performance.now() - begin);
    }
    times.sort((first, second) => first - second);
    return {result, milliseconds: times[1]};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

// Run only in the main session's serial GPU pass, after correctness specs.
it('prominence merge A/B: exact serial/parallel saddles and parents on ridges and cyclic plateaus', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const count of [20_000, 100_000]) {
    for (const plateau of [false, true]) {
      const width = plateau ? 200 : count;
      const input = {
        width,
        height: count / width,
        elevation: Array.from({length: count}, (_, cell) => (plateau ? 7 : cell))
      };
      const serial = await runMerge(device, input, false);
      const parallel = await runMerge(device, input, true);
      expect(parallel.result).toEqual(serial.result);
      // eslint-disable-next-line no-console
      console.log(
        `prominence-merge-ab cells=${count} plateau=${plateau} serial=${serial.milliseconds.toFixed(2)} ms parallel=${parallel.milliseconds.toFixed(2)} ms (warm submit + readback)`
      );
    }
  }
}, 900000);
