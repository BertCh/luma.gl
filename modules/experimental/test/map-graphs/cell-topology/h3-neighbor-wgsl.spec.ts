// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellToChildren, getRes0Cells} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {H3_NEIGHBOR_WGSL} from '../../../src/map-graphs/cell-topology/h3-neighbor-wgsl';
import {createMapGraphKernelNode} from '../../../src/map-graphs/map-graph-kernels';
import {h3ToBigInt, joinCellKey, splitCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {createInputBuffer, createOutputBuffer, readUint32} from '../map-graph-test-utils';
import {getH3Neighbor, isH3Pentagon} from './h3-neighbor-oracle';
import {getH3TestCells} from './h3-neighbor-test-cells';

it('H3_NEIGHBOR_WGSL matches the CPU port bit for bit in every direction', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cells = [
    ...getRes0Cells().flatMap(baseCell => [baseCell, ...cellToChildren(baseCell, 2)]),
    ...getH3TestCells(createRandom(3), 600)
  ].map(h3ToBigInt);
  // Invalid input yields zero in every direction.
  cells.push(0n, 0xffffffffffffffffn);
  const rowCount = cells.length;
  const cellWords = new Uint32Array(rowCount * 2);
  cells.forEach((cell, row) => cellWords.set(splitCellKey(cell), row * 2));

  const graph = new GPUCommandGraph(device, {id: 'h3-neighbor-graph'});
  const outputBuffer = createOutputBuffer(device, rowCount * 7 * 2);
  const pentagonBuffer = createOutputBuffer(device, rowCount);
  graph.add(
    createMapGraphKernelNode(graph, {
      id: 'h3-neighbor-kernel',
      operation: 'H3NeighborTest',
      invocationCount: rowCount * 7,
      bindings: [
        {
          name: 'cells',
          view: importGraphBuffer(graph, 'cells', createInputBuffer(device, cellWords), 'uint32'),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'neighbors',
          view: importGraphBuffer(graph, 'neighbors', outputBuffer, 'uint32'),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'pentagons',
          view: importGraphBuffer(graph, 'pentagons', pentagonBuffer, 'uint32'),
          type: 'u32',
          access: 'read_write'
        }
      ],
      declarations: `${dggs.source}\n${H3_NEIGHBOR_WGSL}`,
      body: `let row = index / 7u;
  let direction = index % 7u;
  let cell = vec2u(cells[cellsOffset + row * 2u + 1u], cells[cellsOffset + row * 2u]);
  let neighbor = cellTopologyH3Neighbor(cell, direction);
  neighbors[neighborsOffset + index * 2u] = neighbor.y;
  neighbors[neighborsOffset + index * 2u + 1u] = neighbor.x;
  if (direction == 0u) {
    pentagons[pentagonsOffset + row] = select(0u, 1u, cellTopologyH3IsPentagon(cell));
  }`
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const words = await readUint32(outputBuffer, rowCount * 14);
  const pentagons = await readUint32(pentagonBuffer, rowCount);
  compiled.destroy?.();

  // Guard against a kernel that never ran: most rows have six nonzero neighbors.
  expect(words.filter(word => word !== 0).length).toBeGreaterThan(rowCount * 10);
  let mismatches = 0;
  for (let row = 0; row < rowCount; row++) {
    const cell = cells[row];
    const isValid = row < rowCount - 2;
    expect(pentagons[row] === 1, `pentagon ${cell.toString(16)}`).toBe(
      isValid && isH3Pentagon(cell)
    );
    for (let direction = 0; direction < 7; direction++) {
      const index = row * 7 + direction;
      const actual = joinCellKey(words[index * 2], words[index * 2 + 1]);
      // Direction 0 is not a neighbor direction.
      const expected = isValid && direction > 0 ? getH3Neighbor(cell, direction) : 0n;
      if (actual !== expected) {
        mismatches++;
      }
    }
  }
  expect(mismatches).toBe(0);
});
