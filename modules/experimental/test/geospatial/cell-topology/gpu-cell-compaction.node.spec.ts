// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {compactCells, getPentagons, polygonToCells, uncompactCells} from 'h3-js';
import {expect, it} from 'vitest';
import {
  GPUCellCompaction,
  type GPUCellCompactionProps
} from '../../../src/geospatial/cell-topology/gpu-cell-compaction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  bigIntToH3,
  h3ToBigInt,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';
import {
  compactCellsOnCPU,
  compareBigInt,
  getCellDescendants,
  uncompactCellsOnCPU
} from './cell-compaction-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellCompactionProps> = {}
): GPUCellCompactionProps {
  const view = <Format extends 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    family: 'quadbin',
    operation: {type: 'compact', resolution: 6},
    cells: view('uint32x2', 32),
    output: {
      cells: view('uint32x2', 16),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    ...overrides
  };
}

it('GPUCellCompaction validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const expectThrows = (overrides: Partial<GPUCellCompactionProps>, message: RegExp) =>
    expect(() => new GPUCellCompaction(createProps(graph, overrides))).toThrow(message);
  expectThrows({operation: {type: 'compact', resolution: 27}}, /resolution/);
  expectThrows({operation: {type: 'compact', resolution: 16}, family: 'h3'}, /resolution/);
  expectThrows(
    {operation: {type: 'compact', resolution: 6, minimumResolution: 7}},
    /minimumResolution/
  );
  expectThrows(
    {operation: {type: 'uncompact', resolution: 6, maximumDepth: 12}, family: 'h3'},
    /maximumDepth/
  );
  expectThrows({operation: {type: 'uncompact', resolution: 6, maximumDepth: -1}}, /maximumDepth/);
  expectThrows({cells: createTransientView(graph, 'bad-cells', 'uint32', 8) as never}, /cells/);
  expectThrows({mask: createTransientView(graph, 'bad-mask', 'uint32', 31)}, /mask/);
  expectThrows({wordOrder: 'big' as never}, /wordOrder/);
  const cells = createTransientView(graph, 'alias-cells', 'uint32x2', 8);
  expectThrows(
    {
      cells,
      output: {
        cells,
        count: createTransientView(graph, 'alias-count', 'uint32', 1),
        overflow: createTransientView(graph, 'alias-overflow', 'uint32', 1)
      }
    },
    /input|alias|disjoint|overlap/i
  );
  expectThrows(
    {
      output: {
        ...createProps(graph).output,
        count: createTransientView(graph, 'empty-count', 'uint32', 0)
      }
    },
    /one uint32 row/
  );
  device.destroy();
});

it('GPUCellCompaction emits deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const getIds = (overrides: Partial<GPUCellCompactionProps>) => {
    const graph = new GPUCommandGraph(device);
    const recipe = new GPUCellCompaction(createProps(graph, {id: 'compaction', ...overrides}));
    return recipe.getCommandNodes(graph).map(node => node.id);
  };
  const compact = getIds({});
  expect(getIds({})).toEqual(compact);
  expect(compact[0]).toBe('compaction-keys');
  expect(compact.at(-1)).toBe('compaction-emit');
  expect(compact).toContain('compaction-analyze');
  expect(compact).toContain('compaction-scatter');
  const sortedIds = getIds({operation: {type: 'compact', resolution: 6, sorted: true}});
  expect(sortedIds.some(id => id.startsWith('compaction-table-sort'))).toBe(false);
  expect(compact.some(id => id.startsWith('compaction-table-sort'))).toBe(true);
  const uncompact = getIds({operation: {type: 'uncompact', resolution: 8}});
  expect(uncompact[0]).toBe('compaction-count');
  expect(uncompact.at(-1)).toBe('compaction-expand');
  expect(getIds({operation: {type: 'uncompact', resolution: 8}})).toEqual(uncompact);
  device.destroy();
});

it('the compaction oracle matches h3-js compactCells and uncompactCells', () => {
  const polygon = [
    [37.5, -122.8],
    [38.1, -122.8],
    [38.1, -122.0],
    [37.5, -122.0]
  ];
  for (const resolution of [5, 6, 7]) {
    const cells = polygonToCells(polygon, resolution);
    const compacted = compactCells(cells).map(h3ToBigInt).sort(compareBigInt);
    expect(compactCellsOnCPU('h3', cells.map(h3ToBigInt), resolution)).toEqual(compacted);
    expect(compacted.length).toBeLessThan(cells.length);
    const expanded = uncompactCells(compactCells(cells), resolution).map(h3ToBigInt);
    expect(uncompactCellsOnCPU('h3', compacted, resolution).sort(compareBigInt)).toEqual(
      expanded.sort(compareBigInt)
    );
  }
  // Pentagon: only its 6 children (no K child) compact.
  const [pentagon] = getPentagons(4);
  const children = getCellDescendants('h3', h3ToBigInt(pentagon), 5);
  expect(children).toHaveLength(6);
  expect(compactCellsOnCPU('h3', children, 5)).toEqual([h3ToBigInt(pentagon)]);
  expect(bigIntToH3(compactCellsOnCPU('h3', children, 5, 5)[0])).toBe(bigIntToH3(children[0]));
});

it('the Quadbin oracle compacts complete blocks and keeps partial ones', () => {
  const block: bigint[] = [];
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) {
      block.push(quadbinTileToCell(4 + x, 8 + y, 5));
    }
  }
  expect(compactCellsOnCPU('quadbin', block, 5)).toEqual([quadbinTileToCell(1, 2, 3)]);
  expect(compactCellsOnCPU('quadbin', block, 5, 4)).toEqual(
    getCellDescendants('quadbin', quadbinTileToCell(1, 2, 3), 4)
  );
  const partial = block.slice(1);
  // Dropping one tile keeps its 3 siblings, the other 3 sibling groups become res 4 parents.
  const compactedPartial = compactCellsOnCPU('quadbin', partial, 5);
  expect(compactedPartial).toHaveLength(6);
  expect(compactCellsOnCPU('quadbin', [...block, ...block], 5)).toEqual([
    quadbinTileToCell(1, 2, 3)
  ]);
});
