// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellToChildren, getPentagons, getRes0Cells, gridDisk, gridDiskDistances} from 'h3-js';
import {expect, it} from 'vitest';
import {H3_NEIGHBOR_WGSL} from '../../../src/geospatial/cell-topology/h3-neighbor-wgsl';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {bigIntToH3, h3ToBigInt} from '../cell-aggregation/cell-aggregation-oracle';
import {getH3DiskByBreadthFirstSearch, getH3Neighbor, isH3Pentagon} from './h3-neighbor-oracle';
import {getH3TestCells} from './h3-neighbor-test-cells';

/** Number of cells whose six-direction neighbor set differs from h3-js `gridDisk(cell, 1)`. */
function countNeighborSetMismatches(cells: readonly string[]): number {
  let mismatches = 0;
  for (const cell of cells) {
    const expected = gridDisk(cell, 1)
      .filter(neighbor => neighbor !== cell)
      .sort();
    const actual: string[] = [];
    for (let direction = 1; direction <= 6; direction++) {
      const neighbor = getH3Neighbor(h3ToBigInt(cell), direction);
      if (neighbor !== 0n) {
        actual.push(bigIntToH3(neighbor));
      }
    }
    actual.sort();
    if (actual.join() !== expected.join()) {
      mismatches++;
    }
  }
  return mismatches;
}

it('H3 neighbor oracle matches h3-js gridDisk(cell, 1) for every cell at resolutions 0-3', () => {
  const cells = getRes0Cells().flatMap(baseCell => [
    baseCell,
    ...[1, 2, 3].flatMap(resolution => cellToChildren(baseCell, resolution))
  ]);
  expect(cells.length).toBe(122 + 842 + 5882 + 41162);
  expect(countNeighborSetMismatches(cells)).toBe(0);
});

it('H3 neighbor oracle matches h3-js at resolutions 4-15, pentagons and their neighborhoods', () => {
  const cells = getH3TestCells(createRandom(7), 1500);
  expect(countNeighborSetMismatches(cells)).toBe(0);
  for (let resolution = 0; resolution <= 15; resolution++) {
    for (const pentagon of getPentagons(resolution)) {
      const cell = h3ToBigInt(pentagon);
      expect(isH3Pentagon(cell)).toBe(true);
      const neighbors = [1, 2, 3, 4, 5, 6].map(direction => getH3Neighbor(cell, direction));
      // The K direction (1) is deleted at a pentagon.
      expect(neighbors[0]).toBe(0n);
      expect(new Set(neighbors.filter(neighbor => neighbor !== 0n)).size).toBe(5);
    }
  }
});

it('H3 breadth-first disks equal h3-js gridDiskDistances near pentagons and base cell edges', () => {
  const random = createRandom(11);
  const origins = [
    ...[2, 5, 9].flatMap(resolution => getPentagons(resolution)),
    ...[2, 5, 9].flatMap(resolution =>
      getPentagons(resolution).flatMap(pentagon => gridDisk(pentagon, 2))
    ),
    ...getH3TestCells(random, 40)
  ];
  for (const origin of origins) {
    for (const k of [1, 2, 3]) {
      const expected = new Map<string, number>();
      gridDiskDistances(origin, k).forEach((ring, distance) => {
        for (const cell of ring) {
          expected.set(cell, distance);
        }
      });
      const actual = getH3DiskByBreadthFirstSearch(h3ToBigInt(origin), k);
      const format = (entries: [string, number][]) =>
        entries.sort((left, right) => (left[0] < right[0] ? -1 : 1)).join(';');
      expect(
        format([...actual].map(([cell, distance]) => [bigIntToH3(cell), distance])),
        `${origin} k=${k}`
      ).toBe(format([...expected]));
    }
  }
});

it('H3_NEIGHBOR_WGSL declares the contract entry points with prefixed identifiers', () => {
  expect(H3_NEIGHBOR_WGSL).toContain(
    'fn cellTopologyH3Neighbor(cell: vec2u, direction: u32) -> vec2u'
  );
  expect(H3_NEIGHBOR_WGSL).toContain('fn cellTopologyH3IsPentagon(cell: vec2u) -> bool');
  const declared = [...H3_NEIGHBOR_WGSL.matchAll(/^(?:fn|const) (\w+)/gm)].map(match => match[1]);
  expect(declared.length).toBeGreaterThan(5);
  for (const name of declared) {
    expect(name.startsWith('cellTopologyH3') || name.startsWith('CELL_TOPOLOGY_H3_')).toBe(true);
  }
});
