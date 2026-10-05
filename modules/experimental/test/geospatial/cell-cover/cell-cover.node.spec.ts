// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getPentagons, cellToLatLng} from 'h3-js';
import {expect, it} from 'vitest';
import {GPUCellCover, type GPUCellCoverProps} from '../../../src/geospatial/cell-cover';
import {quadbinCellToTile} from '../cell-aggregation/cell-aggregation-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  countCoverDisagreements,
  coverH3LatticeOnCPU,
  coverH3OnCPU,
  coverQuadbinCenterFloat64,
  coverQuadbinOnCPU,
  createRectangleRing,
  createStarPolygon,
  flattenCoverFeatures,
  roundFeatures,
  type CoverFeature
} from './cell-cover-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUCellCoverProps> = {}
): GPUCellCoverProps {
  const view = <Format extends 'uint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    family: 'quadbin',
    resolution: 5,
    polygonPositions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2),
    candidateCapacity: 64,
    output: {
      featureIds: view('uint32', 16),
      cells: view('uint32x2', 16),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    ...overrides
  };
}

it('GPUCellCover validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'cell-cover-validation'});
  const view = <Format extends 'uint32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `v-${serial++}`, format, length);
  expect(() => new GPUCellCover(createProps(graph)).getCommandNodes(graph)).not.toThrow();
  expect(new GPUCellCover(createProps(graph)).id).toBe('cell-cover');
  expect(() => new GPUCellCover(createProps(graph, {resolution: 27}))).toThrow(/resolution/);
  expect(() => new GPUCellCover(createProps(graph, {family: 'h3', resolution: 16}))).toThrow(
    /resolution/
  );
  expect(() => new GPUCellCover(createProps(graph, {family: 'h3', containment: 'full'}))).toThrow(
    /center/
  );
  expect(() => new GPUCellCover(createProps(graph, {candidateCapacity: 0}))).toThrow(
    /candidateCapacity/
  );
  expect(
    () =>
      new GPUCellCover(
        createProps(graph, {
          output: {
            featureIds: view('uint32', 4),
            cells: view('uint32x2', 5),
            count: view('uint32', 1),
            overflow: view('uint32', 1)
          }
        })
      )
  ).toThrow(/equal lengths/);
  expect(() => new GPUCellCover(createProps(graph, {featureIds: view('uint32', 3)}))).toThrow(
    /featureIds/
  );
  const shared = createProps(graph);
  expect(
    () =>
      new GPUCellCover({
        ...shared,
        output: {...shared.output, count: shared.featureOffsets}
      })
  ).toThrow(/share/);
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'other'});
  expect(() => new GPUCellCover(createProps(graph)).getCommandNodes(otherGraph)).toThrow(
    /target graph/
  );
});

it('GPUCellCover returns deterministic node ids', () => {
  const ids = (id?: string) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'cell-cover-ids'});
    const nodes = new GPUCellCover(createProps(graph, {id})).getCommandNodes(graph);
    return nodes.map(node => node.id);
  };
  const first = ids();
  expect(first).toEqual(ids());
  expect(first.every(id => id.startsWith('cell-cover-'))).toBe(true);
  expect(first).toContain('cell-cover-count');
  expect(first).toContain('cell-cover-test');
  expect(first).toContain('cell-cover-write');
  expect(first).toContain('cell-cover-publish');
  expect(ids('polyfill').every(id => id.startsWith('polyfill-'))).toBe(true);
  expect(new Set(first).size).toBe(first.length);
});

it('Quadbin oracle: full tile rectangle covers exactly that tile; modes nest', () => {
  const [west, south, east, north] = [8.0, 46.0, 9.0, 47.0];
  const features: CoverFeature[] = roundFeatures([
    [[createRectangleRing([west, south, east, north])]]
  ]);
  const arrays = flattenCoverFeatures(features);
  const center = coverQuadbinOnCPU(arrays, 6, 'center');
  const full = coverQuadbinOnCPU(arrays, 6, 'full');
  const intersects = coverQuadbinOnCPU(arrays, 6, 'intersects');
  const key = (cells: bigint[]) => new Set(cells);
  for (const cell of full.cells) {
    expect(key(intersects.cells).has(cell)).toBe(true);
  }
  for (const cell of center.cells) {
    expect(key(intersects.cells).has(cell)).toBe(true);
  }
  expect(full.cells.length).toBeLessThanOrEqual(intersects.cells.length);
  // Every full tile lies inside the rectangle.
  for (const cell of full.cells) {
    const {x, z} = quadbinCellToTile(cell);
    const tileWest = (x * 360) / 2 ** z - 180;
    expect(tileWest).toBeGreaterThanOrEqual(west - 1e-4);
    expect(((x + 1) * 360) / 2 ** z - 180).toBeLessThanOrEqual(east + 1e-4);
  }
  // A polygon exactly equal to one tile (f32 exact edges) is covered fully by that tile.
  const tileWest = (20 * 360) / 64 - 180;
  const tileEast = (21 * 360) / 64 - 180;
  const tileNorth = (Math.atan(Math.sinh(Math.PI * (1 - (2 * 20) / 64))) * 180) / Math.PI;
  const tileSouth = (Math.atan(Math.sinh(Math.PI * (1 - (2 * 21) / 64))) * 180) / Math.PI;
  const inner = roundFeatures([
    [[createRectangleRing([tileWest + 0.01, tileSouth + 0.01, tileEast - 0.01, tileNorth - 0.01])]]
  ]);
  const innerArrays = flattenCoverFeatures(inner);
  expect(coverQuadbinOnCPU(innerArrays, 6, 'full').cells).toEqual([]);
  expect(coverQuadbinOnCPU(innerArrays, 6, 'center').cells.length).toBe(1);
  expect(coverQuadbinOnCPU(innerArrays, 6, 'intersects').cells.length).toBe(1);
});

it('Quadbin oracle: f32 mirror and independent f64 center oracle agree', () => {
  const features = roundFeatures([
    [[createStarPolygon(3, [8.3, 47.1], 2.5, 17)]],
    [[createStarPolygon(9, [-100.2, 40.4], 6, 23)]]
  ]);
  const arrays = flattenCoverFeatures(features);
  for (const resolution of [6, 8, 10]) {
    const mirror = coverQuadbinOnCPU(arrays, resolution, 'center');
    const exact = coverQuadbinCenterFloat64(arrays, resolution);
    expect(countCoverDisagreements(mirror, exact), `resolution ${resolution}`).toBeLessThanOrEqual(
      2
    );
    expect(mirror.cells.length).toBeGreaterThan(0);
  }
});

it('H3 lattice algorithm (f64 simulation) matches h3-js polygonToCells', () => {
  const pentagon = getPentagons(5)[0];
  const [pentagonLat, pentagonLng] = cellToLatLng(pentagon);
  const features = roundFeatures([
    [[createStarPolygon(5, [8.5, 47.2], 0.6, 13)]],
    [
      [
        createRectangleRing([-100.5, 39.5, -99.0, 40.7]),
        createRectangleRing([-100.0, 39.9, -99.6, 40.3]).reverse()
      ]
    ],
    [[createStarPolygon(7, [20, 70], 0.8, 11)]],
    [[createStarPolygon(11, [pentagonLng, pentagonLat], 0.7, 15)]]
  ]);
  const arrays = flattenCoverFeatures(features);
  let totalDisagreements = 0;
  let totalCells = 0;
  for (const resolution of [3, 4, 5, 6, 7, 8]) {
    const lattice = coverH3LatticeOnCPU(arrays, resolution);
    const expected = coverH3OnCPU(features, resolution);
    const disagreements = countCoverDisagreements(lattice, expected);
    totalDisagreements += disagreements;
    totalCells += expected.cells.length;
    // Each cell is emitted exactly once.
    const keys = lattice.featureRows.map((feature, index) => `${feature}:${lattice.cells[index]}`);
    expect(new Set(keys).size).toBe(keys.length);
  }
  expect(totalCells).toBeGreaterThan(100);
  // eslint-disable-next-line no-console
  console.log(
    `H3 lattice f64 simulation: ${totalDisagreements} disagreements over ${totalCells} cells`
  );
  expect(totalDisagreements).toBe(0);
});
