// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getCellTopologyStride,
  GPUCellTopology,
  type GPUCellTopologyOperation
} from '../../../src/geospatial/cell-topology/gpu-cell-topology';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getCellTopologyRowOnCPU} from './cell-topology-oracle';
import {quadbinTileToCell} from '../cell-aggregation/cell-aggregation-oracle';

let serial = 0;
const view = <Format extends 'uint32' | 'uint32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) => createTransientView(graph, `v-${serial++}`, format, length);

function create(
  family: 'quadbin' | 'h3',
  operation: GPUCellTopologyOperation,
  rows = 4,
  extra: {distances?: boolean; counts?: boolean; id?: string} = {}
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'g'});
  const stride = getCellTopologyStride(family, operation);
  const contributor = new GPUCellTopology({
    id: extra.id,
    family,
    operation,
    cells: view(graph, 'uint32x2', rows),
    output: {
      cells: view(graph, 'uint32x2', rows * stride),
      distances: extra.distances ? view(graph, 'uint32', rows * stride) : undefined,
      counts: extra.counts ? view(graph, 'uint32', rows) : undefined
    }
  });
  return {graph, contributor, stride};
}

it('GPUCellTopology strides', () => {
  expect(getCellTopologyStride('h3', {type: 'disk', k: 0})).toBe(1);
  expect(getCellTopologyStride('h3', {type: 'disk', k: 1})).toBe(7);
  expect(getCellTopologyStride('h3', {type: 'disk', k: 4})).toBe(61);
  expect(getCellTopologyStride('h3', {type: 'disk', k: 8})).toBe(217);
  expect(getCellTopologyStride('quadbin', {type: 'disk', k: 2})).toBe(25);
  expect(getCellTopologyStride('h3', {type: 'ring', k: 0})).toBe(1);
  expect(getCellTopologyStride('h3', {type: 'ring', k: 3})).toBe(18);
  expect(getCellTopologyStride('quadbin', {type: 'ring', k: 3})).toBe(24);
  expect(getCellTopologyStride('quadbin', {type: 'parent', resolution: 3})).toBe(1);
  expect(getCellTopologyStride('h3', {type: 'children', resolution: 7, inputResolution: 5})).toBe(
    49
  );
  expect(
    getCellTopologyStride('quadbin', {type: 'children', resolution: 8, inputResolution: 2})
  ).toBe(4096);
  expect(() =>
    getCellTopologyStride('quadbin', {type: 'children', resolution: 9, inputResolution: 2})
  ).toThrow(/stride/);
  expect(() =>
    getCellTopologyStride('h3', {type: 'children', resolution: 9, inputResolution: 2})
  ).toThrow(/stride/);
  expect(() => getCellTopologyStride('h3', {type: 'disk', k: 9})).toThrow(/k must/);
  expect(() => getCellTopologyStride('h3', {type: 'ring', k: -1})).toThrow(/k must/);
  expect(() => getCellTopologyStride('h3', {type: 'parent', resolution: 16})).toThrow();
  expect(() =>
    getCellTopologyStride('h3', {type: 'children', resolution: 3, inputResolution: 5})
  ).toThrow(/below/);
});

it('GPUCellTopology emits one deterministic node per operation', () => {
  for (const operation of [
    {type: 'disk', k: 2},
    {type: 'ring', k: 2},
    {type: 'parent', resolution: 3},
    {type: 'children', resolution: 6, inputResolution: 5}
  ] as const) {
    for (const family of ['quadbin', 'h3'] as const) {
      const {graph, contributor} = create(family, operation, 3, {counts: true, id: 'topo'});
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.map(node => node.id)).toEqual(['topo-topology']);
    }
  }
  const {graph, contributor} = create('h3', {type: 'disk', k: 1});
  expect(contributor.id).toBe('cell-topology');
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'cell-topology-topology'
  ]);
});

it('GPUCellTopology validates inputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'g'});
  const base = {
    family: 'h3' as const,
    operation: {type: 'disk', k: 1} as const,
    cells: view(graph, 'uint32x2', 2),
    output: {cells: view(graph, 'uint32x2', 14)}
  };
  expect(() => new GPUCellTopology(base)).not.toThrow();
  expect(() => new GPUCellTopology({...base, family: 'geohash' as never})).toThrow(/family/);
  expect(
    () => new GPUCellTopology({...base, output: {cells: view(graph, 'uint32x2', 13)}})
  ).toThrow(/stride/);
  expect(() => new GPUCellTopology({...base, cells: view(graph, 'uint32', 2) as never})).toThrow();
  expect(() => new GPUCellTopology({...base, mask: view(graph, 'uint32', 3)})).toThrow(
    /mask length/
  );
  expect(
    () =>
      new GPUCellTopology({
        ...base,
        output: {cells: view(graph, 'uint32x2', 14), distances: view(graph, 'uint32', 13)}
      })
  ).toThrow(/distances length/);
  expect(
    () =>
      new GPUCellTopology({
        ...base,
        operation: {type: 'ring', k: 1},
        output: {cells: view(graph, 'uint32x2', 12), distances: view(graph, 'uint32', 12)}
      })
  ).toThrow(/distances/);
  expect(
    () =>
      new GPUCellTopology({
        ...base,
        output: {cells: view(graph, 'uint32x2', 14), counts: view(graph, 'uint32', 3)}
      })
  ).toThrow(/counts length/);
  expect(() => new GPUCellTopology({...base, output: {cells: base.cells}})).toThrow();
  expect(() => new GPUCellTopology({...base, operation: {type: 'disk', k: 9}})).toThrow(/k must/);
  // Foreign views are rejected at node creation.
  const other = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'other'});
  expect(() => new GPUCellTopology(base).getCommandNodes(other)).toThrow();
});

it('GPUCellTopology oracle sanity (quadbin wrap, clipping, pentagons)', () => {
  // z = 1: the 3x3 square around (0, 0) wraps onto 2 columns and is clipped to 2 rows.
  const corner = quadbinTileToCell(0, 0, 1);
  const disk = getCellTopologyRowOnCPU('quadbin', {type: 'disk', k: 1}, corner);
  expect(disk.cells.length).toBe(4);
  expect(disk.distances).toEqual([0, 1, 1, 1]);
  const zoomZero = quadbinTileToCell(0, 0, 0);
  expect(getCellTopologyRowOnCPU('quadbin', {type: 'disk', k: 3}, zoomZero).cells).toEqual([
    zoomZero
  ]);
  const middle = quadbinTileToCell(5, 5, 4);
  expect(getCellTopologyRowOnCPU('quadbin', {type: 'ring', k: 2}, middle).cells.length).toBe(16);
  // The first res-0 pentagon (base cell 4) has 5 neighbors and 6 children at res 1.
  const pentagon = 0x8009fffffffffffn;
  expect(getCellTopologyRowOnCPU('h3', {type: 'ring', k: 1}, pentagon).cells.length).toBe(5);
  expect(
    getCellTopologyRowOnCPU('h3', {type: 'children', resolution: 1, inputResolution: 0}, pentagon)
      .cells.length
  ).toBe(6);
  expect(getCellTopologyRowOnCPU('h3', {type: 'disk', k: 1}, 0n).cells).toEqual([]);
});
