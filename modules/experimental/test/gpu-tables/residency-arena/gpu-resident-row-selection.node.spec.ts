// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUResidentRowSelection,
  type GPUResidentRowSelectionProps
} from '../../../src/gpu-tables/residency-arena/gpu-resident-row-selection';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

type OptionSet = {
  tiles?: boolean;
  predicates?: number;
  sourceIds?: boolean;
  outputMask?: boolean;
  drawInstanceCount?: boolean;
  requiredCount?: boolean;
};

function createProps(
  graph: GPUCommandGraph,
  rows: number,
  options: OptionSet = {},
  capacity: number = rows
): GPUResidentRowSelectionProps {
  return {
    liveMask: createTransientView(graph, 'live', 'uint32', rows),
    tileVisibility: options.tiles
      ? {
          rowTileSlots: createTransientView(graph, 'slots', 'uint32', rows),
          tileMask: createTransientView(graph, 'tile-mask', 'uint32', 16)
        }
      : undefined,
    additionalPredicates: Array.from({length: options.predicates ?? 0}, (_, index) => ({
      kind: 'time-range' as const,
      mask: createTransientView(graph, `predicate-${index}`, 'uint32', rows)
    })),
    sourceIds: options.sourceIds ? createTransientView(graph, 'source', 'uint32', rows) : undefined,
    outputMask: options.outputMask ? createTransientView(graph, 'mask', 'uint32', rows) : undefined,
    drawInstanceCount: options.drawInstanceCount
      ? createTransientView(graph, 'draw', 'uint32', 1)
      : undefined,
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', capacity),
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1),
      requiredCount: options.requiredCount
        ? createTransientView(graph, 'total', 'uint32', 1)
        : undefined
    }
  };
}

function getNodeIds(rows: number, options: OptionSet = {}, capacity: number = rows): string[] {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUResidentRowSelection(createProps(graph, rows, options, capacity))
    .getCommandNodes(graph)
    .map(node => node.id);
  device.destroy();
  return ids;
}

it('GPUResidentRowSelection has stable default id, and unique prefixed node IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUResidentRowSelection(
    createProps(graph, 8, {tiles: true, predicates: 1})
  );
  expect(contributor.id).toBe('resident-row-selection');
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('resident-row-selection-tile-expand');
  expect(ids.at(-1)).toBe('resident-row-selection-publish');
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.every(id => id.startsWith('resident-row-selection-'))).toBe(true);
  expect(getNodeIds(8, {tiles: true, predicates: 1})).toEqual(ids);

  const custom = new GPUCommandGraph(device);
  const customIds = new GPUResidentRowSelection({
    ...createProps(custom, 8),
    id: 'arena-a'
  })
    .getCommandNodes(custom)
    .map(node => node.id);
  expect(customIds.every(id => id.startsWith('arena-a-'))).toBe(true);
  device.destroy();
});

it('GPUResidentRowSelection node count depends only on the option set', () => {
  const optionSets: [string, OptionSet][] = [
    ['live only', {}],
    ['tiles', {tiles: true}],
    ['predicate', {predicates: 1}],
    ['tiles + predicate', {tiles: true, predicates: 1}],
    [
      'tiles + predicate + sourceIds + mask + draw + total',
      {
        tiles: true,
        predicates: 1,
        sourceIds: true,
        outputMask: true,
        drawInstanceCount: true,
        requiredCount: true
      }
    ]
  ];
  for (const [, options] of optionSets) {
    // The count is structural: it never reads buffer contents, so tile count, tile placement, and
    // live-mask contents cannot change it. Only the row capacity can, and only through the depth
    // of GPUCompaction's hierarchical scan (one more level, two more nodes, above 65,536 rows).
    const small = getNodeIds(1024, options);
    expect(getNodeIds(4096, options)).toEqual(small);
    expect(getNodeIds(65536, options)).toEqual(small);
    // Capacity smaller than rows only changes how the publish kernel copies.
    expect(getNodeIds(1024, options, 16).length).toBe(small.length);
    const large = getNodeIds(1048576, options);
    expect(large.length).toBe(small.length + 2);
    expect(large.filter(id => !id.includes('-scan-level-'))).toEqual(
      small.filter(id => !id.includes('-scan-level-'))
    );
  }
  expect(getNodeIds(1024, {}).length).toBe(7);
  expect(getNodeIds(1024, {tiles: true}).length).toBe(8);
  expect(getNodeIds(1024, {predicates: 1}).length).toBe(7);
  expect(getNodeIds(1024, {tiles: true, predicates: 2}).length).toBe(8);
  expect(getNodeIds(1024, {tiles: true}).length).toBe(getNodeIds(1024, {}).length + 1);
});

it('GPUResidentRowSelection validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph, 8);
  const create = (overrides: Partial<GPUResidentRowSelectionProps>) =>
    new GPUResidentRowSelection({...base, ...overrides});
  const view = (name: string, length: number = 8) =>
    createTransientView(graph, name, 'uint32', length);

  const vector = createVectorView('vec', 'uint32', [view('v0', 4), view('v1', 4)]);
  expect(() => create({liveMask: vector as never})).toThrow(/packed GraphDataView/);
  expect(() => create({sourceIds: vector as never})).toThrow(/packed GraphDataView/);
  expect(() => create({outputMask: vector as never})).toThrow(/packed GraphDataView/);
  expect(() => create({additionalPredicates: [{kind: 'selection', mask: vector}]})).toThrow(
    /packed GraphDataView/
  );

  expect(() => create({sourceIds: view('s', 7)})).toThrow(/sourceIds length/);
  expect(() => create({outputMask: view('m', 9)})).toThrow(/outputMask length/);
  expect(() =>
    create({
      additionalPredicates: [{kind: 'time-range', mask: view('p', 3)}]
    })
  ).toThrow(/additionalPredicates\[0\] length/);
  expect(() =>
    create({
      tileVisibility: {rowTileSlots: view('rs', 5), tileMask: view('tm', 4)}
    })
  ).toThrow(/rowTileSlots length/);
  // tileMask length is the tile count and may differ from the row count.
  expect(() =>
    create({
      tileVisibility: {rowTileSlots: view('rs2'), tileMask: view('tm2', 3)}
    })
  ).not.toThrow();

  const shared = graph.createTransientBuffer({
    id: 'shared',
    byteLength: 64,
    usage: 128
  });
  expect(() =>
    create({
      liveMask: graph.createDataView(shared, {format: 'uint32', length: 8}),
      outputMask: graph.createDataView(shared, {format: 'uint32', length: 8})
    })
  ).toThrow(/share buffers/);
  expect(() =>
    create({
      sourceIds: graph.createDataView(shared, {format: 'uint32', length: 8}),
      output: {
        ...base.output,
        ids: graph.createDataView(shared, {format: 'uint32', length: 8})
      }
    })
  ).toThrow(/share buffers/);

  expect(() => create({drawInstanceCount: view('empty-draw', 0)})).toThrow(
    /drawInstanceCount must contain one/
  );
  expect(() => create({output: {...base.output, count: view('empty-count', 0)}})).toThrow(
    /output.count/
  );

  const otherGraph = new GPUCommandGraph(device);
  const foreign = new GPUResidentRowSelection(createProps(otherGraph, 8));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/belong to the target graph/);
  device.destroy();
});
