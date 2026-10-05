// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUPickRegionMask,
  GPURegionMask,
  GPURegionStatistics,
  type GPURegionStatisticsProps
} from '../../../src/map-graphs/region-statistics';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

function createProps(
  graph: GPUCommandGraph,
  prefix: string = '',
  overrides: Partial<GPURegionStatisticsProps> = {}
): GPURegionStatisticsProps {
  return {
    selection: {
      kind: 'rectangle',
      bounds: createTransientView(graph, `${prefix}bounds`, 'float32', 4)
    },
    positions: createTransientView(graph, `${prefix}positions`, 'float32x2', 12),
    values: createTransientView(graph, `${prefix}values`, 'float32', 12),
    histogram: {binCount: 4},
    output: {
      ids: createTransientView(graph, `${prefix}ids`, 'uint32', 12),
      count: createTransientView(graph, `${prefix}count`, 'uint32', 1),
      overflow: createTransientView(graph, `${prefix}overflow`, 'uint32', 1)
    },
    summary: createTransientView(graph, `${prefix}summary`, 'uint32', 12),
    ...overrides
  };
}

function findIndex(ids: string[], prefix: string): number {
  return ids.findIndex(id => id.startsWith(prefix));
}

it('GPURegionStatistics orders selection, statistics, and summary nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPURegionStatistics({...createProps(graph), id: 'stats'})
    .getCommandNodes(graph)
    .map(node => node.id);
  const order = [
    'stats-filter',
    'stats-counts-clear',
    'stats-count',
    'stats-sum',
    'stats-extent',
    'stats-histogram',
    'stats-bins-total',
    'stats-visibility',
    'stats-selection-finalize'
  ].map(prefix => findIndex(ids, prefix));
  expect(order.every(index => index >= 0)).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  expect(ids.at(-1)).toBe('stats-summary');
  expect(new Set(ids).size).toBe(ids.length);

  const second = new GPURegionStatistics({...createProps(graph, 'b-'), id: 'stats-b'});
  const secondIds = second.getCommandNodes(graph).map(node => node.id);
  expect(secondIds.some(id => ids.includes(id))).toBe(false);
  device.destroy();
});

it('GPURegionStatistics selects the region kernel per selection kind', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const polygonIds = new GPURegionStatistics({
    ...createProps(graph),
    id: 'stats',
    selection: {
      kind: 'polygon',
      vertices: createTransientView(graph, 'vertices', 'float32x2', 5),
      vertexCount: createTransientView(graph, 'vertex-count', 'uint32', 1)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(polygonIds).toContain('stats-region-overflow');
  expect(polygonIds).toContain('stats-region');

  const screenGraph = new GPUCommandGraph(device);
  const screenIds = new GPURegionStatistics({
    ...createProps(screenGraph),
    id: 'stats',
    selection: {
      kind: 'rectangle',
      bounds: createTransientView(screenGraph, 'screen-bounds', 'float32', 4),
      screenTransform: createTransientView(screenGraph, 'transform', 'float32', 20)
    }
  })
    .getCommandNodes(screenGraph)
    .map(node => node.id);
  expect(screenIds).toContain('stats-region');
  expect(findIndex(screenIds, 'stats-filter')).toBe(-1);

  const maskGraph = new GPUCommandGraph(device);
  const maskIds = new GPURegionStatistics({
    id: 'stats',
    selection: {kind: 'mask', mask: createTransientView(maskGraph, 'mask', 'uint32', 12)},
    summary: createTransientView(maskGraph, 'summary', 'uint32', 8)
  })
    .getCommandNodes(maskGraph)
    .map(node => node.id);
  expect(maskIds).toContain('stats-region-overflow-clear');
  expect(findIndex(maskIds, 'stats-filter')).toBe(-1);
  expect(maskIds.some(id => /^stats-sum(-|$)/.test(id))).toBe(false);
  expect(findIndex(maskIds, 'stats-visibility')).toBe(-1);

  const regionGraph = new GPUCommandGraph(device);
  expect(
    new GPURegionMask({
      id: 'lasso',
      positions: createTransientView(regionGraph, 'p', 'float32x2', 4),
      region: {kind: 'rectangle', bounds: createTransientView(regionGraph, 'b', 'float32', 4)},
      outputMask: createTransientView(regionGraph, 'm', 'uint32', 4),
      overflow: createTransientView(regionGraph, 'o', 'uint32', 1)
    })
      .getCommandNodes(regionGraph)
      .map(node => node.id)
  ).toEqual(['lasso-overflow', 'lasso']);
  const pickProps = {
    id: 'pick',
    result: createTransientView(regionGraph, 'r', 'uint32', 12),
    outputMask: createTransientView(regionGraph, 'pm', 'uint32', 4),
    overflow: createTransientView(regionGraph, 'po', 'uint32', 1)
  };
  expect(
    new GPUPickRegionMask(pickProps).getCommandNodes(regionGraph).map(node => node.id)
  ).toEqual(['pick-clear', 'pick-scatter', 'pick-overflow']);
  expect(() => new GPUPickRegionMask({...pickProps, batchIndex: -1})).toThrow(/batchIndex/);
  device.destroy();
});

it('GPURegionStatistics validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPURegionStatisticsProps>) =>
    new GPURegionStatistics({...base, ...overrides});

  expect(() =>
    create({
      positions: undefined,
      values: undefined,
      output: undefined,
      histogram: undefined,
      summary: createTransientView(graph, 's8', 'uint32', 8),
      outputMask: createTransientView(graph, 'om', 'uint32', 12)
    })
  ).toThrow(/positions are required/);
  expect(() => create({values: createTransientView(graph, 'v11', 'float32', 11)})).toThrow(
    /values length/
  );
  expect(() => create({values: undefined})).toThrow(/histogram requires values/);
  expect(() => create({histogram: {binCount: 0}})).toThrow(/binCount/);
  expect(() => create({summary: createTransientView(graph, 's9', 'uint32', 9)})).toThrow(/summary/);
  expect(() =>
    create({
      selection: {kind: 'mask', mask: createTransientView(graph, 'sel', 'uint32', 12)},
      outputMask: createTransientView(graph, 'out-mask', 'uint32', 12)
    })
  ).toThrow(/outputMask cannot/);
  expect(() =>
    create({
      output: {...base.output!, ids: createTransientView(graph, 'ids0', 'uint32', 0)}
    })
  ).toThrow(/output.ids/);
  expect(() =>
    create({
      selection: {
        kind: 'pick-region',
        result: createTransientView(graph, 'pick-result', 'uint32', 12)
      },
      positions: undefined,
      values: undefined,
      histogram: undefined,
      sourceIds: undefined,
      outputMask: undefined
    })
  ).toThrow(/row count/);

  const regionBase = {
    positions: createTransientView(graph, 'rp', 'float32x2', 4),
    outputMask: createTransientView(graph, 'rm', 'uint32', 4),
    overflow: createTransientView(graph, 'ro', 'uint32', 1)
  };
  expect(
    () =>
      new GPURegionMask({
        ...regionBase,
        region: {
          kind: 'rectangle',
          bounds: createTransientView(graph, 'rb', 'float32', 4),
          screenTransform: createTransientView(graph, 'rt', 'float32', 16)
        }
      })
  ).toThrow(/screenTransform/);
  expect(
    () =>
      new GPURegionMask({
        ...regionBase,
        region: {
          kind: 'polygon',
          vertices: createTransientView(graph, 'rv', 'float32x2', 2),
          vertexCount: createTransientView(graph, 'rc', 'uint32', 1)
        }
      })
  ).toThrow(/vertices/);

  const drawInstanceCount = createTransientView(graph, 'draw-count', 'uint32', 1);
  expect(() => create({drawInstanceCount})).not.toThrow();
  expect(() => create({drawInstanceCount, output: undefined})).toThrow(
    /drawInstanceCount requires output/
  );
  expect(() =>
    create({drawInstanceCount: createTransientView(graph, 'draw-count0', 'uint32', 0)})
  ).toThrow(/drawInstanceCount must contain one uint32 row/);
  expect(() =>
    create({drawInstanceCount: createTransientView(graph, 'draw-count-f', 'float32', 1) as never})
  ).toThrow(/drawInstanceCount/);
  const strided = graph.createTransientBuffer({id: 'strided', byteLength: 64, usage: 128});
  expect(() =>
    create({
      drawInstanceCount: graph.createDataView(strided, {
        format: 'uint32',
        length: 1,
        byteStride: 16,
        byteOffset: 4,
        rowByteLength: 4
      })
    })
  ).toThrow(/drawInstanceCount/);
  const inputAlias = graph.createDataView(strided, {format: 'uint32', length: 1});
  expect(() =>
    create({
      values: graph.createDataView(strided, {format: 'float32', length: 12}),
      histogram: undefined,
      summary: createTransientView(graph, 'summary-8', 'uint32', 8),
      drawInstanceCount: inputAlias
    })
  ).toThrow(/share buffers/);
  const foreignCount = createTransientView(
    new GPUCommandGraph(device),
    'foreign-count',
    'uint32',
    1
  );
  expect(() => create({drawInstanceCount: foreignCount}).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );

  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPURegionStatistics(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});
