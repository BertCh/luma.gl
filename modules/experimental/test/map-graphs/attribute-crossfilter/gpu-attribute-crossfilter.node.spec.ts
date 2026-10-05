// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUAttributeCrossfilterHistogramLayout,
  getGPUAttributeCrossfilterParameterValues,
  GPUAttributeCrossfilter,
  GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE,
  type GPUAttributeCrossfilterProps
} from '../../../src/map-graphs/attribute-crossfilter';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

function createProps(
  graph: GPUCommandGraph,
  rows: number = 8,
  overrides: Partial<GPUAttributeCrossfilterProps> = {}
): GPUAttributeCrossfilterProps {
  return {
    dimensions: [
      {column: createTransientView(graph, 'a', 'float32', rows), binCount: 4},
      {
        column: createTransientView(graph, 'b', 'uint32', rows),
        binCount: 8,
        domain: 'parameters'
      }
    ],
    parameters: createTransientView(graph, 'parameters', 'float32', 16),
    histograms: createTransientView(graph, 'histograms', 'uint32', 12),
    ...overrides
  };
}

it('GPUAttributeCrossfilter schedules the expected nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUAttributeCrossfilter(createProps(graph)).getCommandNodes(graph).map(n => n.id);
  expect(ids).toEqual([
    'attribute-crossfilter-init',
    'attribute-crossfilter-domain-0',
    'attribute-crossfilter-resolve-domains',
    'attribute-crossfilter-fail-0',
    'attribute-crossfilter-histogram-0',
    'attribute-crossfilter-histogram-1'
  ]);

  const outputGraph = new GPUCommandGraph(device);
  const outputIds = new GPUAttributeCrossfilter(
    createProps(outputGraph, 8, {
      id: 'x',
      liveMask: createTransientView(outputGraph, 'live', 'uint32', 8),
      selectedCount: createTransientView(outputGraph, 'selected', 'uint32', 1),
      output: {
        ids: createTransientView(outputGraph, 'ids', 'uint32', 4),
        count: createTransientView(outputGraph, 'count', 'uint32', 1),
        overflow: createTransientView(outputGraph, 'overflow', 'uint32', 1)
      }
    })
  )
    .getCommandNodes(outputGraph)
    .map(n => n.id);
  expect(outputIds).toContain('x-selection');
  expect(outputIds.at(-1)).toBe('x-publish');
  expect(new Set(outputIds).size).toBe(outputIds.length);

  // 7 dimensions need two fail-bit passes.
  const wideGraph = new GPUCommandGraph(device);
  const wideIds = new GPUAttributeCrossfilter(
    createProps(wideGraph, 4, {
      dimensions: Array.from({length: 7}, (_, index) => ({
        column: createTransientView(wideGraph, `c${index}`, 'float32', 4),
        binCount: 2
      })),
      parameters: createTransientView(wideGraph, 'p', 'float32', 56),
      histograms: createTransientView(wideGraph, 'h', 'uint32', 14)
    })
  )
    .getCommandNodes(wideGraph)
    .map(n => n.id);
  expect(wideIds.filter(id => id.includes('-fail-'))).toHaveLength(2);
  device.destroy();
});

it('GPUAttributeCrossfilter validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createProps(graph);
  const create = (overrides: Partial<GPUAttributeCrossfilterProps>) =>
    new GPUAttributeCrossfilter({...base, ...overrides});
  const column = (id: string, rows = 8) => createTransientView(graph, id, 'float32', rows);

  expect(() => create({dimensions: []})).toThrow(/dimensions/);
  expect(() =>
    create({
      dimensions: Array.from({length: 33}, (_, i) => ({
        column: column(`d${i}`),
        binCount: 2
      }))
    })
  ).toThrow(/dimensions/);
  expect(() => create({dimensions: [{column: column('bin-zero'), binCount: 0}]})).toThrow(
    /binCount/
  );
  expect(() => create({dimensions: [{column: column('d'), binCount: 4097}]})).toThrow(/binCount/);
  expect(() =>
    create({
      dimensions: [
        {column: column('a1'), binCount: 2},
        {column: column('a2', 5), binCount: 2}
      ]
    })
  ).toThrow(/equal row counts/);
  expect(() =>
    create({
      dimensions: [
        {
          column: createTransientView(graph, 'i', 'sint32', 8) as never,
          binCount: 2
        }
      ]
    })
  ).toThrow(/packed/);
  expect(() => create({parameters: createTransientView(graph, 'p8', 'float32', 8)})).toThrow(
    /parameters/
  );
  expect(() => create({histograms: createTransientView(graph, 'h8', 'uint32', 8)})).toThrow(
    /histograms/
  );
  expect(() => create({liveMask: createTransientView(graph, 'live5', 'uint32', 5)})).toThrow(
    /liveMask/
  );
  expect(() => create({domains: createTransientView(graph, 'dom3', 'float32', 3)})).toThrow(
    /domains/
  );
  expect(() => create({selection: createTransientView(graph, 'sel5', 'uint32', 5)})).toThrow(
    /selection/
  );

  // Outputs must not alias inputs or each other.
  expect(() =>
    create({
      dimensions: [{column: base.dimensions[1].column, binCount: 8}],
      parameters: createTransientView(graph, 'p-alias', 'float32', 8),
      histograms: base.dimensions[1].column as never
    })
  ).toThrow(/must not share buffers with inputs/);
  const shared = createTransientView(graph, 'shared', 'uint32', 16);
  expect(() => create({histograms: shared, selectedCount: shared})).toThrow(/each other/);

  // Views from another graph are rejected when wiring.
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUAttributeCrossfilter(createProps(otherGraph)).getCommandNodes(graph)).toThrow(
    /belong to the target graph/
  );
  device.destroy();
});

it('packs parameters and histogram offsets', () => {
  const values = getGPUAttributeCrossfilterParameterValues([{brush: [1, 2], domain: [0, 10]}, {}]);
  expect(values).toHaveLength(2 * GPU_ATTRIBUTE_CROSSFILTER_PARAMETER_STRIDE);
  expect(Array.from(values.slice(0, 8))).toEqual([1, 2, 1, 0, 10, 0, 0, 0]);
  expect(Array.from(values.slice(8))).toEqual(new Array(8).fill(0));
  expect(() => getGPUAttributeCrossfilterParameterValues([{}], new Float32Array(4))).toThrow(
    /too short/
  );
  expect(getGPUAttributeCrossfilterHistogramLayout([4, 8, 2])).toEqual({
    offsets: [0, 4, 12],
    totalBinCount: 14
  });
});
