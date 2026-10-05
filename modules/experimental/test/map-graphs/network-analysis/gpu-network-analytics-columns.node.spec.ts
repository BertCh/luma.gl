// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/map-graphs';
import {
  GPUNetworkAnalyticsColumns,
  type GPUNetworkAnalyticsColumnsProps
} from '../../../src/map-graphs/network-analysis/gpu-network-analytics-columns';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

const NODE_COUNT = 6;
const EDGE_COUNT = 10;

type Imports = {
  graph: GPUCommandGraph;
  buffers: Buffer[];
  importView: <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => GraphDataView<Format>;
};

function createImports(device: Device): Imports {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  return {
    graph,
    buffers,
    importView: (name, format, length) => {
      const uniqueName = `${name}-${buffers.length}`;
      const buffer = device.createBuffer({
        id: uniqueName,
        byteLength: Math.max(length, 1) * 4,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return importGraphBuffer(graph, uniqueName, buffer, format, length);
    }
  };
}

function createProps(
  imports: Imports,
  overrides: Partial<GPUNetworkAnalyticsColumnsProps> = {}
): GPUNetworkAnalyticsColumnsProps {
  const {importView} = imports;
  return {
    offsets: importView('offsets', 'uint32', NODE_COUNT + 1),
    neighbors: importView('neighbors', 'uint32', EDGE_COUNT),
    degree: {output: importView('degree', 'uint32', NODE_COUNT)},
    ...overrides
  };
}

it('GPUNetworkAnalyticsColumns schedules algorithms then normalization in a fixed order', () => {
  const device = createNullWebGPUDevice();
  const imports = createImports(device);
  const {importView, graph} = imports;
  const recipe = new GPUNetworkAnalyticsColumns({
    id: 'columns',
    ...createProps(imports),
    reverseOffsets: importView('reverse-offsets', 'uint32', NODE_COUNT + 1),
    reverseNeighbors: importView('reverse-neighbors', 'uint32', EDGE_COUNT),
    degree: {
      output: importView('degree', 'uint32', NODE_COUNT),
      normalized: importView('degree-normalized', 'float32', NODE_COUNT),
      extent: importView('degree-extent', 'uint32', 2)
    },
    inDegree: {output: importView('in-degree', 'uint32', NODE_COUNT)},
    pageRank: {
      output: importView('page-rank', 'float32', NODE_COUNT),
      normalized: importView('page-rank-normalized', 'float32', NODE_COUNT),
      iterations: 2,
      residual: importView('residual', 'float32', 1)
    },
    coreNumber: {
      output: importView('core', 'uint32', NODE_COUNT),
      iterations: 1,
      converged: importView('core-converged', 'uint32', 1),
      degeneracy: importView('core-degeneracy', 'uint32', 1)
    },
    components: {output: importView('components', 'uint32', NODE_COUNT), iterations: 1},
    communities: {output: importView('communities', 'uint32', NODE_COUNT), iterations: 1}
  });
  expect(recipe.recipe).toBe('network-analytics-columns');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  const indexOf = (prefix: string) => ids.findIndex(id => id.startsWith(prefix));
  expect(ids[0]).toBe('columns-degree');
  expect(ids[1]).toBe('columns-in-degree');
  expect(indexOf('columns-page-rank')).toBe(2);
  expect(indexOf('columns-core-number')).toBeGreaterThan(indexOf('columns-page-rank'));
  expect(indexOf('columns-components')).toBeGreaterThan(indexOf('columns-core-number'));
  expect(indexOf('columns-communities')).toBeGreaterThan(indexOf('columns-components'));
  expect(indexOf('columns-degree-extent')).toBeGreaterThan(indexOf('columns-communities'));
  expect(ids).toContain('columns-degree-normalize');
  expect(ids).toContain('columns-page-rank-normalize');
  expect(ids.some(id => id.startsWith('columns-page-rank-extent'))).toBe(true);
  expect(ids).not.toContain('columns-in-degree-normalize');
  expect(ids.indexOf('columns-degree-normalize')).toBeGreaterThan(indexOf('columns-degree-extent'));
  recipe.destroy();
  device.destroy();
});

it('GPUNetworkAnalyticsColumns schedules only the requested metrics', () => {
  const device = createNullWebGPUDevice();
  const imports = createImports(device);
  const recipe = new GPUNetworkAnalyticsColumns(createProps(imports));
  expect(recipe.id).toBe('network-analytics-columns');
  expect(recipe.getCommandNodes(imports.graph).map(node => node.id)).toEqual([
    'network-analytics-columns-degree'
  ]);
  recipe.destroy();
  device.destroy();
});

it('GPUNetworkAnalyticsColumns validates props', () => {
  const device = createNullWebGPUDevice();
  const imports = createImports(device);
  const {importView, graph} = imports;
  const base = createProps(imports);
  const create = (overrides: Partial<GPUNetworkAnalyticsColumnsProps>) =>
    new GPUNetworkAnalyticsColumns({...base, ...overrides});

  expect(() => create({degree: undefined})).toThrow(/at least one metric/);
  expect(() => create({inDegree: {output: importView('in', 'uint32', NODE_COUNT)}})).toThrow(
    /inDegree requires reverseOffsets/
  );
  expect(() =>
    create({reverseOffsets: importView('reverse-offsets', 'uint32', NODE_COUNT + 1)})
  ).toThrow(/given together/);
  expect(() =>
    create({
      reverseOffsets: importView('reverse-offsets-short', 'uint32', NODE_COUNT),
      reverseNeighbors: importView('reverse-neighbors', 'uint32', EDGE_COUNT)
    })
  ).toThrow(/reverseOffsets must contain the same number of rows/);
  expect(() => create({offsets: importView('offsets-one', 'uint32', 1)})).toThrow(
    /at least two rows/
  );
  expect(() => create({degree: {output: importView('degree-short', 'uint32', 3)}})).toThrow(
    /degree\.output length must equal the node count/
  );
  expect(() =>
    create({degree: {output: importView('degree-float', 'float32', NODE_COUNT) as never}})
  ).toThrow(/degree\.output/);
  expect(() =>
    create({
      degree: {
        output: importView('degree-2', 'uint32', NODE_COUNT),
        normalized: importView('degree-normalized-short', 'float32', 3)
      }
    })
  ).toThrow(/degree\.normalized must contain one row per node/);
  expect(() =>
    create({
      degree: {
        output: importView('degree-3', 'uint32', NODE_COUNT),
        normalized: importView('degree-normalized', 'float32', NODE_COUNT),
        extent: importView('degree-extent-long', 'uint32', 3)
      }
    })
  ).toThrow(/degree\.extent must contain exactly two rows/);
  expect(() =>
    create({
      degree: {
        output: importView('degree-4', 'uint32', NODE_COUNT),
        extent: importView('degree-extent', 'uint32', 2)
      }
    })
  ).toThrow(/extent requires degree\.normalized/);
  expect(() =>
    create({
      coreNumber: {
        output: importView('core', 'uint32', NODE_COUNT),
        converged: importView('core-converged', 'uint32', 2)
      }
    })
  ).toThrow(/coreNumber\.converged must contain exactly one row/);

  // Transient views are accepted: the algorithms bind graph views directly.
  const transient = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  expect(() =>
    create({
      offsets: transient('t-offsets', NODE_COUNT + 1),
      neighbors: transient('t-neighbors', EDGE_COUNT),
      degree: {output: transient('t-degree', NODE_COUNT)},
      components: {
        output: transient('t-components', NODE_COUNT),
        converged: transient('t-converged', 1)
      }
    }).getCommandNodes(graph)
  ).not.toThrow();

  // Output aliasing.
  expect(() => create({components: {output: base.degree!.output}})).toThrow(/separate buffers/);
  expect(() => create({components: {output: base.offsets as never}})).toThrow(
    /length must equal|separate buffers/
  );

  // gpu-graph validation surfaces from the constructor.
  expect(() =>
    create({pageRank: {output: importView('pr', 'float32', NODE_COUNT), damping: 2}})
  ).toThrow(/damping/);
  expect(() =>
    create({
      components: {output: importView('cc', 'uint32', NODE_COUNT), iterations: 0}
    })
  ).toThrow(/iterations/);

  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUNetworkAnalyticsColumns(createProps(createImports(device))).getCommandNodes(otherGraph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUNetworkAnalyticsColumns owns no GPU resources and validates algorithms eagerly', () => {
  const device = createNullWebGPUDevice();
  const imports = createImports(device);
  const recipe = new GPUNetworkAnalyticsColumns(
    createProps(imports, {pageRank: {output: imports.importView('pr', 'float32', NODE_COUNT)}})
  );
  const ids = recipe.getCommandNodes(imports.graph).map(node => node.id);
  // PageRank reads an overflow word; the exact CSR gets a graph-owned zero word cleared first.
  expect(ids.indexOf('network-analytics-columns-page-rank-forward-overflow-exact-clear')).toBe(1);
  expect(() => recipe.destroy()).not.toThrow();

  const failing = createImports(device);
  expect(
    () =>
      new GPUNetworkAnalyticsColumns(
        createProps(failing, {
          pageRank: {output: failing.importView('pr', 'float32', NODE_COUNT), iterations: 0}
        })
      )
  ).toThrow(/iterations/);
  device.destroy();
});
