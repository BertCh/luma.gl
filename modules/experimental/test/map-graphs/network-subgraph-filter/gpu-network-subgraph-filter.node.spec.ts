// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/map-graphs';
import {
  decodeGPUNetworkSubgraphFilterCounts,
  getGPUNetworkSubgraphFilterParameterLength,
  getGPUNetworkSubgraphFilterParameterValues,
  GPUNetworkSubgraphFilter,
  type GPUNetworkSubgraphFilterProps
} from '../../../src/map-graphs/network-subgraph-filter';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {computeSubgraphOracle} from './network-subgraph-filter-oracle';

const NODE_COUNT = 5;
const SLOT_COUNT = 8;

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const importView = <Format extends 'uint32' | 'float32' | 'uint32x2'>(
    name: string,
    length: number,
    format: Format = 'uint32' as Format
  ): GraphDataView<Format> => {
    const uniqueName = `${name}-${buffers.length}`;
    const buffer = device.createBuffer({
      id: uniqueName,
      byteLength: Math.max(length, 1) * 8,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return importGraphBuffer(graph, uniqueName, buffer, format, length);
  };
  const props = (
    overrides: Partial<GPUNetworkSubgraphFilterProps> = {}
  ): GPUNetworkSubgraphFilterProps => ({
    offsets: importView('offsets', NODE_COUNT + 1),
    neighbors: importView('neighbors', SLOT_COUNT),
    output: {
      vertexMask: importView('vertex-mask-out', NODE_COUNT),
      edgeMask: importView('edge-mask-out', SLOT_COUNT)
    },
    ...overrides
  });
  return {graph, importView, props};
}

it('GPUNetworkSubgraphFilter schedules a fixed node order', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const recipe = new GPUNetworkSubgraphFilter({
    id: 'sub',
    ...props({
      vertexColumns: [importView('vc', NODE_COUNT, 'float32')],
      edgeColumns: Array.from({length: 7}, (_, i) => importView(`ec${i}`, SLOT_COUNT, 'float32')),
      edgeTimes: importView('times', SLOT_COUNT, 'float32'),
      edgeTimeWords: importView('words', SLOT_COUNT, 'uint32x2'),
      parameters: importView('params', 9 * 4, 'float32'),
      timeWordParameters: importView('twp', 8),
      dropIsolated: true,
      output: {
        vertexMask: importView('vmo', NODE_COUNT),
        edgeMask: importView('emo', SLOT_COUNT),
        counts: importView('counts', 4),
        liveVertices: {
          ids: importView('lv', NODE_COUNT),
          count: importView('lvc', 1),
          overflow: importView('lvo', 1)
        },
        liveEdgeSlots: {
          ids: importView('ls', SLOT_COUNT),
          count: importView('lsc', 1),
          overflow: importView('lso', 1)
        },
        inducedCSR: {
          offsets: importView('io', NODE_COUNT + 1),
          neighbors: importView('in', SLOT_COUNT),
          sourceSlots: importView('is', SLOT_COUNT),
          overflow: importView('iov', 1)
        }
      }
    })
  });
  expect(recipe.recipe).toBe('network-subgraph-filter');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  const indexOf = (name: string) => ids.indexOf(`sub-${name}`);
  expect(ids[0]).toBe('sub-vertex-init');
  expect(indexOf('vertex-columns-0')).toBe(1);
  expect(indexOf('edge-init')).toBeGreaterThan(indexOf('vertex-columns-0'));
  // 7 edge columns split into chunks of 6 and 1.
  expect(indexOf('edge-columns-1')).toBe(indexOf('edge-columns-0') + 1);
  expect(indexOf('edge-times')).toBeGreaterThan(indexOf('edge-columns-1'));
  expect(indexOf('edge-time-words')).toBeGreaterThan(indexOf('edge-times'));
  expect(indexOf('edge-endpoints')).toBeGreaterThan(indexOf('edge-time-words'));
  expect(indexOf('drop-isolated')).toBeGreaterThan(indexOf('mark-incident'));
  expect(indexOf('counts')).toBeGreaterThan(indexOf('drop-isolated'));
  expect(indexOf('live-vertices-scatter')).toBeGreaterThan(indexOf('counts-finish'));
  expect(indexOf('induced-scatter')).toBeGreaterThan(indexOf('live-slots-publish'));
  device.destroy();
});

it('GPUNetworkSubgraphFilter omits unused passes and defaults the id', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const recipe = new GPUNetworkSubgraphFilter(props());
  expect(recipe.id).toBe('network-subgraph-filter');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual([
    'network-subgraph-filter-vertex-init',
    'network-subgraph-filter-edge-init',
    'network-subgraph-filter-edge-endpoints'
  ]);
  device.destroy();
});

it('GPUNetworkSubgraphFilter validates props', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const create = (overrides: Partial<GPUNetworkSubgraphFilterProps>) =>
    new GPUNetworkSubgraphFilter(props(overrides));
  expect(() => create({offsets: importView('o1', 1)})).toThrow(/at least two rows/);
  expect(() => create({neighbors: importView('n0', 0)})).toThrow(/at least one row/);
  expect(() => create({vertexMask: importView('vm', 2)})).toThrow(/vertexMask must contain/);
  expect(() => create({edgeMask: importView('em', 2)})).toThrow(/edgeMask must contain/);
  expect(() => create({vertexColumns: [importView('vc', NODE_COUNT, 'float32')]})).toThrow(
    /parameters are required/
  );
  expect(() =>
    create({
      vertexColumns: [importView('vc2', 2, 'float32')],
      parameters: importView('p', 4, 'float32')
    })
  ).toThrow(/vertexColumns\[0\] must contain/);
  expect(() =>
    create({
      vertexColumns: [importView('vc3', NODE_COUNT, 'float32')],
      parameters: importView('p2', 3, 'float32')
    })
  ).toThrow(/at least 4 float32/);
  expect(() => create({edgeTimeWords: importView('w', SLOT_COUNT, 'uint32x2')})).toThrow(
    /timeWordParameters are required/
  );
  expect(() =>
    create({
      edgeTimeWords: importView('w2', SLOT_COUNT, 'uint32x2'),
      timeWordParameters: importView('twp', 4)
    })
  ).toThrow(/at least 8/);
  expect(() =>
    create({
      output: {
        vertexMask: importView('a', NODE_COUNT),
        edgeMask: importView('b', SLOT_COUNT),
        counts: importView('c', 3)
      }
    })
  ).toThrow(/counts must contain at least 4/);
  expect(() =>
    create({
      output: {
        vertexMask: importView('a2', NODE_COUNT),
        edgeMask: importView('b2', SLOT_COUNT),
        inducedCSR: {
          offsets: importView('io', NODE_COUNT),
          neighbors: importView('in', SLOT_COUNT),
          overflow: importView('iv', 1)
        }
      }
    })
  ).toThrow(/offsets must contain nodeCount \+ 1/);
  expect(() =>
    create({
      output: {
        vertexMask: importView('a3', NODE_COUNT),
        edgeMask: importView('b3', SLOT_COUNT),
        inducedCSR: {
          offsets: importView('io3', NODE_COUNT + 1),
          neighbors: importView('in3', SLOT_COUNT),
          sourceSlots: importView('is3', 2),
          overflow: importView('iv3', 1)
        }
      }
    })
  ).toThrow(/sourceSlots must match/);
  const aliased = importView('aliased', SLOT_COUNT);
  expect(() =>
    create({
      neighbors: aliased,
      output: {vertexMask: importView('a4', NODE_COUNT), edgeMask: aliased}
    })
  ).toThrow(/outputs must not share buffers with inputs/);

  const otherDevice = createNullWebGPUDevice();
  const other = createContext(otherDevice);
  const foreign = new GPUNetworkSubgraphFilter(other.props());
  expect(() => foreign.getCommandNodes(graph)).toThrow(/views must belong to the target graph/);
  otherDevice.destroy();
  device.destroy();
});

it('parameter helpers pack records and decode counts', () => {
  const layout = {
    vertexColumnCount: 2,
    edgeColumnCount: 1,
    hasEdgeTimes: true
  };
  expect(getGPUNetworkSubgraphFilterParameterLength(layout)).toBe(16);
  const values = getGPUNetworkSubgraphFilterParameterValues(layout, {
    vertexRanges: [[1, 2], null],
    edgeRanges: [[3, Infinity]],
    edgeTimeWindow: [10, 20]
  });
  expect(Array.from(values)).toEqual([1, 2, 1, 0, 0, 0, 0, 0, 3, Infinity, 1, 0, 10, 20, 1, 0]);
  expect(() => getGPUNetworkSubgraphFilterParameterValues(layout, {}, new Float32Array(4))).toThrow(
    /too short/
  );
  expect(decodeGPUNetworkSubgraphFilterCounts([4, 3, 1, 2])).toEqual({
    liveVertexCount: 4,
    liveSlotCount: 3,
    selfLoopSlotCount: 1,
    liveEdgeCount: 2
  });
});

it('oracle applies predicates, endpoint rule, dropIsolated and induced CSR on a tiny graph', () => {
  // 0->1, 0->2, 1->2, 2->3, 3->3 ; vertex 4 has no edges.
  const offsets = Uint32Array.from([0, 2, 3, 4, 5, 5]);
  const neighbors = Uint32Array.from([1, 2, 2, 3, 3]);
  const result = computeSubgraphOracle({
    nodeCount: 5,
    offsets,
    neighbors,
    directed: true,
    vertexColumns: [Float32Array.from([0, 1, 2, 3, 4])],
    vertexRanges: [[0, 3]]
  });
  expect(Array.from(result.vertexMask)).toEqual([1, 1, 1, 0, 0]);
  expect(Array.from(result.edgeMask)).toEqual([1, 1, 1, 0, 0]);
  expect(result.counts).toEqual([3, 3, 0, 3]);
  const dropped = computeSubgraphOracle({
    nodeCount: 5,
    offsets,
    neighbors,
    vertexColumns: [Float32Array.from([0, 1, 2, 3, 4])],
    vertexRanges: [[0, 3]],
    dropIsolated: true
  });
  expect(Array.from(dropped.vertexMask)).toEqual([1, 1, 1, 0, 0]);
  expect(Array.from(dropped.inducedOffsets)).toEqual([0, 2, 3, 3, 3, 3]);
  expect(dropped.inducedNeighbors).toEqual([1, 2, 2]);
  expect(dropped.inducedSlots).toEqual([0, 1, 2]);
  // Half-open: max is excluded; NaN fails.
  const edge = computeSubgraphOracle({
    nodeCount: 5,
    offsets,
    neighbors,
    edgeColumns: [Float32Array.from([1, 2, NaN, 3, 2])],
    edgeRanges: [[2, 3]]
  });
  expect(Array.from(edge.edgeMask)).toEqual([0, 1, 0, 0, 1]);
});

it('oracle pairs undirected slots, parallel edges and self-loops', () => {
  // 0 <-> 1 twice (parallel), 1 -> 2 without reverse, 2 -> 2 loop.
  const offsets = Uint32Array.from([0, 2, 5, 6]);
  const neighbors = Uint32Array.from([1, 1, 0, 0, 2, 2]);
  //                       slots:      0  1  2  3  4  5
  const edgeColumns = [Float32Array.from([1, 1, 1, 0, 0, 0])];
  const base = {
    nodeCount: 3,
    offsets,
    neighbors,
    edgeColumns,
    edgeRanges: [[1, 2]] as const
  };
  // Slot 0 pairs with 2 (pass, pass); slot 1 pairs with 3 (pass, fail) so both die; slot 4
  // (1 -> 2) has no reverse and fails by itself; the loop fails by itself.
  expect(Array.from(computeSubgraphOracle(base).edgeMask)).toEqual([1, 0, 1, 0, 0, 0]);
  expect(Array.from(computeSubgraphOracle({...base, pairUndirectedSlots: false}).edgeMask)).toEqual(
    [1, 1, 1, 0, 0, 0]
  );
  expect(Array.from(computeSubgraphOracle({...base, directed: true}).edgeMask)).toEqual([
    1, 1, 1, 0, 0, 0
  ]);
  // A missing reverse keeps the slot's own result.
  const lenient = computeSubgraphOracle({
    ...base,
    edgeColumns: [Float32Array.from([1, 1, 1, 1, 1, 1])]
  });
  expect(Array.from(lenient.edgeMask)).toEqual([1, 1, 1, 1, 1, 1]);
});
