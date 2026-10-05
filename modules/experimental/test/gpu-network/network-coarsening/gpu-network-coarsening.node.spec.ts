// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPUNetworkCoarseningSummary,
  GPUNetworkCoarsening,
  type GPUNetworkCoarseningProps
} from '../../../src/gpu-network/network-coarsening';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  buildCoarseningCSR,
  computeCoarseningOracle,
  dequantize,
  quantize
} from './network-coarsening-oracle';

const NODE_COUNT = 6;
const SLOT_COUNT = 10;
const GROUP_CAPACITY = 4;
const EDGE_CAPACITY = 5;

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  let counter = 0;
  const importView = <Format extends GPUVectorFormat>(
    name: string,
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    const rowWords = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
    name = `${name}-${counter++}`;
    const buffer = device.createBuffer({
      id: name,
      byteLength: Math.max(length * rowWords, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const props = (
    overrides: Partial<GPUNetworkCoarseningProps> = {}
  ): GPUNetworkCoarseningProps => ({
    offsets: importView('offsets', 'uint32', NODE_COUNT + 1),
    neighbors: importView('neighbors', 'uint32', SLOT_COUNT),
    labels: importView('labels', 'uint32', NODE_COUNT),
    groupCapacity: GROUP_CAPACITY,
    groupVertexCount: importView('group-counts', 'uint32', GROUP_CAPACITY),
    edges: {
      ids: importView('edge-ids', 'uint32', EDGE_CAPACITY),
      count: importView('edge-count', 'uint32', 1),
      overflow: importView('edge-overflow', 'uint32', 1)
    },
    edgeTargets: importView('edge-targets', 'uint32', EDGE_CAPACITY),
    edgeCounts: importView('edge-counts', 'uint32', EDGE_CAPACITY),
    ...overrides
  });
  return {graph, importView, props};
}

it('GPUNetworkCoarsening schedules a fixed node order and omits unused passes', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const minimal = new GPUNetworkCoarsening(props({id: 'coarse'}));
  const minimalIds = minimal.getCommandNodes(graph).map(node => node.id);
  expect(minimal.recipe).toBe('network-coarsening');
  expect(minimalIds[0]).toBe('coarse-zero-stats');
  expect(minimalIds.some(id => id === 'coarse-groups')).toBe(false);
  expect(minimalIds.some(id => id.endsWith('-edge-weights'))).toBe(false);
  expect(minimalIds[minimalIds.length - 1]).toBe('coarse-publish');
  const indexOf = (ids: string[], name: string) => ids.indexOf(`coarse-${name}`);
  expect(indexOf(minimalIds, 'vertices')).toBeGreaterThan(indexOf(minimalIds, 'init-groups'));
  expect(indexOf(minimalIds, 'slots')).toBeGreaterThan(indexOf(minimalIds, 'vertices'));
  expect(indexOf(minimalIds, 'segments')).toBeGreaterThan(indexOf(minimalIds, 'segment-total'));

  const other = createContext(device);
  const full = new GPUNetworkCoarsening(
    other.props({
      id: 'full',
      directed: true,
      vertexMask: other.importView('vertex-mask', 'uint32', NODE_COUNT),
      edgeMask: other.importView('edge-mask', 'uint32', SLOT_COUNT),
      weights: other.importView('weights', 'float32', SLOT_COUNT),
      positions: other.importView('positions', 'float32x2', NODE_COUNT),
      vertexValues: other.importView('values', 'float32', NODE_COUNT),
      groupIntraEdgeCount: other.importView('intra-count', 'uint32', GROUP_CAPACITY),
      groupIntraWeight: other.importView('intra-weight', 'float32', GROUP_CAPACITY),
      groupCentroid: other.importView('centroid', 'float32x2', GROUP_CAPACITY),
      groupBounds: other.importView('bounds', 'float32x4', GROUP_CAPACITY),
      groupValueSum: other.importView('value-sum', 'float32', GROUP_CAPACITY),
      edgeWeights: other.importView('edge-weights', 'float32', EDGE_CAPACITY),
      summary: other.importView('summary', 'uint32', 8)
    })
  );
  const fullIds = full.getCommandNodes(other.graph).map(node => node.id);
  expect(fullIds).toContain('full-groups');
  expect(fullIds).toContain('full-edge-weights');
  void importView;
  device.destroy();
});

it('GPUNetworkCoarsening validates props', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const create = (overrides: Partial<GPUNetworkCoarseningProps>) =>
    new GPUNetworkCoarsening(props(overrides));
  expect(() => create({offsets: importView('o1', 'uint32', 1)})).toThrow(/at least two rows/);
  expect(() => create({neighbors: importView('n0', 'uint32', 0)})).toThrow(/at least one slot/);
  expect(() => create({groupCapacity: 0})).toThrow(/groupCapacity/);
  expect(() => create({groupCapacity: 65536})).toThrow(/groupCapacity/);
  expect(() => create({fixedPointScale: 3})).toThrow(/fixedPointScale/);
  expect(() => create({labels: importView('l-short', 'uint32', 2)})).toThrow(/labels must contain/);
  expect(() => create({groupVertexCount: importView('g-short', 'uint32', 2)})).toThrow(
    /groupVertexCount must contain exactly 4 rows/
  );
  expect(() => create({edgeTargets: importView('t-short', 'uint32', 2)})).toThrow(
    /edgeTargets must match edges.ids length/
  );
  expect(() => create({summary: importView('s-short', 'uint32', 3)})).toThrow(
    /summary must contain/
  );
  expect(() => create({groupCentroid: importView('c', 'float32x2', GROUP_CAPACITY)})).toThrow(
    /require positions/
  );
  expect(() => create({groupValueSum: importView('v', 'float32', GROUP_CAPACITY)})).toThrow(
    /requires vertexValues/
  );
  const aliased = importView('aliased', 'uint32', NODE_COUNT);
  expect(() => create({labels: aliased, groupVertexCount: aliased})).toThrow();
  const foreign = createContext(createNullWebGPUDevice());
  expect(() => new GPUNetworkCoarsening(foreign.props()).getCommandNodes(graph)).toThrow(
    /views must belong to the target graph/
  );
  device.destroy();
});

it('coarsening oracle helpers quantize, wrap and order deterministically', () => {
  expect(quantize(1.5, 4)).toBe(6n);
  expect(quantize(-2.5, 1)).toBe(-2n);
  expect(quantize(3.5, 1)).toBe(4n);
  expect(dequantize(-6n, 4)).toBe(-1.5);
  expect(dequantize(1n << 40n, 1)).toBe(2 ** 40);
  const csr = buildCoarseningCSR(4, [
    [0, 1, 1],
    [1, 0, 1],
    [2, 3, 2],
    [3, 2, 2],
    [0, 0, 5]
  ]);
  const result = computeCoarseningOracle({
    nodeCount: 4,
    csr,
    labels: Uint32Array.from([0, 0, 1, 9]),
    groupCapacity: 3,
    edgeCapacity: 4
  });
  expect(Array.from(result.groupVertexCount)).toEqual([2, 1, 0]);
  expect(Array.from(result.groupIntraEdgeCount)).toEqual([2, 0, 0]);
  expect(result.edges).toEqual([]);
  expect(decodeGPUNetworkCoarseningSummary(result.summary)).toMatchObject({
    liveVertexCount: 4,
    overflowedVertexCount: 1,
    droppedEdgeCount: 1,
    intraEdgeCount: 2,
    groupCount: 2
  });
  expect(result.overflow).toBe(true);
  expect(() => decodeGPUNetworkCoarseningSummary([1, 2])).toThrow(/eight words/);
});
