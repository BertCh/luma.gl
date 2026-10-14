// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getResourceBuffer} from '../network-resource-buffers';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPUNetworkCoarseningSummary,
  GPUNetworkCoarsening,
  type GPUNetworkCoarseningOutput,
  type GPUNetworkCoarseningProps
} from '../../../src/gpu-network/network-coarsening';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCoarseningCSR,
  computeCoarseningOracle,
  dequantize,
  quantize
} from './network-coarsening-oracle';

type PropsOverrides = Omit<Partial<GPUNetworkCoarseningProps>, 'output'> & {
  output?: Partial<GPUNetworkCoarseningOutput>;
};

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
  const props = (overrides: PropsOverrides = {}): GPUNetworkCoarseningProps => ({
    topology: {
      offsets: importView('offsets', 'uint32', NODE_COUNT + 1),
      neighbors: importView('neighbors', 'uint32', SLOT_COUNT)
    },
    labels: importView('labels', 'uint32', NODE_COUNT),
    groupCapacity: GROUP_CAPACITY,
    ...overrides,
    output: {
      groupNodeCount: importView('group-counts', 'uint32', GROUP_CAPACITY),
      edges: {
        ids: importView('edge-ids', 'uint32', EDGE_CAPACITY),
        status: {
          count: importView('edge-count', 'uint32', 1),
          overflow: importView('edge-overflow', 'uint32', 1)
        }
      },
      edgeTargetGroups: importView('edge-targets', 'uint32', EDGE_CAPACITY),
      edgeCounts: importView('edge-counts', 'uint32', EDGE_CAPACITY),
      ...overrides.output
    }
  });
  return {graph, importView, props};
}

it('GPUNetworkCoarsening schedules a fixed node order and omits unused passes', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const minimal = new GPUNetworkCoarsening(props({id: 'coarse'}));
  const minimalIds = minimal.getCommandNodes(graph).map(node => node.id);
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
      topology: {
        offsets: other.importView('offsets', 'uint32', NODE_COUNT + 1),
        neighbors: other.importView('neighbors', 'uint32', SLOT_COUNT),
        weights: other.importView('weights', 'float32', SLOT_COUNT),
        nodePositions: other.importView('positions', 'float32x2', NODE_COUNT)
      },
      nodeMask: other.importView('vertex-mask', 'uint32', NODE_COUNT),
      edgeMask: other.importView('edge-mask', 'uint32', SLOT_COUNT),
      nodeValues: other.importView('values', 'float32', NODE_COUNT),
      output: {
        groupIntraEdgeCount: other.importView('intra-count', 'uint32', GROUP_CAPACITY),
        groupIntraWeight: other.importView('intra-weight', 'float32', GROUP_CAPACITY),
        groupCentroid: other.importView('centroid', 'float32x2', GROUP_CAPACITY),
        groupBounds: other.importView('bounds', 'float32x4', GROUP_CAPACITY),
        groupValueSum: other.importView('value-sum', 'float32', GROUP_CAPACITY),
        edgeWeights: other.importView('edge-weights', 'float32', EDGE_CAPACITY),
        summary: other.importView('summary', 'uint32', 8)
      }
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
  const create = (overrides: PropsOverrides) => new GPUNetworkCoarsening(props(overrides));
  const topology = (offsets: number, neighbors: number) => ({
    offsets: importView('o', 'uint32', offsets),
    neighbors: importView('n', 'uint32', neighbors)
  });
  expect(() => create({topology: topology(1, SLOT_COUNT)})).toThrow(/at least two rows/);
  expect(() => create({topology: topology(NODE_COUNT + 1, 0)})).toThrow(/at least one slot/);
  expect(() =>
    create({
      topology: {...topology(NODE_COUNT + 1, SLOT_COUNT), weights: importView('w', 'float32', 3)}
    })
  ).toThrow(/topology\.weights length/);
  expect(() => create({groupCapacity: 0})).toThrow(/groupCapacity/);
  expect(() => create({groupCapacity: 65536})).toThrow(/groupCapacity/);
  expect(() => create({fixedPointScale: 3})).toThrow(/fixedPointScale/);
  expect(() => create({labels: importView('l-short', 'uint32', 2)})).toThrow(/labels must contain/);
  expect(() => create({output: {groupNodeCount: importView('g-short', 'uint32', 2)}})).toThrow(
    /output\.groupNodeCount must contain exactly 4 rows/
  );
  expect(() => create({output: {edgeTargetGroups: importView('t-short', 'uint32', 2)}})).toThrow(
    /output\.edgeTargetGroups must match output\.edges\.ids length/
  );
  expect(() => create({output: {summary: importView('s-short', 'uint32', 3)}})).toThrow(
    /output\.summary must contain/
  );
  expect(() =>
    create({output: {groupCentroid: importView('c', 'float32x2', GROUP_CAPACITY)}})
  ).toThrow(/require topology\.nodePositions/);
  expect(() =>
    create({output: {groupValueSum: importView('v', 'float32', GROUP_CAPACITY)}})
  ).toThrow(/requires nodeValues/);
  const aliased = importView('aliased', 'uint32', NODE_COUNT);
  expect(() => create({labels: aliased, output: {groupNodeCount: aliased}})).toThrow();
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
    liveNodeCount: 4,
    overflowedNodeCount: 1,
    droppedEdgeCount: 1,
    intraEdgeCount: 2,
    groupCount: 2
  });
  expect(result.overflow).toBe(true);
  expect(() => decodeGPUNetworkCoarseningSummary([1, 2])).toThrow(/eight words/);
});

it('GPUNetworkCoarsening gives CSR tail slots the sentinel key and folds its counters', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const nodes = new GPUNetworkCoarsening(props({id: 'tail'})).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // Review B-09: slots past offsets[NODE_COUNT] are never written by the per-row slots pass, so
  // the sort keys are cleared to the sentinel first and those slots sort last.
  const clear = ids.indexOf('tail-clear-sort-keys');
  expect(clear).toBeGreaterThanOrEqual(0);
  expect(clear).toBeLessThan(ids.indexOf('tail-slots'));
  const clearResource = nodes[clear].resources?.[0];
  const clearBuffer = clearResource ? getResourceBuffer(clearResource) : undefined;
  expect(clearBuffer?.length).toBe(SLOT_COUNT);
  const slots = nodes[ids.indexOf('tail-slots')];
  expect(slots.resources?.map(resource => getResourceBuffer(resource))).toContain(clearBuffer);
  const sources = getParsedKernelSources(nodes);
  const slotsSource = sources.get('tail-slots') ?? '';
  expect(slotsSource).toContain(
    'let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);'
  );
  // Rule 14: summary words take one atomic per workgroup.
  expect(slotsSource).toContain('workgroupCounters');
  expect(sources.get('tail-vertices')).toContain('workgroupCounters');
});
