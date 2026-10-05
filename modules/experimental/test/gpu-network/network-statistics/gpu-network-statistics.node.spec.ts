// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPUNetworkStatistics,
  encodeGPUNetworkStatisticsParameters,
  getGPUNetworkStatisticsLength,
  GPUNetworkStatistics,
  type GPUNetworkStatisticsProps
} from '../../../src/gpu-network/network-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const NODE_COUNT = 6;
const SLOT_COUNT = 10;

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const importView = (name: string, length: number): GraphDataView<'uint32'> => {
    const uniqueName = `${name}-${buffers.length}`;
    const buffer = device.createBuffer({
      id: uniqueName,
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return importGraphBuffer(graph, uniqueName, buffer, 'uint32', length);
  };
  const props = (
    overrides: Partial<GPUNetworkStatisticsProps> = {}
  ): GPUNetworkStatisticsProps => ({
    offsets: importView('offsets', NODE_COUNT + 1),
    neighbors: importView('neighbors', SLOT_COUNT),
    output: importView('output', getGPUNetworkStatisticsLength(4)),
    degreeBinCount: 4,
    ...overrides
  });
  return {graph, importView, props};
}

it('GPUNetworkStatistics schedules a fixed node order', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const recipe = new GPUNetworkStatistics({
    id: 'stats',
    ...props({
      directed: true,
      vertexMask: importView('vertex-mask', NODE_COUNT),
      edgeMask: importView('edge-mask', SLOT_COUNT),
      communities: importView('communities', NODE_COUNT),
      parameters: importView('parameters', 2)
    })
  });
  expect(recipe.recipe).toBe('network-statistics');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  const indexOf = (prefix: string) => ids.findIndex(id => id.startsWith(prefix));
  expect(ids[0]).toBe('stats-zero-output');
  expect(indexOf('stats-slots')).toBeGreaterThan(indexOf('stats-zero-in-sum'));
  expect(indexOf('stats-components')).toBeGreaterThan(indexOf('stats-slots'));
  expect(indexOf('stats-component-sizes')).toBeGreaterThan(indexOf('stats-components'));
  expect(indexOf('stats-degrees')).toBeGreaterThan(indexOf('stats-component-sizes'));
  expect(indexOf('stats-intra-community')).toBeGreaterThan(indexOf('stats-degrees'));
  expect(ids[ids.length - 1]).toBe('stats-finish');
  recipe.destroy();
  device.destroy();
});

it('GPUNetworkStatistics omits unused passes and defaults the id', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const recipe = new GPUNetworkStatistics(props());
  expect(recipe.id).toBe('network-statistics');
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => id.includes('intra-community'))).toBe(false);
  expect(ids.some(id => id.includes('zero-in-degree'))).toBe(false);
  expect(ids.some(id => id.includes('zero-out-sum'))).toBe(false);
  recipe.destroy();
  device.destroy();
});

it('GPUNetworkStatistics validates props', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const create = (overrides: Partial<GPUNetworkStatisticsProps>) =>
    new GPUNetworkStatistics(props(overrides));

  expect(() => create({offsets: importView('offsets-one', 1)})).toThrow(/at least two rows/);
  expect(() => create({degreeBinCount: 0})).toThrow(/degreeBinCount/);
  expect(() => create({degreeBinCount: 2.5})).toThrow(/degreeBinCount/);
  expect(() => create({degreeBinning: 'sqrt' as never})).toThrow(/degreeBinning/);
  expect(() => create({componentIterations: 0})).toThrow(/componentIterations/);
  expect(() => create({output: importView('output-short', 10)})).toThrow(
    /output must contain exactly 28 rows/
  );
  expect(() => create({vertexMask: importView('mask-short', 3)})).toThrow(
    /vertexMask must contain/
  );
  expect(() => create({edgeMask: importView('edge-short', 3)})).toThrow(/edgeMask must contain/);
  expect(() => create({communities: importView('communities-short', 3)})).toThrow(
    /communities must contain/
  );
  expect(() => create({parameters: importView('parameters-short', 1)})).toThrow(
    /parameters must contain at least two/
  );
  const aliased = importView('aliased', getGPUNetworkStatisticsLength(4));
  expect(() => create({neighbors: aliased, output: aliased})).toThrow(
    /outputs must not share buffers with inputs/
  );
  expect(() =>
    create({offsets: createTransientView(graph, 'transient-offsets', 'uint32', NODE_COUNT + 1)})
  ).toThrow(/offsets must be imported with a default buffer/);

  // Views from another graph are rejected when the recipe is added.
  const otherDevice = createNullWebGPUDevice();
  const other = createContext(otherDevice);
  const foreign = new GPUNetworkStatistics(other.props());
  expect(() => foreign.getCommandNodes(graph)).toThrow(/views must belong to the target graph/);
  foreign.destroy();
  otherDevice.destroy();
  device.destroy();
});

it('GPUNetworkStatistics stays inside the 8 storage binding limit', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const recipe = new GPUNetworkStatistics(
    props({
      directed: true,
      vertexMask: importView('vertex-mask', NODE_COUNT),
      edgeMask: importView('edge-mask', SLOT_COUNT),
      communities: importView('communities', NODE_COUNT),
      parameters: importView('parameters', 2)
    })
  );
  expect(() => recipe.getCommandNodes(graph)).not.toThrow();
  recipe.destroy();
  device.destroy();
});

it('encodeGPUNetworkStatisticsParameters and decodeGPUNetworkStatistics round trip', () => {
  const words = encodeGPUNetworkStatisticsParameters({resolution: 1.5, degreeBinWidth: 4.7});
  expect(words[1]).toBe(4);
  expect(new Float32Array(words.buffer)[0]).toBe(1.5);
  expect(Array.from(encodeGPUNetworkStatisticsParameters())).toEqual([0x3f800000, 1]);
  expect(Array.from(encodeGPUNetworkStatisticsParameters({degreeBinWidth: 0}))[1]).toBe(1);

  const summary = new Uint32Array(getGPUNetworkStatisticsLength(3));
  summary.set([7, 8, 9, 2, 5, 1, 1, 4, 3, 6, 0, 1, 11, 2]);
  new Float32Array(summary.buffer)[10] = 0.25;
  summary.set([1, 2, 3, 4, 5, 6, 7, 8, 9], 16);
  const decoded = decodeGPUNetworkStatistics(summary, {degreeBinCount: 3});
  expect(decoded).toMatchObject({
    liveVertexCount: 7,
    liveEdgeCount: 8,
    liveSlotCount: 9,
    componentCount: 2,
    largestComponentSize: 5,
    componentsConverged: true,
    isolatedVertexCount: 1,
    maxOutDegree: 4,
    maxInDegree: 3,
    maxTotalDegree: 6,
    modularity: 0.25,
    modularityValid: true,
    intraCommunitySlotCount: 11,
    selfLoopSlotCount: 2,
    outDegreeHistogram: [1, 2, 3],
    inDegreeHistogram: [4, 5, 6],
    totalDegreeHistogram: [7, 8, 9]
  });
  expect(() => decodeGPUNetworkStatistics(new Uint32Array(5), {degreeBinCount: 3})).toThrow(
    /full summary/
  );
});
