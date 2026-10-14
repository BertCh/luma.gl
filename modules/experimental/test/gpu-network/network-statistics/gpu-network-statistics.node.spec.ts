// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  decodeGPUNetworkStatistics,
  getGPUNetworkStatisticsParameterValues,
  getGPUNetworkStatisticsLength,
  GPUNetworkStatistics,
  type GPUNetworkStatisticsProps
} from '../../../src/gpu-network/network-statistics';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

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
    topology: {
      offsets: importView('offsets', NODE_COUNT + 1),
      neighbors: importView('neighbors', SLOT_COUNT)
    },
    output: importView('output', getGPUNetworkStatisticsLength(4)),
    degreeBinCount: 4,
    ...overrides
  });
  return {graph, importView, props};
}

it('GPUNetworkStatistics schedules a fixed node order', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const contributor = new GPUNetworkStatistics({
    id: 'stats',
    ...props({
      directed: true,
      nodeMask: importView('vertex-mask', NODE_COUNT),
      edgeMask: importView('edge-mask', SLOT_COUNT),
      communities: importView('communities', NODE_COUNT),
      parameters: importView('parameters', 2)
    })
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  const indexOf = (prefix: string) => ids.findIndex(id => id.startsWith(prefix));
  expect(ids[0]).toBe('stats-zero-output');
  expect(indexOf('stats-slots')).toBeGreaterThan(indexOf('stats-zero-in-sum'));
  expect(indexOf('stats-components')).toBeGreaterThan(indexOf('stats-slots'));
  expect(indexOf('stats-component-sizes')).toBeGreaterThan(indexOf('stats-components'));
  expect(indexOf('stats-degrees')).toBeGreaterThan(indexOf('stats-component-sizes'));
  expect(indexOf('stats-intra-community')).toBeGreaterThan(indexOf('stats-degrees'));
  expect(ids[ids.length - 1]).toBe('stats-finish');
  // Load-balanced slot loops: row lengths (walk, fixup), self-loops per row (walk, fixup) and a
  // finalize that folds the summary words.
  expect(indexOf('stats-slots-fixup')).toBe(indexOf('stats-slots') + 1);
  expect(indexOf('stats-self-loops')).toBe(indexOf('stats-slots') + 2);
  expect(indexOf('stats-self-loops-fixup')).toBe(indexOf('stats-slots') + 3);
  expect(indexOf('stats-slots-finalize')).toBe(indexOf('stats-slots') + 4);
  expect(indexOf('stats-components')).toBeGreaterThan(indexOf('stats-slots-finalize'));
  expect(indexOf('stats-intra-community-fixup')).toBe(indexOf('stats-intra-community') + 1);
  expect(indexOf('stats-intra-community-total')).toBe(indexOf('stats-intra-community') + 2);
  device.destroy();
});

it('GPUNetworkStatistics omits unused passes and defaults the id', () => {
  const device = createNullWebGPUDevice();
  const {graph, props} = createContext(device);
  const contributor = new GPUNetworkStatistics(props());
  expect(contributor.id).toBe('network-statistics');
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => id.includes('intra-community'))).toBe(false);
  expect(ids.some(id => id.includes('zero-in-degree'))).toBe(false);
  expect(ids.some(id => id.includes('zero-out-sum'))).toBe(false);
  device.destroy();
});

it('GPUNetworkStatistics validates props', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const create = (overrides: Partial<GPUNetworkStatisticsProps>) =>
    new GPUNetworkStatistics(props(overrides));

  expect(() =>
    create({
      topology: {offsets: importView('offsets-one', 1), neighbors: importView('n-one', SLOT_COUNT)}
    })
  ).toThrow(/topology\.offsets must contain at least two rows/);
  expect(() =>
    create({
      topology: {
        offsets: importView('offsets-w', NODE_COUNT + 1),
        neighbors: importView('n-w', SLOT_COUNT),
        weights: createTransientView(graph, 'weights-short', 'float32', SLOT_COUNT - 1)
      }
    })
  ).toThrow(/topology\.weights length/);
  expect(() => create({degreeBinCount: 0})).toThrow(/degreeBinCount/);
  expect(() => create({degreeBinCount: 2.5})).toThrow(/degreeBinCount/);
  expect(() => create({degreeBinning: 'sqrt' as never})).toThrow(/degreeBinning/);
  expect(() => create({maximumComponentIterations: 0})).toThrow(/maximumComponentIterations/);
  expect(() => create({output: importView('output-short', 10)})).toThrow(
    /output must contain exactly 28 rows/
  );
  expect(() => create({nodeMask: importView('mask-short', 3)})).toThrow(/nodeMask must contain/);
  expect(() => create({edgeMask: importView('edge-short', 3)})).toThrow(/edgeMask must contain/);
  expect(() => create({communities: importView('communities-short', 3)})).toThrow(
    /communities must contain/
  );
  expect(() => create({parameters: importView('parameters-short', 1)})).toThrow(
    /parameters must contain at least two/
  );
  const aliased = importView('aliased', getGPUNetworkStatisticsLength(4));
  expect(() =>
    create({
      topology: {offsets: importView('offsets-a', NODE_COUNT + 1), neighbors: aliased},
      output: aliased
    })
  ).toThrow(/outputs must not share buffers with inputs/);
  // The contributor owns no buffers and takes no device from its views, so graph transient
  // topology views are accepted.
  const transientTopology = new GPUNetworkStatistics(
    props({
      topology: {
        offsets: createTransientView(graph, 'transient-offsets', 'uint32', NODE_COUNT + 1),
        neighbors: createTransientView(graph, 'transient-neighbors', 'uint32', SLOT_COUNT)
      }
    })
  );
  expect(() => transientTopology.getCommandNodes(graph)).not.toThrow();
  expect('destroy' in transientTopology).toBe(false);

  // Views from another graph are rejected when the contributor is added.
  const otherDevice = createNullWebGPUDevice();
  const other = createContext(otherDevice);
  const foreign = new GPUNetworkStatistics(other.props());
  expect(() => foreign.getCommandNodes(graph)).toThrow(/views must belong to the target graph/);
  otherDevice.destroy();
  device.destroy();
});

it('GPUNetworkStatistics stays inside the 8 storage binding limit', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const contributor = new GPUNetworkStatistics(
    props({
      directed: true,
      nodeMask: importView('vertex-mask', NODE_COUNT),
      edgeMask: importView('edge-mask', SLOT_COUNT),
      communities: importView('communities', NODE_COUNT),
      parameters: importView('parameters', 2)
    })
  );
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  device.destroy();
});

it('getGPUNetworkStatisticsParameterValues and decodeGPUNetworkStatistics round trip', () => {
  const words = getGPUNetworkStatisticsParameterValues({resolution: 1.5, degreeBinWidth: 4.7});
  expect(words[1]).toBe(4);
  expect(new Float32Array(words.buffer)[0]).toBe(1.5);
  expect(Array.from(getGPUNetworkStatisticsParameterValues())).toEqual([0x3f800000, 1]);
  expect(Array.from(getGPUNetworkStatisticsParameterValues({degreeBinWidth: 0}))[1]).toBe(1);

  const summary = new Uint32Array(getGPUNetworkStatisticsLength(3));
  summary.set([7, 8, 9, 2, 5, 1, 1, 4, 3, 6, 0, 1, 11, 2]);
  new Float32Array(summary.buffer)[10] = 0.25;
  summary.set([1, 2, 3, 4, 5, 6, 7, 8, 9], 16);
  const decoded = decodeGPUNetworkStatistics(summary, {degreeBinCount: 3});
  expect(decoded).toMatchObject({
    liveNodeCount: 7,
    liveEdgeCount: 8,
    liveSlotCount: 9,
    componentCount: 2,
    largestComponentSize: 5,
    componentsConverged: true,
    isolatedNodeCount: 1,
    maximumOutDegree: 4,
    maximumInDegree: 3,
    maximumTotalDegree: 6,
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

it('GPUNetworkStatistics finishes with a two-stage reduction and parses', () => {
  const device = createNullWebGPUDevice();
  const {graph, importView, props} = createContext(device);
  const nodes = new GPUNetworkStatistics({
    id: 'two-stage',
    ...props({communities: importView('communities', NODE_COUNT)})
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  // Review E-12: partials per 256 labels, then one workgroup over the partials.
  expect(ids.indexOf('two-stage-finish-partials')).toBeGreaterThanOrEqual(0);
  expect(ids.indexOf('two-stage-finish-partials')).toBeLessThan(ids.indexOf('two-stage-finish'));
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('two-stage-finish')).toContain('PARTIAL_COUNT');
  expect(sources.get('two-stage-finish')).not.toContain('componentSize[');
  // Rule 14: the intra-community count folds per workgroup.
  expect(sources.get('two-stage-intra-community-total')).toContain('workgroupCounters');
  expect(sources.get('two-stage-slots-finalize')).toContain('workgroupCounters');
  // Review B-15: undirected self-loops always count twice in the modularity volume.
  expect(sources.get('two-stage-finish')).toContain('f32(liveSlots + selfLoops)');
});
