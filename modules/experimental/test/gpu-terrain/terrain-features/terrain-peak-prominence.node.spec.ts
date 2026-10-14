// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainPeakProminence,
  getGPUTerrainPeakProminenceMaximumCellCount
} from '../../../src/gpu-terrain';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import fixture from '../../oracles/terrain-peak-prominence.json';
import {getParallelProminence, getSerialProminence} from './terrain-peak-prominence-model';

function build(width: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const grid = makeGPURasterMetadata({width, height: 1});
  const contributor = new GPUTerrainPeakProminence({
    grid,
    elevation: makeGPURasterBand(createTransientView(graph, 'elevation', 'float32', width), grid),
    output: {
      prominence: createTransientView(graph, 'prominence', 'float32', width),
      keySaddle: createTransientView(graph, 'saddle', 'float32', width),
      parentPeak: createTransientView(graph, 'parent', 'uint32', width)
    }
  });
  return {graph, contributor, nodes: contributor.getCommandNodes(graph)};
}

it('builds parallel splicing and gated elder doubling with invariant pipelines across round counts', () => {
  const counts = [819, 820].map(width => {
    const {nodes} = build(width);
    const tree = nodes.filter(node => node.id.includes('-tree-'));
    const jumps = tree.filter(node => /-elders-jump-\d+$/.test(node.id));
    expect(jumps).toHaveLength(Math.ceil(Math.log2(width * 5)) + 1);
    for (const node of tree.filter(node => /-elders-(jump|copy)-/.test(node.id))) {
      expect(node.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
      expect(node.workload?.maximumInvocationCount).toBe(width * 5);
    }
    const sources = getParsedKernelSources(nodes);
    for (const source of sources.values())
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    expect(sources.get('terrain-peak-prominence-tree-splice')).toContain('atomicMin(&parents');
    expect(sources.get('terrain-peak-prominence-tree-record')).toContain(
      'firstHeight == secondHeight && firstPeak < secondPeak'
    );
    expect(
      nodes.find(node => node.id.endsWith('-tree-splice'))?.workload?.maximumInvocationCount
    ).toBe(width * 4);
    expect(
      nodes.find(node => node.id.endsWith('-tree-record'))?.workload?.maximumInvocationCount
    ).toBe(width * 4);
    expect(nodes.some(node => node.id.includes('-union-'))).toBe(false);
    expect(nodes.findIndex(node => node.id.endsWith('-tree-splice'))).toBeGreaterThan(
      nodes.findIndex(node => node.id.endsWith('-spill-edges'))
    );
    return getGPUCommandNodePipelineKeys(tree).size;
  });
  expect(counts[1]).toBe(counts[0]);
});

it('records the independent vertex-activation reference and adversarial definitions', () => {
  expect(fixture.definition).toContain('vertex activation');
  expect(fixture.cases.find(testCase => testCase.name === 'two-peaks')?.prominence).toEqual([
    0,
    'Infinity',
    0,
    4,
    0
  ]);
  expect(fixture.cases.find(testCase => testCase.name === 'equal-peaks')?.prominence).toEqual([
    'Infinity',
    0,
    7,
    0,
    4
  ]);
  expect(fixture.cases.find(testCase => testCase.name === 'invalid')?.prominence).toEqual([
    'NaN',
    'NaN',
    'NaN',
    'NaN'
  ]);
});

it('rejects output length mismatch and input/output aliases', () => {
  const {graph, contributor} = build(4);
  expect(
    () =>
      new GPUTerrainPeakProminence({
        ...contributor.props,
        output: {prominence: createTransientView(graph, 'short', 'float32', 3)}
      })
  ).toThrow();
  const elevation = createTransientView(graph, 'shared', 'float32', 4);
  expect(
    () =>
      new GPUTerrainPeakProminence({
        ...contributor.props,
        elevation: makeGPURasterBand(elevation, contributor.props.grid),
        output: {prominence: elevation}
      })
  ).toThrow();
});

it('bounds the parallel child table by the device storage binding limit', () => {
  const device = createNullWebGPUDevice();
  const maximum = getGPUTerrainPeakProminenceMaximumCellCount(device);
  expect(maximum).toBe(Math.floor(device.limits.maxStorageBufferBindingSize / 40));
  expect(() => build(maximum + 1)).toThrow();
});

it('parallel cyclic join trees preserve the reference and serial parentPeak ties on every fixture', () => {
  for (const testCase of fixture.cases) {
    const serial = getSerialProminence(testCase);
    expect(serial.saddles, testCase.name).toEqual(testCase.keySaddle.map(Number));
    expect(serial.prominence, testCase.name).toEqual(testCase.prominence.map(Number));
    for (const seed of [1, 2, 17, 88]) {
      expect(getParallelProminence(testCase, seed), `${testCase.name}/${seed}`).toEqual(serial);
    }
  }
});

it('parallel elder propagation handles long ridges, cycles and equal-height plateaus', () => {
  for (const width of [1, 2, 33, 257]) {
    for (const plateau of [false, true]) {
      const testCase = {
        width,
        height: 3,
        elevation: Array.from({length: width * 3}, (_, cell) =>
          plateau ? 7 : (cell * 104729) % 41
        )
      };
      for (const seed of [3, 4])
        expect(getParallelProminence(testCase, seed)).toEqual(getSerialProminence(testCase));
    }
  }
});
