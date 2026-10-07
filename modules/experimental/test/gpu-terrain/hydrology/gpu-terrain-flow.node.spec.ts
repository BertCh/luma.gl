// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowProps
} from '../../../src/gpu-terrain/hydrology';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainFlowProps> = {}) => {
    instance++;
    return new GPUTerrainFlow({
      width: 6,
      height: 5,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `elevation-${instance}`, 'float32', 30)
        }
      },
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      flowDirections: createTransientView(graph, `directions-${instance}`, 'uint32', 30),
      ...overrides
    });
  };
  return {device, graph, create};
}

it('getGPUTerrainFlowParameterValues packs settings', () => {
  expect(GPU_TERRAIN_FLOW_PARAMETER_LENGTH).toBe(8);
  expect(
    Array.from(
      getGPUTerrainFlowParameterValues({
        cellSize: [2, 3],
        northEdge: 0.25,
        southEdge: 0.5,
        fillEpsilon: 0.125,
        streamThreshold: 10
      })
    )
  ).toEqual([2, 3, 0.25, 0.5, 0.125, 10, 0, 0]);
  const defaults = getGPUTerrainFlowParameterValues({cellSize: [1, 1]});
  expect(defaults[4]).toBe(0);
  expect(defaults[5]).toBe(Infinity);
  expect(() => getGPUTerrainFlowParameterValues({cellSize: [1, 1]}, new Float32Array(7))).toThrow(
    /8 values/
  );
});

it('GPUTerrainFlow prefixes node ids and grows with iteration limits', () => {
  const {device, graph, create} = createFixture();
  const small = create({
    maxFillIterations: 2,
    maxAccumulationIterations: 2,
    fillDepressions: true,
    accumulation: createTransientView(graph, 'acc-small', 'float32', 30),
    fillConverged: createTransientView(graph, 'fc-small', 'uint32', 1),
    accumulationConverged: createTransientView(graph, 'ac-small', 'uint32', 1)
  });
  const smallNodes = small.getCommandNodes(graph);
  expect(smallNodes.every(node => node.id.startsWith('terrain-flow-'))).toBe(true);
  const ids = smallNodes.map(node => node.id);
  expect(ids).toContain('terrain-flow-flow-direction');
  expect(ids).toContain('terrain-flow-fill-finalize');
  expect(ids).toContain('terrain-flow-accumulate-finalize');
  expect(new Set(ids).size).toBe(ids.length);

  const {graph: largeGraph, create: createLarge} = createFixture();
  const large = createLarge({
    id: 'big',
    maxFillIterations: 5,
    maxAccumulationIterations: 7,
    fillDepressions: true,
    streams: createTransientView(largeGraph, 'streams', 'uint32', 30)
  });
  const largeNodes = large.getCommandNodes(largeGraph);
  expect(largeNodes.every(node => node.id.startsWith('big-'))).toBe(true);
  // Fill adds two resets, an init, one node per iteration, and one gate per four iterations.
  const withoutFill = createLarge({
    id: 'nofill',
    maxAccumulationIterations: 7,
    streams: createTransientView(largeGraph, 'streams2', 'uint32', 30)
  }).getCommandNodes(largeGraph);
  expect(largeNodes.length - withoutFill.length).toBe(3 + 5 + 2);
  const moreRounds = createLarge({
    id: 'rounds',
    maxAccumulationIterations: 9,
    streams: createTransientView(largeGraph, 'streams3', 'uint32', 30)
  }).getCommandNodes(largeGraph);
  expect(moreRounds.length - withoutFill.length).toBe(2 + 1);
  // Direction-only requests skip accumulation entirely.
  expect(
    create({id: 'dir'})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['dir-elevation', 'dir-flow-direction']);
  device.destroy();
});

it('GPUTerrainFlow validates its props', () => {
  const {device, graph, create} = createFixture();
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({flowDirections: undefined})).toThrow(/at least one output/);
  expect(() => create({settings: view('s7', 'float32', 7)})).toThrow(/settings/);
  expect(() => create({flowDirections: view('short', 'uint32', 29)})).toThrow(/one value per cell/);
  expect(() => create({accumulation: view('wrong-format', 'uint32', 30) as never})).toThrow(
    /float32/
  );
  expect(() => create({filledElevation: view('filled', 'float32', 30)})).toThrow(/fillDepressions/);
  expect(() => create({fillConverged: view('fc', 'uint32', 1)})).toThrow(/fillDepressions/);
  expect(() => create({accumulationConverged: view('ac', 'uint32', 1)})).toThrow(
    /accumulation or streams/
  );
  expect(() =>
    create({streams: view('st', 'uint32', 30), accumulationConverged: view('ac2', 'uint32', 2)})
  ).toThrow(/one row/);
  expect(() => create({maxFillIterations: 0})).toThrow(/maxFillIterations/);
  expect(() => create({maxAccumulationIterations: 2000})).toThrow(/maxAccumulationIterations/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({accumulationUnits: 'acres' as never})).toThrow(/accumulationUnits/);
  expect(() => create({runoff: view('runoff', 'float32', 29)})).toThrow(/one value per cell/);
  const shared = view('shared', 'uint32', 30);
  expect(() => create({flowDirections: shared, cellClasses: shared})).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainFlow({
      width: 2,
      height: 2,
      elevation: {
        id: 'foreign',
        format: 'float32',
        storage: {kind: 'buffer', values: createTransientView(otherGraph, 'foreign', 'float32', 4)}
      },
      settings: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      flowDirections: createTransientView(otherGraph, 'foreign-out', 'uint32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainFlow lets outputs share a buffer only over disjoint byte ranges', () => {
  const {device, graph, create} = createFixture();
  const summary = graph.createTransientBuffer({id: 'summary', byteLength: 1024, usage: 128});
  const flagsAt = (byteOffset: number) =>
    graph.createDataView(summary, {format: 'uint32', length: 1, byteOffset});
  const common = {fillDepressions: true, streams: createTransientView(graph, 'st', 'uint32', 30)};

  // Two 1-row flags in different words of one buffer are accepted and schedule normally.
  const contributor = create({
    ...common,
    fillConverged: flagsAt(0),
    accumulationConverged: flagsAt(4)
  });
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();

  // The same bytes, or overlapping ranges, are refused.
  expect(() =>
    create({...common, fillConverged: flagsAt(0), accumulationConverged: flagsAt(0)})
  ).toThrow(/overlapping byte ranges/);
  expect(() =>
    create({
      flowDirections: graph.createDataView(summary, {format: 'uint32', length: 30}),
      cellClasses: graph.createDataView(summary, {format: 'uint32', length: 30, byteOffset: 64})
    })
  ).toThrow(/overlapping byte ranges/);

  // Disjoint ranges are not enough for outputs written by one node: they would share a
  // storage binding window.
  const columns = graph.createTransientBuffer({id: 'columns', byteLength: 4096, usage: 128});
  const column = (byteOffset: number) =>
    graph.createDataView(columns, {format: 'uint32', length: 30, byteOffset});
  expect(() => create({flowDirections: column(0), cellClasses: column(128)})).toThrow(
    /256 bytes apart/
  );
  expect(() => create({flowDirections: column(0), cellClasses: column(256)})).not.toThrow();

  // An output may never share a buffer with an input, even over disjoint ranges.
  const inputs = graph.createTransientBuffer({id: 'inputs', byteLength: 4096, usage: 128});
  const settings = graph.createDataView(inputs, {format: 'float32', length: 8});
  expect(() =>
    create({
      settings,
      fillDepressions: true,
      fillConverged: graph.createDataView(inputs, {format: 'uint32', length: 1, byteOffset: 1024})
    })
  ).toThrow(/share buffers with inputs/);
  device.destroy();
});
