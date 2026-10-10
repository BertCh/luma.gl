// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTrajectoryMetricsParameterValues,
  GPUTrajectoryMetrics,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  type GPUTrajectoryMetricsProps
} from '../../../src/gpu-spatial-analysis/trajectory-analysis';
import {getSegmentedReductionNodes} from '../../../src/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-kernels';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let uniqueIndex = 0;
const unique = (id: string) => `${id}-${uniqueIndex++}`;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUTrajectoryMetricsProps> = {}
): GPUTrajectoryMetricsProps {
  return {
    spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
    positions: createTransientView(graph, unique('positions'), 'float32x2', 8),
    timestamps: createTransientView(graph, unique('timestamps'), 'float32', 8),
    trackOffsets: createTransientView(graph, unique('offsets'), 'uint32', 3),
    parameters: createTransientView(graph, unique('parameters'), 'float32', 4),
    trackLengths: createTransientView(graph, unique('lengths'), 'float32', 2),
    ...overrides
  };
}

function createStops(graph: GPUCommandGraph, capacity: number = 4) {
  return {
    output: {
      ids: createTransientView(graph, unique('stop-ids'), 'uint32', capacity),
      count: createTransientView(graph, unique('stop-count'), 'uint32', 1),
      overflow: createTransientView(graph, unique('stop-overflow'), 'uint32', 1)
    }
  };
}

it('getGPUTrajectoryMetricsParameterValues packs and validates', () => {
  expect(
    Array.from(
      getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 0.5, stopMinimumDuration: 30})
    )
  ).toEqual([0.5, 30, 0, 0]);
  const target = new Float32Array(6).fill(9);
  getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 1, stopMinimumDuration: 2}, target);
  expect(Array.from(target)).toEqual([1, 2, 0, 0, 9, 9]);
  expect(GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH).toBe(4);
  expect(() =>
    getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: Number.NaN, stopMinimumDuration: 1})
  ).toThrow(/finite/);
  expect(() =>
    getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: 1, stopMinimumDuration: Infinity})
  ).toThrow(/finite/);
  expect(() =>
    getGPUTrajectoryMetricsParameterValues({stopSpeedThreshold: -1, stopMinimumDuration: 1})
  ).toThrow(/minimum/);
  expect(() =>
    getGPUTrajectoryMetricsParameterValues(
      {stopSpeedThreshold: 1, stopMinimumDuration: 1},
      new Float32Array(3)
    )
  ).toThrow(/4 elements/);
});

it('GPUTrajectoryMetrics omits stop nodes without stop outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUTrajectoryMetrics(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids[0]).toBe('trajectory-metrics-steps');
  expect(ids).toContain('trajectory-metrics-track-lengths');
  expect(ids.some(id => id.includes('run-') || id.includes('stop'))).toBe(false);
  expect(ids).not.toContain('trajectory-metrics-maximum-speeds');
  expect(ids).not.toContain('trajectory-metrics-track-finalize');
  device.destroy();
});

it('GPUTrajectoryMetrics orders and prefixes stop nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUTrajectoryMetrics(
    createProps(graph, {
      id: 'tm',
      trackDurations: createTransientView(graph, 'durations', 'float32', 2),
      averageSpeeds: createTransientView(graph, 'averages', 'float32', 2),
      maximumSpeeds: createTransientView(graph, 'maxima', 'float32', 2),
      trackStopCounts: createTransientView(graph, 'stop-counts', 'uint32', 2),
      stops: {
        ...createStops(graph),
        startRows: createTransientView(graph, 'starts', 'uint32', 4),
        endRows: createTransientView(graph, 'ends', 'uint32', 4),
        centroids: createTransientView(graph, 'centroids', 'float32x2', 4),
        durations: createTransientView(graph, 'stop-durations', 'float32', 4)
      }
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.every(id => id.startsWith('tm-'))).toBe(true);
  expect(new Set(ids).size).toBe(ids.length);
  const order = [
    'tm-steps',
    'tm-track-lengths',
    'tm-maximum-speeds',
    'tm-track-finalize',
    'tm-row-ids',
    'tm-run-qualify',
    'tm-centroid-offsets',
    'tm-split-positions',
    'tm-centroid-sums-x',
    'tm-centroid-sums-y',
    'tm-stop-gather',
    'tm-stop-gather-durations',
    'tm-stop-gather-centroids',
    'tm-publish'
  ].map(id => ids.indexOf(id));
  expect(order.every(index => index >= 0)).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  expect(ids.some(id => id.startsWith('tm-run-starts'))).toBe(true);
  expect(ids.some(id => id.startsWith('tm-run-ends'))).toBe(true);
  expect(ids.some(id => id.startsWith('tm-stop-indices'))).toBe(true);
  expect(ids.some(id => id.startsWith('tm-stop-counts'))).toBe(true);
  expect(ids.at(-1)).toBe('tm-publish');
  device.destroy();
});

it('GPUTrajectoryMetrics schedules only stop counts when stops are absent', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUTrajectoryMetrics(
    createProps(graph, {
      trackLengths: undefined,
      trackStopCounts: createTransientView(graph, 'stop-counts', 'uint32', 2)
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.some(id => id.startsWith('trajectory-metrics-stop-counts'))).toBe(true);
  expect(ids).not.toContain('trajectory-metrics-publish');
  expect(ids.some(id => id.includes('stop-gather'))).toBe(false);
  expect(ids.some(id => id.includes('centroid'))).toBe(false);
  device.destroy();
});

it('GPUTrajectoryMetrics validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length);
  const create = (overrides: Partial<GPUTrajectoryMetricsProps>) =>
    new GPUTrajectoryMetrics(createProps(graph, overrides));

  expect(() => create({})).not.toThrow();
  expect(() => create({timestamps: view('t7', 'float32', 7)})).toThrow(/timestamps length/);
  expect(() => create({trackOffsets: view('o1', 'uint32', 1)})).toThrow(/at least two rows/);
  expect(() => create({trackLengths: view('l3', 'float32', 3)})).toThrow(/track count/);
  expect(() => create({maximumSpeeds: view('m3', 'float32', 3)})).toThrow(/maximumSpeeds/);
  expect(() => create({parameters: view('p3', 'float32', 3)})).toThrow(/parameters must hold/);
  expect(() => create({trackLengths: undefined, trackDurations: undefined})).toThrow(
    /at least one output/
  );
  expect(() => create({parameters: undefined, stops: createStops(graph)})).toThrow(/parameters/);
  expect(() => create({parameters: undefined, trackStopCounts: view('c2', 'uint32', 2)})).toThrow(
    /parameters/
  );
  expect(() => create({trackStopCounts: view('c3', 'uint32', 3)})).toThrow(/trackStopCounts/);
  expect(() =>
    create({stops: {...createStops(graph, 4), startRows: view('s3', 'uint32', 3)}})
  ).toThrow(/stop capacity/);
  expect(() =>
    create({stops: {...createStops(graph, 4), centroids: view('c4', 'float32', 4) as never}})
  ).toThrow(/centroids/);

  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 256, usage: 128});
  expect(() =>
    create({
      trackLengths: graph.createDataView(shared, {format: 'float32', length: 2}),
      timestamps: graph.createDataView(shared, {format: 'float32', length: 8, byteOffset: 64})
    })
  ).toThrow(/must not share buffers/);

  const otherGraph = new GPUCommandGraph(device);
  const foreign = new GPUTrajectoryMetrics(createProps(otherGraph));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/target graph/);
  device.destroy();
});

it('getSegmentedReductionNodes chunks segments above the dispatch limit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const options = {
    id: 'reduce',
    input: createTransientView(graph, 'input', 'float32', 20),
    segmentOffsets: createTransientView(graph, 'segments', 'uint32', 11),
    output: createTransientView(graph, 'sums', 'float32', 10),
    operation: 'sum' as const
  };
  expect(getSegmentedReductionNodes(graph, options, 10).map(node => node.id)).toEqual(['reduce']);
  expect(getSegmentedReductionNodes(graph, options, 4).map(node => node.id)).toEqual([
    'reduce-chunk-0',
    'reduce-chunk-1',
    'reduce-chunk-2'
  ]);
  device.destroy();
});

it('GPUTrajectoryMetrics validates word and double-single timestamp inputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const create = (overrides: Partial<GPUTrajectoryMetricsProps>) =>
    new GPUTrajectoryMetrics(createProps(graph, overrides));
  const words = (id: string, length: number) =>
    createTransientView(graph, unique(id), 'uint32x2', length);
  const floats = (id: string, length: number) =>
    createTransientView(graph, unique(id), 'float32', length);

  expect(() => create({timestamps: words('words', 8)})).not.toThrow();
  expect(() => create({timestampsLow: floats('low', 8)})).not.toThrow();
  expect(() => create({timestamps: words('words', 7)})).toThrow(/timestamps length/);
  expect(() => create({timestamps: words('words', 8), timestampsLow: floats('low', 8)})).toThrow(
    /timestampsLow requires float32/
  );
  expect(() => create({timestampsLow: floats('low', 7)})).toThrow(/timestampsLow length/);
  expect(() =>
    create({timestampsLow: createTransientView(graph, unique('low'), 'uint32', 8) as never})
  ).toThrow(/timestampsLow/);
  expect(() =>
    create({timestamps: createTransientView(graph, unique('ts'), 'uint32', 8) as never})
  ).toThrow(/timestamps/);
  device.destroy();
});

it('GPUTrajectoryMetrics splits step and qualify kernels only for double-single times', () => {
  const device = createNullWebGPUDevice();
  const getIds = (getOverrides: (graph: GPUCommandGraph) => Partial<GPUTrajectoryMetricsProps>) => {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {
      trackDurations: createTransientView(graph, unique('durations'), 'float32', 2),
      maximumSpeeds: createTransientView(graph, unique('maxima'), 'float32', 2),
      trackStopCounts: createTransientView(graph, unique('stop-counts'), 'uint32', 2),
      stops: createStops(graph),
      ...getOverrides(graph)
    });
    return new GPUTrajectoryMetrics(props).getCommandNodes(graph).map(node => node.id);
  };
  const split = [
    'trajectory-metrics-steps',
    'trajectory-metrics-steps-runs',
    'trajectory-metrics-run-qualify',
    'trajectory-metrics-run-qualify-tracks'
  ];
  const doubleSingle = getIds(graph => ({
    timestampsLow: createTransientView(graph, unique('low'), 'float32', 8)
  }));
  for (const id of split) {
    expect(doubleSingle).toContain(id);
  }
  for (const ids of [
    getIds(() => ({})),
    getIds(graph => ({
      timestamps: createTransientView(graph, unique('words'), 'uint32x2', 8)
    }))
  ]) {
    expect(ids).toContain('trajectory-metrics-steps');
    expect(ids).toContain('trajectory-metrics-run-qualify');
    expect(ids).not.toContain('trajectory-metrics-steps-runs');
    expect(ids).not.toContain('trajectory-metrics-run-qualify-tracks');
  }
  device.destroy();
});
