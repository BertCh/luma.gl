// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTrajectoryClosestApproach,
  type GPUTrajectoryClosestApproachProps
} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {CLOSEST_APPROACH_FIXTURE, computeClosestApproachOracle} from './closest-approach-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  trackCount: number,
  overrides: Partial<GPUTrajectoryClosestApproachProps> = {}
): GPUTrajectoryClosestApproachProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const rows = trackCount * 8;
  return {
    positions: view('float32x2', rows),
    timestamps: view('float32', rows),
    trackOffsets: view('uint32', trackCount + 1),
    trackValid: view('uint32', trackCount),
    distance: view('float32', 1),
    cellSize: 10,
    timeCellSize: 60,
    entryCapacity: 1024,
    hitCapacity: 512,
    output: {
      pairs: {
        ids: view('uint32', 64),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)},
        partners: view('uint32', 64),
        minimumDistances: view('float32', 64),
        closestTimes: view('float32', 64),
        firstTimes: view('float32', 64),
        durations: view('float32', 64)
      },
      status: {entryOverflow: view('uint32', 1), hitOverflow: view('uint32', 1)}
    },
    ...overrides
  };
}

it('GPUTrajectoryClosestApproach declares parseable kernels within the core limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUTrajectoryClosestApproach(createProps(graph, 6)).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  const order = [
    'trajectory-closest-approach-clear',
    'trajectory-closest-approach-segments',
    'trajectory-closest-approach-segment-scan',
    'trajectory-closest-approach-entries',
    'trajectory-closest-approach-entry-sort',
    'trajectory-closest-approach-pairs',
    'trajectory-closest-approach-enter-sort',
    'trajectory-closest-approach-high-sort',
    'trajectory-closest-approach-low-sort',
    'trajectory-closest-approach-sorted-hits',
    'trajectory-closest-approach-runs',
    'trajectory-closest-approach-reduce',
    'trajectory-closest-approach-status',
    'trajectory-closest-approach-publish'
  ];
  let previous = -1;
  for (const prefix of order) {
    const index = ids.findIndex(id => id === prefix || id.startsWith(`${prefix}-`));
    expect(index, prefix).toBeGreaterThan(previous);
    previous = index;
  }
  for (const [kernelId, source] of getParsedKernelSources(nodes)) {
    expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), kernelId).toBeLessThanOrEqual(8192);
  }
  // Word timestamps build too.
  const words = new GPUCommandGraph(createNullWebGPUDevice());
  const wordProps = createProps(words, 6);
  getParsedKernelSources(
    new GPUTrajectoryClosestApproach({
      ...wordProps,
      timestamps: createTransientView(words, 'word-times', 'uint32x2', 48)
    }).getCommandNodes(words)
  );
  // Rule 12: sizes add no kernel variants.
  const other = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    getGPUCommandNodePipelineKeys(
      new GPUTrajectoryClosestApproach(createProps(other, 7)).getCommandNodes(other)
    ).size
  ).toBe(getGPUCommandNodePipelineKeys(nodes).size);
});

it('GPUTrajectoryClosestApproach validates props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 4);
  expect(
    () =>
      new GPUTrajectoryClosestApproach({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow();
  expect(() => new GPUTrajectoryClosestApproach({...props, cellSize: 0})).toThrow(/cellSize/);
  expect(() => new GPUTrajectoryClosestApproach({...props, timeCellSize: NaN})).toThrow(
    /timeCellSize/
  );
  expect(() => new GPUTrajectoryClosestApproach({...props, entryCapacity: 0})).toThrow(
    /entryCapacity/
  );
  expect(() => new GPUTrajectoryClosestApproach({...props, hitCapacity: 1.5})).toThrow(
    /hitCapacity/
  );
  expect(
    () =>
      new GPUTrajectoryClosestApproach({
        ...props,
        timestamps: createTransientView(graph, 'short-times', 'float32', 3)
      })
  ).toThrow(/timestamps length/);
  expect(
    () =>
      new GPUTrajectoryClosestApproach({
        ...props,
        output: {
          ...props.output,
          pairs: {
            ...props.output.pairs,
            durations: createTransientView(graph, 'short-durations', 'float32', 3)
          }
        }
      })
  ).toThrow(/durations/);
});

it('closest-approach JS oracle reproduces the generated fixture', () => {
  for (const scene of CLOSEST_APPROACH_FIXTURE.scenes) {
    const actual = computeClosestApproachOracle(scene.tracks, scene.distance);
    expect(
      actual.map(pair => [pair.track, pair.partner]),
      scene.name
    ).toEqual(scene.pairs.map(pair => [pair.track, pair.partner]));
    actual.forEach((pair, index) => {
      const expected = scene.pairs[index];
      expect(pair.minimumDistance, `${scene.name} ${index}`).toBeCloseTo(
        expected.minimumDistance,
        6
      );
      expect(pair.duration, `${scene.name} ${index}`).toBeCloseTo(expected.duration, 6);
      expect(pair.firstTime, `${scene.name} ${index}`).toBeCloseTo(expected.firstTime, 6);
    });
  }
});

const DISTANCE_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
} as const;
const AREA_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
} as const;

it('GPUTrajectoryClosestApproach validates its distance frame', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 2);
  expect(() => new GPUTrajectoryClosestApproach({...props, spatialContext: AREA_FRAME})).toThrow(
    /measures distance; project into an azimuthal-equidistant frame/
  );
  expect(
    () => new GPUTrajectoryClosestApproach({...props, spatialContext: DISTANCE_FRAME})
  ).not.toThrow();
  device.destroy();
});
