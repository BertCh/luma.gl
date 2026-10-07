// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let uniqueIndex = 0;
const unique = (id: string) => `${id}-${uniqueIndex++}`;

function createGraph() {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: unique('graph')});
}

function createCompactOutput(graph: GPUCommandGraph, capacity: number) {
  return {
    ids: createTransientView(graph, unique('ids'), 'uint32', capacity),
    count: createTransientView(graph, unique('count'), 'uint32', 1),
    overflow: createTransientView(graph, unique('overflow'), 'uint32', 1)
  };
}

it('GPUZoneEvents validates props and declares nodes inside the storage binding limit', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const props = {
    positions: view('float32x2', 8),
    timestamps: view('float32', 8),
    trackOffsets: view('uint32', 3),
    edgeStarts: view('float32x2', 4),
    edgeEnds: view('float32x2', 4),
    edgeZones: view('uint32', 4),
    zoneCount: 2,
    candidateCapacity: 16,
    maxEventsPerTrack: 4,
    events: {
      output: createCompactOutput(graph, 8),
      eventTimes: view('float32', 8),
      eventZones: view('uint32', 8),
      eventTypes: view('uint32', 8),
      eventRows: view('uint32', 8)
    },
    dwellTimes: view('float32', 4),
    visitCounts: view('uint32', 4),
    trackEventCounts: view('uint32', 2)
  };
  const zoneEvents = new GPUZoneEvents(props);
  expect(zoneEvents.getCommandNodes(graph).length).toBeGreaterThan(20);
  expect(() => new GPUZoneEvents({...props, zoneCount: 0})).toThrow(/zoneCount/);
  expect(() => new GPUZoneEvents({...props, dwellTimes: view('float32', 3)})).toThrow(/dwellTimes/);
  expect(
    () => new GPUZoneEvents({...props, events: {...props.events, eventTimes: view('float32', 7)}})
  ).toThrow(/eventTimes/);
  expect(() => new GPUZoneEvents({...props, edgeZones: view('uint32', 3)})).toThrow(/same length/);
  // Words time mode builds too.
  const words = new GPUZoneEvents({
    ...props,
    id: 'zone-events-words',
    timestamps: view('uint32x2', 8)
  });
  expect(words.getCommandNodes(graph).length).toBeGreaterThan(20);
});

it('GPUTrajectoryEncounters validates props and declares nodes', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const props = {
    samples: view('float32x2', 12),
    trackCount: 3,
    bucketCount: 4,
    distance: view('float32', 1),
    cellSize: 10,
    bounds: [0, 0, 100, 100] as const,
    hitCapacity: 32,
    pairs: {
      output: createCompactOutput(graph, 8),
      partners: view('uint32', 8),
      firstBuckets: view('uint32', 8),
      minimumDistances: view('float32', 8),
      bucketCounts: view('uint32', 8)
    }
  };
  expect(new GPUTrajectoryEncounters(props).getCommandNodes(graph).length).toBeGreaterThan(10);
  expect(() => new GPUTrajectoryEncounters({...props, samples: view('float32x2', 11)})).toThrow(
    /samples length/
  );
  expect(() => new GPUTrajectoryEncounters({...props, cellSize: 0})).toThrow(/cellSize/);
  expect(() => new GPUTrajectoryEncounters({...props, bounds: [0, 0, 0, 1]})).toThrow(/bounds/);
  expect(
    () =>
      new GPUTrajectoryEncounters({
        ...props,
        pairs: {...props.pairs, firstTimes: view('float32', 8)}
      })
  ).toThrow(/bucketTimes/);
  expect(() => new GPUTrajectoryEncounters({...props, cellSize: 1e-3})).toThrow(/lattice/);
});

it('GPUTrackSimilarity validates props and the Frechet cap', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const props = {
    positionsA: view('float32x2', 6),
    offsetsA: view('uint32', 3),
    pairA: view('uint32', 2),
    pairB: view('uint32', 2),
    hausdorff: view('float32', 2),
    frechet: view('float32', 2)
  };
  const similarity = new GPUTrackSimilarity(props);
  expect(similarity.maxFrechetVertices).toBe(256);
  expect(similarity.densifySubdivisions).toBe(1);
  expect(similarity.getCommandNodes(graph).length).toBe(3);
  expect(new GPUTrackSimilarity({...props, densify: 0.25}).densifySubdivisions).toBe(4);
  expect(() => new GPUTrackSimilarity({...props, maxFrechetVertices: 2049})).toThrow(
    /maxFrechetVertices/
  );
  expect(() => new GPUTrackSimilarity({...props, densify: 1.5})).toThrow(/densify/);
  expect(() => new GPUTrackSimilarity({...props, positionsB: view('float32x2', 2)})).toThrow(
    /together/
  );
  expect(
    () => new GPUTrackSimilarity({...props, hausdorff: undefined, frechet: undefined})
  ).toThrow(/output/);
  expect(() => new GPUTrackSimilarity({...props, pairB: view('uint32', 3)})).toThrow(/pairA/);
});
