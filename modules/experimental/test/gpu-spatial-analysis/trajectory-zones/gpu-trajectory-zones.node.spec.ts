// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

let uniqueIndex = 0;
const unique = (id: string) => `${id}-${uniqueIndex++}`;

function createGraph() {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: unique('graph')});
}

function createCompactOutput(graph: GPUCommandGraph, capacity: number) {
  return {
    ids: createTransientView(graph, unique('ids'), 'uint32', capacity),
    status: {
      count: createTransientView(graph, unique('count'), 'uint32', 1),
      overflow: createTransientView(graph, unique('overflow'), 'uint32', 1)
    }
  };
}

it('GPUZoneEvents validates props and declares nodes inside the storage binding limit', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const props = {
    tracks: {
      kind: 'trajectories' as const,
      positions: view('float32x2', 8),
      timestamps: view('float32', 8),
      trackOffsets: view('uint32', 3)
    },
    // Two zones of one four-vertex ring each.
    zones: {
      kind: 'polygons' as const,
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      ringOffsets: view('uint32', 3)
    },
    candidateCapacity: 16,
    trackEventCapacity: 4,
    output: {
      events: {
        ...createCompactOutput(graph, 8),
        eventTimes: view('float32', 8),
        eventZones: view('uint32', 8),
        eventTypes: view('uint32', 8),
        eventRows: view('uint32', 8)
      },
      dwellTimes: view('float32', 4),
      visitCounts: view('uint32', 4),
      trackEventCounts: view('uint32', 2)
    }
  };
  const zoneEvents = new GPUZoneEvents(props);
  const nodes = zoneEvents.getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(20);
  // The edges derive from the polygon layout in the first node, one edge per zone vertex.
  expect(nodes[0].id).toBe('zone-events-edges');
  expect(zoneEvents.zoneCount).toBe(2);
  expect(zoneEvents.edgeCount).toBe(8);
  const sources = getParsedKernelSources(nodes);
  expect(getStorageBindingCount(sources.get('zone-events-edges') ?? '')).toBe(7);
  for (const [kernelId, source] of sources) {
    expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
  }
  expect(
    () => new GPUZoneEvents({...props, zones: {...props.zones, featureOffsets: view('uint32', 1)}})
  ).toThrow(/at least one feature/);
  expect(
    () => new GPUZoneEvents({...props, output: {...props.output, dwellTimes: view('float32', 3)}})
  ).toThrow(/dwellTimes/);
  expect(
    () =>
      new GPUZoneEvents({
        ...props,
        output: {...props.output, events: {...props.output.events, eventTimes: view('float32', 7)}}
      })
  ).toThrow(/eventTimes/);
  expect(() => new GPUZoneEvents({...props, trackEventCapacity: 0})).toThrow(/trackEventCapacity/);
  expect(() => new GPUZoneEvents({...props, candidateCapacity: 0})).toThrow(/candidateCapacity/);
  // Words time mode builds too.
  const words = new GPUZoneEvents({
    ...props,
    id: 'zone-events-words',
    tracks: {...props.tracks, timestamps: view('uint32x2', 8)}
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
    output: {
      pairs: {
        ...createCompactOutput(graph, 8),
        partners: view('uint32', 8),
        firstBuckets: view('uint32', 8),
        minimumDistances: view('float32', 8),
        bucketCounts: view('uint32', 8)
      },
      status: {candidateOverflow: view('uint32', 1)}
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
        output: {...props.output, pairs: {...props.output.pairs, firstTimes: view('float32', 8)}}
      })
  ).toThrow(/bucketTimes/);
  // The lattice limit depends on the device, so it is checked when the nodes are declared.
  expect(() =>
    new GPUTrajectoryEncounters({...props, cellSize: 1e-3}).getCommandNodes(graph)
  ).toThrow(/columns x rows x bucketCount/);
});

it('GPUTrackSimilarity validates props and the Frechet cap', () => {
  const graph = createGraph();
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, unique('view'), format, length);
  const props = {
    linesA: {
      kind: 'lines' as const,
      positions: view('float32x2', 6),
      lineOffsets: view('uint32', 3)
    },
    pairA: view('uint32', 2),
    pairB: view('uint32', 2),
    output: {hausdorff: view('float32', 2), frechet: view('float32', 2)}
  };
  const similarity = new GPUTrackSimilarity(props);
  expect(similarity.maximumFrechetVertexCount).toBe(256);
  expect(similarity.densifySubdivisions).toBe(1);
  expect(similarity.getCommandNodes(graph).length).toBe(3);
  expect(new GPUTrackSimilarity({...props, densify: 0.25}).densifySubdivisions).toBe(4);
  expect(() => new GPUTrackSimilarity({...props, maximumFrechetVertexCount: 2049})).toThrow(
    /maximumFrechetVertexCount/
  );
  expect(() => new GPUTrackSimilarity({...props, densify: 1.5})).toThrow(/densify/);
  expect(
    () =>
      new GPUTrackSimilarity({
        ...props,
        linesB: {kind: 'lines', positions: view('float32x2', 2), lineOffsets: view('uint32', 1)}
      })
  ).toThrow(/linesB.lineOffsets/);
  expect(
    () => new GPUTrackSimilarity({...props, output: {hausdorff: undefined, frechet: undefined}})
  ).toThrow(/needs hausdorff/);
  expect(() => new GPUTrackSimilarity({...props, pairB: view('uint32', 3)})).toThrow(/pairA/);
});
