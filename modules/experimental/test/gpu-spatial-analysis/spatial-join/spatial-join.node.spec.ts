// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin,
  GPUSpatialPredicateJoin,
  type GPUNearestFeatureJoinProps,
  type GPUPointInPolygonJoinProps,
  type GPUSpatialPredicateJoinProps
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {
  buildModelBVH,
  getLeafCapacity,
  getMortonLeafOrder,
  measureTraversal,
  simulateTraversal,
  type FeatureBounds
} from './spatial-join-traversal-model';

function createPolygonProps(
  graph: GPUCommandGraph,
  prefix: string = '',
  overrides: Partial<GPUPointInPolygonJoinProps> = {}
): GPUPointInPolygonJoinProps {
  const view = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${prefix}${name}`, format, length);
  return {
    points: view('points', 'float32x2', 10),
    sourceIds: view('source-ids', 'uint32', 10),
    polygonPositions: view('polygon-positions', 'float32x2', 20),
    featureOffsets: view('feature-offsets', 'uint32', 5),
    polygonOffsets: view('polygon-offsets', 'uint32', 5),
    ringOffsets: view('ring-offsets', 'uint32', 6),
    featureIds: view('feature-ids', 'uint32', 4),
    candidateCapacity: 16,
    pointFeatureIds: view('point-feature-ids', 'uint32', 10),
    featureCounts: view('feature-counts', 'uint32', 4),
    overflow: view('overflow', 'uint32', 1),
    candidateCount: view('candidate-count', 'uint32', 1),
    uncertainCount: view('uncertain-count', 'uint32', 1),
    matches: {
      ids: view('match-ids', 'uint32', 10),
      count: view('match-count', 'uint32', 1),
      overflow: view('match-overflow', 'uint32', 1)
    },
    ...overrides
  };
}

function createNearestProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNearestFeatureJoinProps> = {}
): GPUNearestFeatureJoinProps {
  return {
    id: 'n',
    points: createTransientView(graph, 'points', 'float32x2', 6),
    features: {
      kind: 'segments',
      starts: createTransientView(graph, 'starts', 'float32x2', 3),
      ends: createTransientView(graph, 'ends', 'float32x2', 3)
    },
    radius: createTransientView(graph, 'radius', 'float32', 1),
    candidateCapacity: 16,
    nearestFeatureIds: createTransientView(graph, 'nearest', 'uint32', 6),
    nearestDistances: createTransientView(graph, 'distances', 'float32', 6),
    overflow: createTransientView(graph, 'overflow', 'uint32', 1),
    matches: {
      ids: createTransientView(graph, 'match-ids', 'uint32', 6),
      count: createTransientView(graph, 'match-count', 'uint32', 1),
      overflow: createTransientView(graph, 'match-overflow', 'uint32', 1)
    },
    ...overrides
  };
}

it('GPUPointInPolygonJoin schedules nodes in dependency order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUPointInPolygonJoin({...createPolygonProps(graph), id: 'j'})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.filter(id => !id.startsWith('j-bvh-'))).toEqual([
    'j-bounds',
    'j-clear',
    'j-probe',
    'j-probe-scan-level-0-scan',
    'j-probe-total',
    'j-probe-write',
    'j-candidate-dispatch',
    'j-classify',
    'j-resolve',
    'j-assign',
    'j-collect-matches',
    'j-collect-matches-scan-level-0-scan',
    'j-collect-matches-total',
    'j-collect-matches-write',
    'j-finalize'
  ]);
  const firstBVH = ids.findIndex(id => id.startsWith('j-bvh-'));
  expect(firstBVH).toBeGreaterThan(ids.indexOf('j-bounds'));
  expect(firstBVH).toBeLessThan(ids.indexOf('j-clear'));

  const second = new GPUPointInPolygonJoin({...createPolygonProps(graph, 'b-'), id: 'k'});
  const secondIds = second.getCommandNodes(graph).map(node => node.id);
  expect(secondIds.some(id => ids.includes(id))).toBe(false);
  device.destroy();
});

it('GPUPointInPolygonJoin suffixes per-chunk nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const points = createVectorView('chunked-points', 'float32x2', [
    createTransientView(graph, 'p0', 'float32x2', 6),
    createTransientView(graph, 'p1', 'float32x2', 4)
  ]);
  const pointFeatureIds = createVectorView('chunked-ids', 'uint32', [
    createTransientView(graph, 'i0', 'uint32', 6),
    createTransientView(graph, 'i1', 'uint32', 4)
  ]);
  const ids = new GPUPointInPolygonJoin({
    ...createPolygonProps(graph),
    id: 'j',
    points,
    pointFeatureIds,
    sourceIds: undefined,
    matches: undefined
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  for (const id of ['j-probe-0', 'j-probe-1', 'j-assign-0', 'j-assign-1']) {
    expect(ids).toContain(id);
  }
  device.destroy();
});

it('GPUNearestFeatureJoin schedules nodes in dependency order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUNearestFeatureJoin(createNearestProps(graph)).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.filter(id => !id.startsWith('n-bvh-'))).toEqual([
    'n-bounds',
    'n-clear',
    'n-probe',
    'n-probe-scan-level-0-scan',
    'n-probe-total',
    'n-probe-write',
    'n-candidate-dispatch',
    'n-expand',
    'n-distance',
    'n-reduce-distance',
    'n-reduce-feature',
    'n-assign',
    'n-collect-matches',
    'n-collect-matches-scan-level-0-scan',
    'n-collect-matches-total',
    'n-collect-matches-write',
    'n-finalize'
  ]);
  for (const id of ['n-expand', 'n-distance', 'n-reduce-distance', 'n-reduce-feature']) {
    expect(nodes.find(node => node.id === id)?.condition).toMatchObject({
      source: 'gpu',
      mode: 'indirect'
    });
  }
  device.destroy();
});

it('GPUSpatialPredicateJoin indirectly dispatches candidate and cooperative refinement', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const createProps = (
    id: string,
    overrides: Partial<GPUSpatialPredicateJoinProps> = {}
  ): GPUSpatialPredicateJoinProps => ({
    id,
    left: {kind: 'points', positions: view(`${id}-left`, 'float32x2', 4)},
    right: {kind: 'points', positions: view(`${id}-right`, 'float32x2', 3)},
    predicate: 'intersects',
    candidateCapacity: 4096,
    pairs: {
      leftIds: view(`${id}-left-ids`, 'uint32', 4096),
      rightIds: view(`${id}-right-ids`, 'uint32', 4096)
    },
    ...overrides
  });
  const fastNodes = new GPUSpatialPredicateJoin(createProps('p')).getCommandNodes(graph);
  for (const id of ['p-exact', 'p-scatter']) {
    expect(fastNodes.find(node => node.id === id)?.condition).toMatchObject({
      source: 'gpu',
      mode: 'indirect'
    });
  }
  expect(fastNodes.map(node => node.id)).toContain('p-match-total');

  const cooperativeNodes = new GPUSpatialPredicateJoin(
    createProps('w', {
      predicate: 'dwithin',
      distance: view('w-distance', 'float32', 1),
      engine: 'relate'
    })
  ).getCommandNodes(graph);
  expect(cooperativeNodes.map(node => node.id)).toContain('w-candidate-dispatch-64');
  expect(cooperativeNodes.find(node => node.id === 'w-exact')?.condition).toMatchObject({
    source: 'gpu',
    mode: 'indirect'
  });
  device.destroy();
});

it('GPUNearestFeatureJoin evaluates neighbor foot points with one invocation per query', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const queryCount = 3;
  const neighborCapacity = 64;
  const slotCount = queryCount * neighborCapacity;
  const nodes = new GPUNearestFeatureJoin({
    id: 'knn',
    points: createTransientView(graph, 'knn-points', 'float32x2', queryCount),
    features: {
      kind: 'segments',
      starts: createTransientView(graph, 'knn-starts', 'float32x2', 2),
      ends: createTransientView(graph, 'knn-ends', 'float32x2', 2)
    },
    k: 1,
    ties: 'all',
    neighborCapacity,
    neighborIds: createTransientView(graph, 'knn-ids', 'uint32', slotCount),
    neighborCounts: createTransientView(graph, 'knn-counts', 'uint32', queryCount),
    neighborDistances: createTransientView(graph, 'knn-distances', 'float32', slotCount),
    neighborFootPoints: createTransientView(graph, 'knn-foot', 'float32x2', slotCount),
    overflow: createTransientView(graph, 'knn-overflow', 'uint32', 1)
  }).getCommandNodes(graph);
  expect(nodes.find(node => node.id === 'knn-foot-points')?.workload?.maximumInvocationCount).toBe(
    queryCount
  );
  device.destroy();
});

it('spatial joins validate props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = createPolygonProps(graph);
  const create = (overrides: Partial<GPUPointInPolygonJoinProps>) =>
    new GPUPointInPolygonJoin({...base, ...overrides});
  expect(() => create({leafCapacity: 3})).toThrow(/leafCapacity/);
  expect(() => create({candidateCapacity: 0})).toThrow(/candidateCapacity/);
  expect(() => create({candidateCapacity: 1.5})).toThrow(/candidateCapacity/);
  expect(() =>
    create({pointFeatureIds: createTransientView(graph, 'short-ids', 'uint32', 9)})
  ).toThrow(/chunk topology/);
  expect(() =>
    create({featureCounts: createTransientView(graph, 'short-counts', 'uint32', 3)})
  ).toThrow(/feature count/);
  expect(() =>
    create({featureIds: createTransientView(graph, 'short-feature-ids', 'uint32', 5)})
  ).toThrow(/feature count/);
  expect(() =>
    create({points: createTransientView(graph, 'points3', 'float32x3', 10) as never})
  ).toThrow(/points/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      sourceIds: graph.createDataView(shared, {format: 'uint32', length: 10}),
      pointFeatureIds: graph.createDataView(shared, {format: 'uint32', length: 10})
    })
  ).toThrow(/overlap/);

  const nearestGraph = new GPUCommandGraph(device);
  const nearestBase = createNearestProps(nearestGraph);
  expect(
    () =>
      new GPUNearestFeatureJoin({
        ...nearestBase,
        radius: createTransientView(nearestGraph, 'radius2', 'float32', 2)
      })
  ).toThrow(/radius/);
  expect(
    () =>
      new GPUNearestFeatureJoin({
        ...nearestBase,
        features: {
          kind: 'segments',
          starts: createTransientView(nearestGraph, 's3', 'float32x2', 3),
          ends: createTransientView(nearestGraph, 'e2', 'float32x2', 2)
        }
      })
  ).toThrow(/equal lengths/);

  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUPointInPolygonJoin(createPolygonProps(otherGraph)).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('spatialSort defaults to false and only adds nodes when enabled', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const plain = new GPUPointInPolygonJoin({...createPolygonProps(graph), id: 'j'});
  expect(plain.spatialSort).toBe(false);
  const plainIds = plain.getCommandNodes(graph).map(node => node.id);
  expect(plainIds.some(id => id.includes('-sort-'))).toBe(false);

  const sorted = new GPUPointInPolygonJoin({
    ...createPolygonProps(graph, 'b-'),
    id: 'k',
    spatialSort: true
  });
  expect(sorted.spatialSort).toBe(true);
  const sortedIds = sorted.getCommandNodes(graph).map(node => node.id);
  const order = ['k-bounds', 'k-sort-scene-bounds', 'k-sort-keys', 'k-sort-gather', 'k-bvh-'];
  const positions = order.map(prefix => sortedIds.findIndex(id => id.startsWith(prefix)));
  expect(positions.every(position => position >= 0)).toBe(true);
  expect([...positions].sort((left, right) => left - right)).toEqual(positions);
  expect(positions[4]).toBeLessThan(sortedIds.indexOf('k-clear'));
  expect(sortedIds.some(id => id.startsWith('k-sort-order'))).toBe(true);
  // Probe and everything after it are unchanged.
  expect(sortedIds.filter(id => !id.startsWith('k-bvh-') && !id.includes('-sort-'))).toEqual(
    plainIds.filter(id => !id.startsWith('j-bvh-')).map(id => id.replace('j-', 'k-'))
  );

  const nearestGraph = new GPUCommandGraph(device);
  const nearest = new GPUNearestFeatureJoin({
    ...createNearestProps(nearestGraph),
    spatialSort: true
  });
  expect(nearest.spatialSort).toBe(true);
  const defaultGraph = new GPUCommandGraph(device);
  expect(new GPUNearestFeatureJoin(createNearestProps(defaultGraph)).spatialSort).toBe(false);
  device.destroy();
});

it('spatialSort skips sorting for fewer than two features', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const join = new GPUNearestFeatureJoin({
    ...createNearestProps(graph),
    id: 'one',
    features: {
      kind: 'segments',
      starts: createTransientView(graph, 'one-starts', 'float32x2', 1),
      ends: createTransientView(graph, 'one-ends', 'float32x2', 1)
    },
    spatialSort: true
  });
  expect(join.getCommandNodes(graph).some(node => node.id.includes('-sort-'))).toBe(false);
  device.destroy();
});

it('traversal model: Morton order cuts visited nodes and keeps the candidate set', () => {
  const columns = 32;
  const bounds: FeatureBounds[] = [];
  for (let row = 0; row < columns; row++) {
    for (let column = 0; column < columns; column++) {
      bounds.push([column, row, column + 0.8, row + 0.8]);
    }
  }
  // Deterministic shuffle.
  const shuffled = bounds.map((_, index) => index);
  let state = 12345;
  for (let index = shuffled.length - 1; index > 0; index--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const swap = state % (index + 1);
    [shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]];
  }
  const shuffledBounds = shuffled.map(row => bounds[row]);
  const points: [number, number][] = Array.from({length: 500}, (_, index) => [
    ((index * 37) % 320) / 10,
    ((index * 91) % 320) / 10
  ]);
  const capacity = getLeafCapacity(shuffledBounds.length);
  const identity = shuffledBounds.map((_, row) => row);
  const unsorted = measureTraversal(buildModelBVH(shuffledBounds, identity, capacity), points, 0);
  const order = getMortonLeafOrder(shuffledBounds);
  expect([...order].sort((left, right) => left - right)).toEqual(identity);
  const sorted = measureTraversal(buildModelBVH(shuffledBounds, order, capacity), points, 0);
  expect(sorted.candidates).toBe(unsorted.candidates);
  expect(sorted.visitedPerPoint).toBeLessThan(unsorted.visitedPerPoint / 4);

  // Empty features sort last and never become candidates.
  const withEmpty: FeatureBounds[] = [
    [5, 5, 6, 6],
    [Infinity, Infinity, -Infinity, -Infinity],
    [0, 0, 1, 1]
  ];
  expect(getMortonLeafOrder(withEmpty)).toEqual([2, 0, 1]);
  const model = buildModelBVH(withEmpty, [2, 0, 1], 4);
  expect(simulateTraversal(model, [0.5, 0.5], [0.5, 0.5]).candidates).toBe(1);
  expect(simulateTraversal(model, [100, 100], [100, 100]).candidates).toBe(0);
});
