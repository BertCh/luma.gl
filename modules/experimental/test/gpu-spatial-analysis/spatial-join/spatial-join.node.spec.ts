// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin,
  GPUSpatialJoinCandidates,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  type GPUNearestFeatureJoinProps,
  type GPUPointInPolygonJoinProps,
  type GPUSpatialPredicateJoinProps
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {HALF_OPEN_SCENES} from './half-open-fixtures';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../../src/analysis-core/contracts/earth';
import {
  classifyPointHalfOpen,
  createRandom,
  joinPointsInPolygons,
  type OraclePolygonFeature
} from './spatial-join-oracle';
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
    points: {
      kind: 'points',
      positions: view('points', 'float32x2', 10),
      sourceIds: view('source-ids', 'uint32', 10)
    },
    polygons: {
      kind: 'polygons',
      positions: view('polygon-positions', 'float32x2', 20),
      featureOffsets: view('feature-offsets', 'uint32', 5),
      polygonOffsets: view('polygon-offsets', 'uint32', 5),
      ringOffsets: view('ring-offsets', 'uint32', 6),
      sourceIds: view('feature-ids', 'uint32', 4)
    },
    candidateCapacity: 16,
    output: {
      pointFeatureIds: view('point-feature-ids', 'uint32', 10),
      featureCounts: view('feature-counts', 'uint32', 4),
      matches: {
        ids: view('match-ids', 'uint32', 10),
        status: {
          count: view('match-count', 'uint32', 1),
          overflow: view('match-overflow', 'uint32', 1)
        }
      },
      status: {
        overflow: view('overflow', 'uint32', 1),
        candidateCount: view('candidate-count', 'uint32', 1),
        uncertainCount: view('uncertain-count', 'uint32', 1)
      }
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
    queries: {kind: 'points', positions: createTransientView(graph, 'points', 'float32x2', 6)},
    features: {
      kind: 'segments',
      starts: createTransientView(graph, 'starts', 'float32x2', 3),
      ends: createTransientView(graph, 'ends', 'float32x2', 3)
    },
    radius: createTransientView(graph, 'radius', 'float32', 1),
    candidateCapacity: 16,
    output: {
      nearestFeatureIds: createTransientView(graph, 'nearest', 'uint32', 6),
      nearestDistances: createTransientView(graph, 'distances', 'float32', 6),
      matches: {
        ids: createTransientView(graph, 'match-ids', 'uint32', 6),
        status: {
          count: createTransientView(graph, 'match-count', 'uint32', 1),
          overflow: createTransientView(graph, 'match-overflow', 'uint32', 1)
        }
      },
      status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
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
    'j-probe-carry',
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
  const base = createPolygonProps(graph);
  const ids = new GPUPointInPolygonJoin({
    ...base,
    id: 'j',
    points: {kind: 'points', positions: points},
    output: {...base.output, pointFeatureIds, matches: undefined}
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
    'n-probe-carry',
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
    output: {
      pairs: {
        leftIds: view(`${id}-left-ids`, 'uint32', 4096),
        rightIds: view(`${id}-right-ids`, 'uint32', 4096),
        status: {
          count: view(`${id}-count`, 'uint32', 1),
          overflow: view(`${id}-pair-overflow`, 'uint32', 1)
        }
      }
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
    queries: {
      kind: 'points',
      positions: createTransientView(graph, 'knn-points', 'float32x2', queryCount)
    },
    features: {
      kind: 'segments',
      starts: createTransientView(graph, 'knn-starts', 'float32x2', 2),
      ends: createTransientView(graph, 'knn-ends', 'float32x2', 2)
    },
    k: 1,
    ties: 'all',
    neighborCapacity,
    output: {
      neighborIds: createTransientView(graph, 'knn-ids', 'uint32', slotCount),
      neighborCounts: createTransientView(graph, 'knn-counts', 'uint32', queryCount),
      neighborDistances: createTransientView(graph, 'knn-distances', 'float32', slotCount),
      neighborFootPoints: createTransientView(graph, 'knn-foot', 'float32x2', slotCount),
      status: {overflow: createTransientView(graph, 'knn-overflow', 'uint32', 1)}
    }
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
  const createOutput = (output: Partial<GPUPointInPolygonJoinProps['output']>) =>
    create({output: {...base.output, ...output}});
  expect(() => create({leafCapacity: 3})).toThrow(/leafCapacity/);
  expect(() => create({candidateCapacity: 0})).toThrow(/candidateCapacity/);
  expect(() => create({candidateCapacity: 1.5})).toThrow(/candidateCapacity/);
  expect(() =>
    createOutput({pointFeatureIds: createTransientView(graph, 'short-ids', 'uint32', 9)})
  ).toThrow(/chunk topology/);
  expect(() =>
    createOutput({featureCounts: createTransientView(graph, 'short-counts', 'uint32', 3)})
  ).toThrow(/feature/);
  expect(() =>
    create({
      polygons: {
        ...base.polygons,
        sourceIds: createTransientView(graph, 'short-feature-ids', 'uint32', 5)
      }
    })
  ).toThrow(/feature/);
  expect(() =>
    create({
      points: {
        kind: 'points',
        positions: createTransientView(graph, 'points3', 'float32x3', 10) as never
      }
    })
  ).toThrow(/points/);
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 64, usage: 128});
  expect(() =>
    create({
      points: {
        kind: 'points',
        positions: base.points.positions,
        sourceIds: graph.createDataView(shared, {format: 'uint32', length: 10})
      },
      output: {
        ...base.output,
        pointFeatureIds: graph.createDataView(shared, {format: 'uint32', length: 10})
      }
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
  ).toThrow(/starts and ends lengths differ/);

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

let serial = 0;
const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
const DEGREES = {coordinateSpace: 'longitude-latitude', metric: 'none'} as const;

/** Every permutation of `[0, count)`. */
function getPermutations(count: number): number[][] {
  if (count <= 1) {
    return [Array.from({length: count}, (_, row) => row)];
  }
  return getPermutations(count - 1).flatMap(permutation =>
    Array.from({length: count}, (_, slot) => [
      ...permutation.slice(0, slot),
      count - 1,
      ...permutation.slice(slot)
    ])
  );
}

for (const [name, scene] of Object.entries(HALF_OPEN_SCENES)) {
  it(`half-open oracle matches Shapely 2.2.0 and ignores feature row order (${name})`, () => {
    // Owners pinned from shapely.Polygon.contains(Point(x + 1e-9, y + 1e-6)), see half-open-fixtures.ts.
    for (const permutation of getPermutations(scene.features.length)) {
      const features: OraclePolygonFeature[] = permutation.map(row => scene.features[row]);
      const {featureRows} = joinPointsInPolygons(scene.points, features, 'half-open');
      const owners = featureRows.map(row => (row === 0xffffffff ? null : permutation[row]));
      expect(owners, `${name} order ${permutation}`).toEqual(scene.owners);
    }
  });
}

it('half-open rule partitions a coverage: at most one owner per point, exactly one inside the lower-left closure', () => {
  const {features} = HALF_OPEN_SCENES.squares;
  for (let x = -1; x <= 5; x += 0.5) {
    for (let y = -1; y <= 5; y += 0.5) {
      const owners = features.filter(feature => {
        const classification = classifyPointHalfOpen([x, y], feature);
        return classification === 'inside' || classification === 'owned-boundary';
      });
      const inCoverage = x >= 0 && x < 4 && y >= 0 && y < 4;
      expect(owners.length, `(${x}, ${y})`).toBe(inCoverage ? 1 : 0);
    }
  }
  // The closed rule keeps the smallest row, so the owner of a shared edge follows the feature order.
  const closed = joinPointsInPolygons([[2, 1]], features, 'closed').featureRows[0];
  const reordered = joinPointsInPolygons(
    [[2, 1]],
    [features[1], features[0], features[2], features[3]],
    'closed'
  ).featureRows[0];
  expect([closed, [1, 0, 2, 3][reordered]]).toEqual([0, 1]);
});

it('GPUPointInPolygonJoin resolves boundaryRule', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const create = (overrides: Partial<GPUPointInPolygonJoinProps>, prefix: string) =>
    new GPUPointInPolygonJoin({...createPolygonProps(graph, prefix), ...overrides});
  expect(create({}, 'a-').boundaryRule).toBe('half-open');
  expect(create({boundaryRule: 'closed'}, 'c-').boundaryRule).toBe('closed');
  expect(create({boundaryRule: 'open'}, 'd-').boundaryRule).toBe('open');
  expect(() => create({boundaryRule: 'inclusive' as never}, 'i-')).toThrow(/boundaryRule/);
  device.destroy();
});

it('planar spatial joins resolve a planar context and reject longitude-latitude', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const pip = new GPUPointInPolygonJoin(createPolygonProps(graph, 'ok-'));
  expect(pip.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  expect(
    () =>
      new GPUPointInPolygonJoin(createPolygonProps(graph, 'bad-', {spatialContext: GREAT_CIRCLE}))
  ).toThrow(/needs metric 'none' on longitude-latitude/);
  // Containment is topological (edges straight in degrees, as GeoJSON): metric 'none' is planar.
  const degrees = new GPUPointInPolygonJoin(
    createPolygonProps(graph, 'deg-', {spatialContext: DEGREES})
  );
  expect(degrees.spatialContext).toEqual(DEGREES);
  const degreeGraph = new GPUCommandGraph(device);
  const planarGraph = new GPUCommandGraph(device);
  expect(
    new GPUPointInPolygonJoin(
      createPolygonProps(degreeGraph, 'd-', {spatialContext: DEGREES})
    ).getCommandNodes(degreeGraph).length
  ).toBe(
    new GPUPointInPolygonJoin(createPolygonProps(planarGraph, 'p-')).getCommandNodes(planarGraph)
      .length
  );
  expect(
    () =>
      new GPUPointInPolygonJoin(
        createPolygonProps(graph, 'deg-gc-', {
          spatialContext: {...GREAT_CIRCLE, units: 'meters'}
        })
      )
  ).toThrow(/needs metric 'none' on longitude-latitude/);

  expect(
    new GPUNearestFeatureJoin(createNearestProps(new GPUCommandGraph(device))).spatialContext
      .coordinateSpace
  ).toBe('planar');
  // The nearest-feature join also accepts great-circle distances in meters; see the context
  // matrix test below.

  const candidateGraph = new GPUCommandGraph(device);
  const candidateProps = (spatialContext?: typeof GREAT_CIRCLE) => ({
    left: {
      kind: 'points' as const,
      positions: createTransientView(candidateGraph, `cl-${serial++}`, 'float32x2', 3)
    },
    right: {
      kind: 'points' as const,
      positions: createTransientView(candidateGraph, `cr-${serial++}`, 'float32x2', 3)
    },
    output: {
      leftIds: createTransientView(candidateGraph, `ci-${serial++}`, 'uint32', 8),
      rightIds: createTransientView(candidateGraph, `cj-${serial++}`, 'uint32', 8),
      status: {
        count: createTransientView(candidateGraph, `cc-${serial++}`, 'uint32', 1),
        overflow: createTransientView(candidateGraph, `co-${serial++}`, 'uint32', 1)
      }
    },
    spatialContext
  });
  expect(new GPUSpatialJoinCandidates(candidateProps()).spatialContext.coordinateSpace).toBe(
    'planar'
  );
  expect(() => new GPUSpatialJoinCandidates(candidateProps(GREAT_CIRCLE))).toThrow(
    /project positions to a planar frame/
  );
  device.destroy();
});

it('GPUSpatialPredicateJoin accepts great-circle dwithin and planar-in-degrees topology only', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const create = (overrides: Partial<GPUSpatialPredicateJoinProps>) => {
    const id = `ctx-${serial++}`;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    return new GPUSpatialPredicateJoin({
      id,
      left: {kind: 'points', positions: view('left', 'float32x2', 4)},
      right: {kind: 'points', positions: view('right', 'float32x2', 3)},
      predicate: 'intersects',
      candidateCapacity: 64,
      output: {
        pairs: {
          leftIds: view('left-ids', 'uint32', 64),
          rightIds: view('right-ids', 'uint32', 64),
          status: {
            count: view('count', 'uint32', 1),
            overflow: view('pair-overflow', 'uint32', 1)
          }
        }
      },
      ...overrides
    });
  };
  const lambert = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'local-frame', kind: 'local-lambert-equal-area'}
  } as const;
  expect(() => create({predicate: 'dwithin', distance: 1, spatialContext: lambert})).toThrow(
    /measures distance/
  );
  expect(() => create({predicate: 'intersects', spatialContext: lambert})).not.toThrow();
  expect(() =>
    create({
      predicate: 'dwithin',
      distance: 1,
      spatialContext: {...lambert, frame: {...lambert.frame, kind: 'local-azimuthal-equidistant'}}
    })
  ).not.toThrow();
  // Default: planar, no radius.
  expect(create({}).spatialContext.coordinateSpace).toBe('planar');
  expect(create({}).sphereRadius).toBeUndefined();
  // Great-circle dwithin, meters on the mean Earth sphere unless the context sets a radius.
  const geodesic = create({predicate: 'dwithin', distance: 1500, spatialContext: GREAT_CIRCLE});
  expect(geodesic.sphereRadius).toBe(6371008.8);
  expect(geodesic.usesWorkgroupDistance).toBe(false);
  expect(
    create({
      predicate: 'dwithin',
      distance: 1500,
      spatialContext: {...GREAT_CIRCLE, sphereRadius: 6378137}
    }).sphereRadius
  ).toBe(6378137);
  // `auto` never selects the workgroup kernel for the great-circle metric, even for large features.
  expect(
    create({
      predicate: 'dwithin',
      distance: 1500,
      spatialContext: GREAT_CIRCLE,
      left: {
        kind: 'polygons',
        positions: createTransientView(graph, `big-${serial++}`, 'float32x2', 4000),
        featureOffsets: createTransientView(graph, `big-${serial++}`, 'uint32', 2),
        polygonOffsets: createTransientView(graph, `big-${serial++}`, 'uint32', 2),
        ringOffsets: createTransientView(graph, `big-${serial++}`, 'uint32', 2)
      },
      right: {
        kind: 'polygons',
        positions: createTransientView(graph, `big-${serial++}`, 'float32x2', 4000),
        featureOffsets: createTransientView(graph, `big-${serial++}`, 'uint32', 2),
        polygonOffsets: createTransientView(graph, `big-${serial++}`, 'uint32', 2),
        ringOffsets: createTransientView(graph, `big-${serial++}`, 'uint32', 2)
      }
    }).usesWorkgroupDistance
  ).toBe(false);
  // Topological predicates take degrees with no metric.
  expect(create({predicate: 'touches', spatialContext: DEGREES}).spatialContext.metric).toBe(
    'none'
  );
  expect(() => create({predicate: 'intersects', spatialContext: GREAT_CIRCLE})).toThrow(
    /needs metric 'none' on longitude-latitude/
  );
  expect(() => create({predicate: 'dwithin', distance: 5, spatialContext: DEGREES})).toThrow(
    /great-circle/
  );
  expect(() =>
    create({
      predicate: 'dwithin',
      distance: 5,
      spatialContext: {coordinateSpace: 'planar', metric: 'great-circle'}
    })
  ).toThrow(/incompatible/);
  expect(() =>
    create({
      predicate: 'dwithin',
      distance: 5,
      spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}
    })
  ).toThrow(/meters/);
  expect(() =>
    create({predicate: 'dwithin', distance: 5, engine: 'relate', spatialContext: GREAT_CIRCLE})
  ).toThrow(/not available for great-circle/);
  // About 8000 km is the largest great-circle distance.
  expect(() => create({predicate: 'dwithin', distance: 9e6, spatialContext: GREAT_CIRCLE})).toThrow(
    /distance exceeds 8005809 m/
  );
  expect(
    create({predicate: 'dwithin', distance: 7.9e6, spatialContext: GREAT_CIRCLE}).sphereRadius
  ).toBe(6371008.8);
  expect(() =>
    create({
      predicate: 'dwithin',
      distance: 5,
      spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
    })
  ).toThrow(/does not support discrete-grid/);
  device.destroy();
});

it('GPUNearestFeatureJoin resolves its spatial context matrix', () => {
  const device = createNullWebGPUDevice();
  const create = (overrides: Partial<GPUNearestFeatureJoinProps>) =>
    new GPUNearestFeatureJoin(createNearestProps(new GPUCommandGraph(device), overrides));
  expect(create({}).sphereRadius).toBeUndefined();
  const earth = create({spatialContext: GREAT_CIRCLE});
  expect(earth.sphereRadius).toBe(GPU_EARTH_MEAN_RADIUS_METERS);
  expect(earth.spatialContext.coordinateSpace).toBe('longitude-latitude');
  expect(create({spatialContext: {...GREAT_CIRCLE, sphereRadius: 1737400}}).sphereRadius).toBe(
    1737400
  );
  expect(create({spatialContext: {...GREAT_CIRCLE, units: 'meters'}}).sphereRadius).toBe(
    GPU_EARTH_MEAN_RADIUS_METERS
  );
  // Distances honor kilometers; degrees need a metric; planar contexts cannot be great-circle.
  expect(create({spatialContext: {...GREAT_CIRCLE, units: 'kilometers'}}).sphereRadius).toBe(
    GPU_EARTH_MEAN_RADIUS_METERS / 1000
  );
  expect(() => create({spatialContext: DEGREES})).toThrow(/great-circle/);
  expect(() =>
    create({spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb'}})
  ).toThrow(/great-circle metric/);
  expect(() =>
    create({spatialContext: {coordinateSpace: 'planar', metric: 'great-circle'}})
  ).toThrow(/incompatible with planar/);
  // candidateCapacity is unused on the great-circle path, and required on the planar one.
  expect(() => create({spatialContext: GREAT_CIRCLE, candidateCapacity: undefined})).not.toThrow();
  expect(() => create({candidateCapacity: undefined})).toThrow(/candidateCapacity/);
  device.destroy();
});

it('great-circle nearest-feature mode schedules the k = 1 traversal and the nearest outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUNearestFeatureJoin(createNearestProps(graph, {spatialContext: GREAT_CIRCLE}))
    .getCommandNodes(graph)
    .map(node => node.id);
  const own = ids.filter(id => !id.startsWith('n-bvh-'));
  expect(own).toContain('n-traverse');
  expect(own).toContain('n-assign');
  expect(own).toContain('n-collect-matches');
  expect(own[own.length - 1]).toBe('n-finalize');
  expect(own).not.toContain('n-probe');
  expect(own.indexOf('n-clear-state')).toBeLessThan(own.indexOf('n-traverse'));
  expect(own.indexOf('n-traverse')).toBeLessThan(own.indexOf('n-assign'));
  device.destroy();
});

/**
 * JS port of the great-circle lower bound `boxLowerAngle` (nearest-wgsl.ts) in f64, checked
 * against the haversine of random point pairs: the bound must never exceed the true distance.
 */
function boxLowerAngle(
  node: [number, number, number, number],
  query: [number, number, number, number]
): number {
  const radians = Math.PI / 180;
  const latitudeGap = Math.max(node[1] - query[3], query[1] - node[3], 0);
  let longitudeGap = 0;
  if (node[2] - node[0] < 180 && query[2] - query[0] < 180) {
    const direct = Math.max(node[0] - query[2], query[0] - node[2]);
    const around = 360 - (Math.max(node[2], query[2]) - Math.min(node[0], query[0]));
    longitudeGap = Math.max(Math.min(direct, around), 0);
  }
  const farLatitude = Math.max(
    Math.abs(node[1]),
    Math.abs(node[3]),
    Math.abs(query[1]),
    Math.abs(query[3])
  );
  const cosine = Math.cos(farLatitude * radians);
  const term =
    Math.sin((latitudeGap * radians) / 2) ** 2 +
    cosine ** 2 * Math.sin((longitudeGap * radians) / 2) ** 2;
  return 2 * Math.asin(Math.sqrt(Math.min(term, 1)));
}

function haversineAngle(a: [number, number], b: [number, number]): number {
  const radians = Math.PI / 180;
  const term =
    Math.sin(((b[1] - a[1]) * radians) / 2) ** 2 +
    Math.cos(a[1] * radians) *
      Math.cos(b[1] * radians) *
      Math.sin(((b[0] - a[0]) * radians) / 2) ** 2;
  return 2 * Math.asin(Math.sqrt(Math.min(term, 1)));
}

it('the great-circle box lower bound never exceeds the distance of any point pair', () => {
  const random = createRandom(7);
  const randomBox = (): [number, number, number, number] => {
    const longitude = random() * 360 - 180;
    const latitude = random() * 180 - 90;
    const width = random() * 60;
    const height = random() * 30;
    return [longitude, latitude, Math.min(longitude + width, 180), Math.min(latitude + height, 90)];
  };
  const pointIn = (box: [number, number, number, number]): [number, number] => [
    box[0] + random() * (box[2] - box[0]),
    box[1] + random() * (box[3] - box[1])
  ];
  for (let trial = 0; trial < 4000; trial++) {
    const node = randomBox();
    const query = randomBox();
    const bound = boxLowerAngle(node, query);
    for (let sample = 0; sample < 8; sample++) {
      expect(bound).toBeLessThanOrEqual(haversineAngle(pointIn(node), pointIn(query)) + 1e-12);
    }
  }
  // Across the antimeridian the circular gap is used.
  const across = boxLowerAngle([179, 0, 179.5, 0.1], [-179.5, 0, -179, 0.1]);
  expect(across).toBeCloseTo(haversineAngle([179.5, 0], [-179.5, 0]), 6);
  // A box straddling the antimeridian (span of 180 degrees or more) never prunes in longitude.
  expect(boxLowerAngle([-179.9, 10, 179.9, 11], [180, 10, 180, 10.5])).toBe(0);
});

it('GPUSpatialJoinPrepared validates before allocating and forgets freed storage on destroy', () => {
  // E-21: a rejected getCommandNodes used to allocate storage and latch the graph first, and
  // destroy() left the handle pointing at freed buffers.
  const device = createNullWebGPUDevice();
  const owner = new GPUCommandGraph(device, {id: 'owner'});
  const other = new GPUCommandGraph(device, {id: 'other'});
  const geometry = {
    kind: 'polygons' as const,
    positions: createTransientView(owner, 'positions', 'float32x2', 8),
    featureOffsets: createTransientView(owner, 'feature-offsets', 'uint32', 3),
    polygonOffsets: createTransientView(owner, 'polygon-offsets', 'uint32', 3),
    ringOffsets: createTransientView(owner, 'ring-offsets', 'uint32', 3)
  };
  const prepared = new GPUSpatialJoinPrepared({id: 'p', geometry});
  expect(() => prepared.getCommandNodes(other)).toThrow();
  expect(prepared.isDeclaredIn(other)).toBe(false);
  expect(() => prepared.storage).toThrow();
  expect(prepared.getCommandNodes(owner).length).toBeGreaterThan(0);
  expect(prepared.isDeclaredIn(owner)).toBe(true);
  expect(prepared.storage.leafIds.length).toBeGreaterThan(0);
  prepared.destroy();
  expect(prepared.isDeclaredIn(owner)).toBe(false);
  expect(() => prepared.storage).toThrow();
  device.destroy();
});

it('GPUNearestFeatureJoin stays within 8 storage buffers with every filter enabled', () => {
  // A-2: exclusive with queryIds and featureIds plus onAttribute bound 9-10 storage buffers in the
  // reductions; the filters now run in their own pass. Kernel declaration throws above the null
  // device's limit of 8, so building the nodes is the assertion.
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const join = new GPUNearestFeatureJoin(
    createNearestProps(graph, {
      exclusive: true,
      queryIds: createTransientView(graph, 'query-ids', 'uint32', 6),
      features: {
        kind: 'segments',
        starts: createTransientView(graph, 'filter-starts', 'float32x2', 3),
        ends: createTransientView(graph, 'filter-ends', 'float32x2', 3),
        sourceIds: createTransientView(graph, 'feature-ids', 'uint32', 3)
      },
      onAttribute: {
        left: createTransientView(graph, 'query-keys', 'uint32', 6),
        right: createTransientView(graph, 'feature-keys', 'uint32', 3)
      },
      output: {
        nearestFeatureIds: createTransientView(graph, 'filter-nearest', 'uint32', 6),
        nearestDistances: createTransientView(graph, 'filter-distances', 'float32', 6),
        featureCounts: createTransientView(graph, 'feature-counts', 'uint32', 3),
        matches: {
          ids: createTransientView(graph, 'filter-match-ids', 'uint32', 6),
          status: {
            count: createTransientView(graph, 'filter-match-count', 'uint32', 1),
            overflow: createTransientView(graph, 'filter-match-overflow', 'uint32', 1)
          }
        },
        status: {
          overflow: createTransientView(graph, 'filter-overflow', 'uint32', 1),
          candidateCount: createTransientView(graph, 'candidate-count', 'uint32', 1)
        }
      }
    })
  );
  const ids = join.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('n-filter');
  expect(ids.indexOf('n-filter')).toBeLessThan(ids.indexOf('n-reduce-distance'));
  device.destroy();
});

it('GPUSpatialPredicateJoin rejects capacities whose cooperative lanes overflow u32', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const left = createTransientView(graph, 'c-left', 'float32x2', 4);
  const right = createTransientView(graph, 'c-right', 'float32x2', 3);
  const output = {
    pairs: {
      leftIds: createTransientView(graph, 'c-left-ids', 'uint32', 16),
      rightIds: createTransientView(graph, 'c-right-ids', 'uint32', 16),
      status: {
        count: createTransientView(graph, 'c-count', 'uint32', 1),
        overflow: createTransientView(graph, 'c-pair-overflow', 'uint32', 1)
      }
    }
  };
  const createJoin = (overrides: Partial<GPUSpatialPredicateJoinProps>) =>
    new GPUSpatialPredicateJoin({
      id: 'c',
      left: {kind: 'points', positions: left},
      right: {kind: 'points', positions: right},
      predicate: 'intersects',
      candidateCapacity: 4096,
      output,
      ...overrides
    });
  // The fast engine has one lane per candidate, so a very large capacity is still accepted.
  expect(() => createJoin({candidateCapacity: 2 ** 30})).not.toThrow();
  // Cooperative dwithin launches 64 lanes per candidate, the relate engine 256.
  const distance = createTransientView(graph, 'c-distance', 'float32', 1);
  const dwithin = {predicate: 'dwithin', distance, engine: 'relate'} as const;
  expect(() => createJoin({...dwithin, candidateCapacity: 2 ** 26 - 1})).not.toThrow();
  expect(() => createJoin({...dwithin, candidateCapacity: 2 ** 26})).toThrow(
    /candidateCapacity exceeds 67108863/
  );
  expect(() =>
    createJoin({predicate: 'relate', pattern: 'T********', candidateCapacity: 2 ** 24})
  ).toThrow(/candidateCapacity exceeds 16777215/);
  device.destroy();
});

it('distance measurements reject Lambert frames and accept AEQD meters', () => {
  const device = createNullWebGPUDevice();
  const lambert = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'local-frame', kind: 'local-lambert-equal-area'}
  } as const;
  const create = (spatialContext: GPUSpatialContext) =>
    new GPUNearestFeatureJoin(createNearestProps(new GPUCommandGraph(device), {spatialContext}));
  expect(() => create(lambert)).toThrow(/measures distance/);
  expect(() =>
    create({...lambert, frame: {...lambert.frame, kind: 'local-azimuthal-equidistant'}})
  ).not.toThrow();
  device.destroy();
});
