// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE as N,
  GPUPointInPolygonJoin,
  type GPUPointInPolygonJoinProps
} from '../../../src/geospatial/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readCompactIds,
  readUint32,
  sortNumbers,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildPolygonFeatureArrays,
  classifyPointInFeature,
  createRandom,
  joinPointsInPolygons,
  POLYGON_FEATURES,
  POLYGON_POINTS
} from './spatial-join-oracle';

type JoinBufferName =
  | 'points'
  | 'sourceIds'
  | 'polygonPositions'
  | 'featureOffsets'
  | 'polygonOffsets'
  | 'ringOffsets'
  | 'featureIds'
  | 'pointFeatureIds'
  | 'featureCounts'
  | 'overflow'
  | 'candidateCount'
  | 'uncertainCount'
  | 'matchIds'
  | 'matchCount'
  | 'matchOverflow'
  | 'matchTotal';

type JoinFixture = {
  graph: GPUCommandGraph;
  buffers: Record<JoinBufferName, Buffer>;
  props: GPUPointInPolygonJoinProps;
};

function createJoinFixture(
  device: Device,
  points: [number, number][],
  options: Partial<GPUPointInPolygonJoinProps> & {matchCapacity?: number} = {}
): JoinFixture {
  const {matchCapacity, ...overrides} = options;
  const graph = new GPUCommandGraph(device, {id: 'pip-join'});
  const arrays = buildPolygonFeatureArrays(POLYGON_FEATURES);
  const pointCount = points.length;
  const featureCount = POLYGON_FEATURES.length;
  const buffers: Record<JoinBufferName, Buffer> = {
    points: createInputBuffer(device, Float32Array.from(points.flat())),
    sourceIds: createInputBuffer(device, Uint32Array.from(points.map((_, index) => 1000 + index))),
    polygonPositions: createInputBuffer(device, arrays.polygonPositions),
    featureOffsets: createInputBuffer(device, arrays.featureOffsets),
    polygonOffsets: createInputBuffer(device, arrays.polygonOffsets),
    ringOffsets: createInputBuffer(device, arrays.ringOffsets),
    featureIds: createInputBuffer(device, Uint32Array.from([100, 101, 102, 103])),
    pointFeatureIds: createOutputBuffer(device, pointCount),
    featureCounts: createOutputBuffer(device, featureCount),
    overflow: createOutputBuffer(device, 1),
    candidateCount: createOutputBuffer(device, 1),
    uncertainCount: createOutputBuffer(device, 1),
    matchIds: createOutputBuffer(device, matchCapacity ?? pointCount),
    matchCount: createOutputBuffer(device, 1),
    matchOverflow: createOutputBuffer(device, 1),
    matchTotal: createOutputBuffer(device, 1)
  };
  const props: GPUPointInPolygonJoinProps = {
    points: importGraphBuffer(graph, 'points', buffers.points, 'float32x2', pointCount),
    sourceIds: importGraphBuffer(graph, 'source-ids', buffers.sourceIds, 'uint32', pointCount),
    polygonPositions: importGraphBuffer(
      graph,
      'polygon-positions',
      buffers.polygonPositions,
      'float32x2',
      arrays.polygonPositions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      buffers.featureOffsets,
      'uint32',
      arrays.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      buffers.polygonOffsets,
      'uint32',
      arrays.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      buffers.ringOffsets,
      'uint32',
      arrays.ringOffsets.length
    ),
    featureIds: importGraphBuffer(graph, 'feature-ids', buffers.featureIds, 'uint32', featureCount),
    candidateCapacity: 16,
    pointFeatureIds: importGraphBuffer(
      graph,
      'point-feature-ids',
      buffers.pointFeatureIds,
      'uint32',
      pointCount
    ),
    featureCounts: importGraphBuffer(
      graph,
      'feature-counts',
      buffers.featureCounts,
      'uint32',
      featureCount
    ),
    overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
    candidateCount: importGraphBuffer(
      graph,
      'candidate-count',
      buffers.candidateCount,
      'uint32',
      1
    ),
    uncertainCount: importGraphBuffer(
      graph,
      'uncertain-count',
      buffers.uncertainCount,
      'uint32',
      1
    ),
    matches: {
      ids: importGraphBuffer(
        graph,
        'match-ids',
        buffers.matchIds,
        'uint32',
        matchCapacity ?? pointCount
      ),
      count: importGraphBuffer(graph, 'match-count', buffers.matchCount, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'match-overflow', buffers.matchOverflow, 'uint32', 1),
      totalCount: importGraphBuffer(graph, 'match-total', buffers.matchTotal, 'uint32', 1)
    },
    ...overrides
  };
  return {graph, buffers, props};
}

function destroyFixture(fixture: JoinFixture): void {
  for (const buffer of Object.values(fixture.buffers)) {
    buffer.destroy();
  }
}

it('GPUPointInPolygonJoin joins points to features and updates per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createJoinFixture(device, POLYGON_POINTS);
  fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
  const compiled = fixture.graph.compile();
  const {buffers} = fixture;

  submitGraph(device, compiled, undefined);
  expect(await readUint32(buffers.pointFeatureIds, 10)).toEqual([
    100,
    N,
    100,
    100,
    101,
    102,
    102,
    N,
    N,
    N
  ]);
  expect(await readUint32(buffers.featureCounts, 4)).toEqual([3, 1, 2, 0]);
  expect(await readUint32(buffers.candidateCount, 1)).toEqual([9]);
  expect(await readUint32(buffers.uncertainCount, 1)).toEqual([0]);
  expect(await readUint32(buffers.overflow, 1)).toEqual([0]);
  expect((await readCompactIds(buffers.matchIds, buffers.matchCount)).sort(sortNumbers)).toEqual([
    1000, 1002, 1003, 1004, 1005, 1006
  ]);

  const moved = POLYGON_POINTS.map(point => [...point]);
  moved[8] = [6, 3];
  buffers.points.write(Float32Array.from(moved.flat()));
  submitGraph(device, compiled, undefined);
  expect((await readUint32(buffers.pointFeatureIds, 10))[8]).toBe(101);
  expect(await readUint32(buffers.featureCounts, 4)).toEqual([3, 2, 2, 0]);
  expect(await readUint32(buffers.candidateCount, 1)).toEqual([10]);

  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUPointInPolygonJoin excludes boundary points when requested', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createJoinFixture(device, POLYGON_POINTS, {includeBoundary: false});
  fixture.graph.add(new GPUPointInPolygonJoin(fixture.props));
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect((await readUint32(fixture.buffers.pointFeatureIds, 10))[3]).toBe(N);
  expect(await readUint32(fixture.buffers.featureCounts, 4)).toEqual([2, 1, 2, 0]);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUPointInPolygonJoin reports candidate, leaf, and match overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const containing = POLYGON_POINTS.map(point =>
    POLYGON_FEATURES.map((feature, row) =>
      classifyPointInFeature(point, feature) !== 'outside' ? row : -1
    ).filter(row => row >= 0)
  );

  const candidateFixture = createJoinFixture(device, POLYGON_POINTS, {candidateCapacity: 4});
  candidateFixture.graph.add(new GPUPointInPolygonJoin(candidateFixture.props));
  let compiled = candidateFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(candidateFixture.buffers.overflow, 1)).toEqual([1]);
  expect(await readUint32(candidateFixture.buffers.candidateCount, 1)).toEqual([9]);
  const featureIds = await readUint32(candidateFixture.buffers.pointFeatureIds, 10);
  for (const [row, featureId] of featureIds.entries()) {
    expect(featureId === N || containing[row].includes(featureId - 100)).toBe(true);
  }
  compiled.destroy();
  destroyFixture(candidateFixture);

  const leafFixture = createJoinFixture(device, POLYGON_POINTS, {leafCapacity: 2});
  leafFixture.graph.add(new GPUPointInPolygonJoin(leafFixture.props));
  compiled = leafFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(leafFixture.buffers.overflow, 1)).toEqual([1]);
  expect((await readUint32(leafFixture.buffers.pointFeatureIds, 10)).slice(0, 7)).toEqual([
    100,
    N,
    100,
    100,
    101,
    N,
    N
  ]);
  compiled.destroy();
  destroyFixture(leafFixture);

  const matchFixture = createJoinFixture(device, POLYGON_POINTS, {matchCapacity: 3});
  matchFixture.graph.add(new GPUPointInPolygonJoin(matchFixture.props));
  compiled = matchFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const {buffers} = matchFixture;
  const matchIds = await readCompactIds(buffers.matchIds, buffers.matchCount);
  expect(matchIds.length).toBe(3);
  expect(matchIds.every(id => [1000, 1002, 1003, 1004, 1005, 1006].includes(id))).toBe(true);
  expect(await readUint32(buffers.matchTotal, 1)).toEqual([6]);
  expect(await readUint32(buffers.matchOverflow, 1)).toEqual([1]);
  expect(await readUint32(buffers.overflow, 1)).toEqual([1]);
  compiled.destroy();
  destroyFixture(matchFixture);
});

it('GPUPointInPolygonJoin preserves point chunks and matches a random oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createJoinFixture(device, POLYGON_POINTS);
  const firstPoints = createInputBuffer(
    device,
    Float32Array.from(POLYGON_POINTS.slice(0, 6).flat())
  );
  const secondPoints = createInputBuffer(device, Float32Array.from(POLYGON_POINTS.slice(6).flat()));
  const firstIds = createOutputBuffer(device, 6);
  const secondIds = createOutputBuffer(device, 4);
  const {graph} = fixture;
  graph.add(
    new GPUPointInPolygonJoin({
      ...fixture.props,
      points: createVectorView('chunked-points', 'float32x2', [
        importGraphBuffer(graph, 'p0', firstPoints, 'float32x2', 6),
        importGraphBuffer(graph, 'p1', secondPoints, 'float32x2', 4)
      ]),
      pointFeatureIds: createVectorView('chunked-ids', 'uint32', [
        importGraphBuffer(graph, 'i0', firstIds, 'uint32', 6),
        importGraphBuffer(graph, 'i1', secondIds, 'uint32', 4)
      ]),
      sourceIds: undefined,
      matches: undefined
    })
  );
  let compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect([...(await readUint32(firstIds, 6)), ...(await readUint32(secondIds, 4))]).toEqual([
    100,
    N,
    100,
    100,
    101,
    102,
    102,
    N,
    N,
    N
  ]);
  compiled.destroy();
  destroyFixture(fixture);
  for (const buffer of [firstPoints, secondPoints, firstIds, secondIds]) buffer.destroy();

  const random = createRandom(7);
  const lattice: [number, number][] = Array.from({length: 400}, (_, index) => [
    (index % 20) * 0.8 + 0.0625,
    Math.floor(index / 20) * 0.3 + 0.0625
  ]);
  for (let index = lattice.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [lattice[index], lattice[swap]] = [lattice[swap], lattice[index]];
  }
  const randomFixture = createJoinFixture(device, lattice, {
    candidateCapacity: 1024,
    featureIds: undefined
  });
  randomFixture.graph.add(new GPUPointInPolygonJoin(randomFixture.props));
  compiled = randomFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = joinPointsInPolygons(lattice, POLYGON_FEATURES, true);
  expect(await readUint32(randomFixture.buffers.pointFeatureIds, 400)).toEqual(oracle.featureRows);
  expect(await readUint32(randomFixture.buffers.featureCounts, 4)).toEqual(oracle.counts);
  expect(await readUint32(randomFixture.buffers.overflow, 1)).toEqual([0]);
  compiled.destroy();
  destroyFixture(randomFixture);
});
