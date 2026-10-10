// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicateJoinProps
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildOracleArrays,
  evaluateOraclePredicate,
  generateRandomFeatures,
  joinWithOracle,
  line,
  point,
  polygonWithRings,
  rectangle,
  triangle,
  type OracleFeature,
  type OraclePredicate
} from './spatial-predicate-oracle';

type JoinResult = {
  pairs: [number, number][];
  count: number;
  total: number;
  overflow: number;
  candidateOverflow: number;
  candidateCount: number;
  uncertainCount: number;
  weightOffsets: number[];
  weightNeighbors: number[];
  weightValues: number[];
};

type JoinOptions = {
  distance?: number;
  excludeSameRow?: boolean;
  candidateCapacity?: number;
  pairCapacity?: number;
  leafCapacity?: number;
};

function createGeometry(
  device: Device,
  graph: GPUCommandGraph,
  name: string,
  kind: OracleFeature['kind'],
  features: OracleFeature[],
  buffers: Buffer[]
): GPUSpatialJoinGeometry {
  const arrays = buildOracleArrays(kind, features);
  const upload = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const positions = importGraphBuffer(
    graph,
    `${name}-positions`,
    upload(arrays.positions),
    'float32x2',
    arrays.positions.length / 2
  );
  const offsets = (suffix: string, data: Uint32Array) =>
    importGraphBuffer(graph, `${name}-${suffix}`, upload(data), 'uint32', data.length);
  if (arrays.kind === 'points') {
    return {kind: 'points', positions};
  }
  if (arrays.kind === 'lines') {
    return {kind: 'lines', positions, lineOffsets: offsets('line-offsets', arrays.lineOffsets)};
  }
  return {
    kind: 'polygons',
    positions,
    featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
    polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
    ringOffsets: offsets('ring-offsets', arrays.ringOffsets)
  };
}

/** Builds, compiles and runs one join, then reads every output back. */
async function runJoin(
  device: Device,
  leftKind: OracleFeature['kind'],
  lefts: OracleFeature[],
  rightKind: OracleFeature['kind'],
  rights: OracleFeature[],
  predicate: OraclePredicate,
  options: JoinOptions = {}
): Promise<JoinResult> {
  const graph = new GPUCommandGraph(device, {id: 'predicate-join'});
  const buffers: Buffer[] = [];
  const left = createGeometry(device, graph, 'left', leftKind, lefts, buffers);
  const right = createGeometry(device, graph, 'right', rightKind, rights, buffers);
  const pairCapacity = options.pairCapacity ?? Math.max(lefts.length * rights.length, 1);
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const leftIds = output('left-ids', pairCapacity);
  const rightIds = output('right-ids', pairCapacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const candidateOverflow = output('candidate-overflow', 1);
  const total = output('total', 1);
  const candidateCount = output('candidate-count', 1);
  const uncertain = output('uncertain', 1);
  const offsets = output('weight-offsets', lefts.length + 1);
  const neighbors = output('weight-neighbors', pairCapacity);
  const weightBuffer = createOutputBuffer(device, pairCapacity);
  buffers.push(weightBuffer);
  const props: GPUSpatialPredicateJoinProps = {
    left,
    right,
    predicate,
    distance: options.distance,
    excludeSameRow: options.excludeSameRow,
    candidateCapacity: options.candidateCapacity ?? Math.max(lefts.length * rights.length, 1),
    leafCapacity: options.leafCapacity,
    pairs: {
      leftIds: leftIds.view,
      rightIds: rightIds.view,
      count: count.view,
      overflow: overflow.view,
      candidateOverflow: candidateOverflow.view,
      requiredCount: total.view
    },
    weights: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: importGraphBuffer(graph, 'weight-values', weightBuffer, 'float32', pairCapacity)
    },
    candidateCount: candidateCount.view,
    uncertainCount: uncertain.view
  };
  graph.add(new GPUSpatialPredicateJoin(props));
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [countValue] = await readUint32(count.buffer, 1);
  const [totalValue] = await readUint32(total.buffer, 1);
  const lefted = await readUint32(leftIds.buffer, pairCapacity);
  const righted = await readUint32(rightIds.buffer, pairCapacity);
  const weightBytes = await weightBuffer.readAsync();
  const result: JoinResult = {
    pairs: lefted.slice(0, countValue).map((leftRow, slot) => [leftRow, righted[slot]]),
    count: countValue,
    total: totalValue,
    overflow: (await readUint32(overflow.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    candidateCount: (await readUint32(candidateCount.buffer, 1))[0],
    uncertainCount: (await readUint32(uncertain.buffer, 1))[0],
    weightOffsets: await readUint32(offsets.buffer, lefts.length + 1),
    weightNeighbors: await readUint32(neighbors.buffer, pairCapacity),
    weightValues: Array.from(
      new Float32Array(weightBytes.buffer, weightBytes.byteOffset, pairCapacity)
    )
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** Asserts that the GPU join equals the oracle, with sorted pairs and a consistent CSR. */
async function expectJoinMatchesOracle(
  device: Device,
  leftKind: OracleFeature['kind'],
  lefts: OracleFeature[],
  rightKind: OracleFeature['kind'],
  rights: OracleFeature[],
  predicate: OraclePredicate,
  options: JoinOptions = {}
): Promise<[number, number][]> {
  const expected = joinWithOracle(predicate, lefts, rights, options);
  const result = await runJoin(device, leftKind, lefts, rightKind, rights, predicate, options);
  const label = `${leftKind}/${rightKind}/${predicate}`;
  expect(result.overflow, `${label} overflow`).toBe(0);
  if (JSON.stringify(result.pairs) !== JSON.stringify(expected)) {
    const key = (pair: number[]) => pair.join(',');
    const got = new Set(result.pairs.map(key));
    const want = new Set(expected.map(key));
    for (const pair of expected) {
      if (!got.has(key(pair))) {
        console.log(
          'MISSING',
          label,
          'uncertain',
          result.uncertainCount,
          pair,
          JSON.stringify(lefts[pair[0]]),
          JSON.stringify(rights[pair[1]])
        );
      }
    }
    for (const pair of result.pairs) {
      if (!want.has(key(pair))) {
        console.log(
          'EXTRA',
          label,
          pair,
          JSON.stringify(lefts[pair[0]]),
          JSON.stringify(rights[pair[1]])
        );
      }
    }
  }
  expect(result.pairs, label).toEqual(expected);
  expect(result.total, `${label} total`).toBe(expected.length);
  // CSR: rows are left features, neighbors ascend within a row.
  const rowPairs: [number, number][] = [];
  for (let row = 0; row < lefts.length; row++) {
    for (let slot = result.weightOffsets[row]; slot < result.weightOffsets[row + 1]; slot++) {
      rowPairs.push([row, result.weightNeighbors[slot]]);
      expect(result.weightValues[slot]).toBe(1);
    }
  }
  expect(rowPairs, `${label} weights`).toEqual(expected);
  return expected;
}

const ROUNDS = 2;
const KINDS: OracleFeature['kind'][] = ['points', 'lines', 'polygons'];
const PREDICATES: OraclePredicate[] = ['intersects', 'contains', 'within', 'dwithin'];

it('oracle sanity: OGC boundary semantics', () => {
  const square = rectangle(0, 0, 4, 4);
  expect(evaluateOraclePredicate('intersects', point(0, 2), square)).toBe(true);
  expect(evaluateOraclePredicate('contains', square, point(0, 2))).toBe(false);
  expect(evaluateOraclePredicate('within', point(0, 2), square)).toBe(false);
  expect(evaluateOraclePredicate('within', point(2, 2), square)).toBe(true);
  expect(evaluateOraclePredicate('contains', line([0, 0], [4, 0]), point(0, 0))).toBe(false);
  expect(evaluateOraclePredicate('contains', line([0, 0], [4, 0]), point(2, 0))).toBe(true);
  expect(evaluateOraclePredicate('contains', square, line([0, 0], [4, 0]))).toBe(false);
  expect(evaluateOraclePredicate('contains', square, square)).toBe(true);
  expect(evaluateOraclePredicate('intersects', square, rectangle(4, 4, 2, 2))).toBe(true);
  expect(evaluateOraclePredicate('contains', square, rectangle(4, 0, 2, 2))).toBe(false);
});

it('GPUSpatialPredicateJoin matches the oracle for all kinds, predicates and degenerate cases', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Hand-built scenes cover shared edges, touching corners, collinear overlaps, holes and endpoints.
  const hole = polygonWithRings(
    [
      [0, 0],
      [6, 0],
      [6, 6],
      [0, 6]
    ],
    [
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2]
    ]
  );
  const scenes: Record<OracleFeature['kind'], OracleFeature[]> = {
    points: [
      point(0, 0),
      point(3, 3),
      point(2, 2),
      point(2, 3),
      point(6, 6),
      point(7, 7),
      point(1, 1),
      point(3, 0),
      point(4.5, 4.5),
      point(10, 10)
    ],
    lines: [
      line([0, 0], [6, 0]),
      line([1, 0], [3, 0]),
      line([2, 2], [4, 4]),
      line([3, 1], [3, 5]),
      line([1, 1], [5, 1]),
      line([6, 6], [8, 8]),
      line([0, 0], [4, 0], [4, 4], [0, 4], [0, 0]),
      line([7, 0], [9, 0]),
      line([2, 3], [4, 3]),
      line([0, 3], [6, 3])
    ],
    polygons: [
      hole,
      rectangle(0, 0, 6, 6),
      rectangle(2, 2, 2, 2),
      rectangle(6, 0, 2, 2),
      rectangle(6, 6, 2, 2),
      triangle(0, 0, 3),
      rectangle(1, 1, 4, 4),
      rectangle(2.5, 2.5, 1, 1),
      rectangle(1, 1, 1, 1),
      rectangle(10, 10, 1, 1)
    ]
  };
  for (const leftKind of KINDS) {
    for (const rightKind of KINDS) {
      for (const predicate of PREDICATES) {
        await expectJoinMatchesOracle(
          device,
          leftKind,
          scenes[leftKind],
          rightKind,
          scenes[rightKind],
          predicate,
          {distance: predicate === 'dwithin' ? 1.5 : undefined}
        );
      }
    }
  }
});

it('GPUSpatialPredicateJoin matches the oracle on random scenes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let seed = 1;
  let totalMatches = 0;
  for (let round = 0; round < ROUNDS; round++) {
    for (const leftKind of KINDS) {
      for (const rightKind of KINDS) {
        for (const predicate of PREDICATES) {
          const lefts = generateRandomFeatures(leftKind, 14, seed++);
          const rights = generateRandomFeatures(rightKind, 14, seed++);
          const expected = await expectJoinMatchesOracle(
            device,
            leftKind,
            lefts,
            rightKind,
            rights,
            predicate,
            {distance: predicate === 'dwithin' ? 1 : undefined}
          );
          totalMatches += expected.length;
        }
      }
    }
  }
  // A failed WGSL compile gives silent zeros, so the scenes must produce plenty of matches.
  expect(totalMatches).toBeGreaterThan(200);
});

it('GPUSpatialPredicateJoin documented boundary semantics', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const square = rectangle(0, 0, 4, 4);
  const run = async (
    leftKind: OracleFeature['kind'],
    lefts: OracleFeature[],
    rightKind: OracleFeature['kind'],
    rights: OracleFeature[],
    predicate: OraclePredicate
  ) => (await runJoin(device, leftKind, lefts, rightKind, rights, predicate)).pairs;

  // A point on a polygon boundary intersects but is not within; an interior point is within.
  const boundaryAndInterior = [point(0, 2), point(2, 2), point(9, 9)];
  expect(await run('points', boundaryAndInterior, 'polygons', [square], 'intersects')).toEqual([
    [0, 0],
    [1, 0]
  ]);
  expect(await run('points', boundaryAndInterior, 'polygons', [square], 'within')).toEqual([
    [1, 0]
  ]);
  expect(await run('polygons', [square], 'points', boundaryAndInterior, 'contains')).toEqual([
    [0, 1]
  ]);
  // Polygons touching along an edge or at a corner intersect but neither contains the other.
  const neighbors = [rectangle(4, 0, 2, 2), rectangle(4, 4, 1, 1), rectangle(5, 5, 1, 1)];
  expect(await run('polygons', [square], 'polygons', neighbors, 'intersects')).toEqual([
    [0, 0],
    [0, 1]
  ]);
  expect(await run('polygons', [square], 'polygons', neighbors, 'contains')).toEqual([]);
  // A polygon contains itself and an inner polygon that touches its boundary from inside.
  const inner = [square, rectangle(0, 0, 2, 2), rectangle(1, 1, 2, 2)];
  expect(await run('polygons', [square], 'polygons', inner, 'contains')).toEqual([
    [0, 0],
    [0, 1],
    [0, 2]
  ]);
  expect(await run('polygons', inner, 'polygons', [square], 'within')).toEqual([
    [0, 0],
    [1, 0],
    [2, 0]
  ]);
  // A hole makes a polygon inside it disjoint; a polygon equal to the hole is not contained.
  const donut = polygonWithRings(
    [
      [0, 0],
      [6, 0],
      [6, 6],
      [0, 6]
    ],
    [
      [2, 2],
      [2, 4],
      [4, 4],
      [4, 2]
    ]
  );
  const inHole = [rectangle(2, 2, 2, 2), rectangle(2.5, 2.5, 1, 1), rectangle(1, 1, 4, 4)];
  expect(await run('polygons', [donut], 'polygons', inHole, 'contains')).toEqual([]);
  expect(await run('polygons', [donut], 'polygons', inHole, 'intersects')).toEqual([
    [0, 0],
    [0, 2]
  ]);
  expect(
    await run('points', [point(3, 3), point(2, 3), point(1, 1)], 'polygons', [donut], 'within')
  ).toEqual([[2, 0]]);
  // Linestring boundary: endpoints are boundary unless closed.
  const open = line([0, 0], [4, 0]);
  expect(await run('lines', [open], 'points', [point(0, 0), point(2, 0)], 'contains')).toEqual([
    [0, 1]
  ]);
  expect(
    await run('lines', [line([0, 0], [4, 0], [4, 4], [0, 0])], 'points', [point(0, 0)], 'contains')
  ).toEqual([[0, 0]]);
  // Collinear overlap: contained only when the segment is fully covered.
  expect(
    await run(
      'lines',
      [line([0, 0], [2, 0], [4, 0])],
      'lines',
      [line([1, 0], [3, 0]), line([3, 0], [5, 0])],
      'contains'
    )
  ).toEqual([[0, 0]]);
});

it('GPUSpatialPredicateJoin dwithin is inclusive and joins distant features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const lefts = [point(0, 0), point(20, 20)];
  const rights = [point(3, 4), point(3, 4.5), point(20, 25)];
  const result = await runJoin(device, 'points', lefts, 'points', rights, 'dwithin', {distance: 5});
  expect(result.pairs).toEqual([
    [0, 0],
    [1, 2]
  ]);
});

it('GPUSpatialPredicateJoin self-join with excludeSameRow produces symmetric weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = generateRandomFeatures('polygons', 12, 99);
  const expected = await expectJoinMatchesOracle(
    device,
    'polygons',
    features,
    'polygons',
    features,
    'intersects',
    {excludeSameRow: true}
  );
  expect(expected.length).toBeGreaterThan(0);
  expect(expected.some(([left, right]) => left === right)).toBe(false);
  const keys = new Set(expected.map(pair => pair.join(',')));
  for (const [left, right] of expected) {
    expect(keys.has(`${right},${left}`)).toBe(true);
  }
});

it('GPUSpatialPredicateJoin reports overflow and keeps a sorted prefix', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const lefts = generateRandomFeatures('polygons', 10, 5);
  const rights = generateRandomFeatures('polygons', 10, 6);
  const expected = joinWithOracle('intersects', lefts, rights);
  expect(expected.length).toBeGreaterThan(8);

  const pairLimited = await runJoin(device, 'polygons', lefts, 'polygons', rights, 'intersects', {
    pairCapacity: 5
  });
  expect(pairLimited.overflow).toBe(1);
  expect(pairLimited.count).toBe(5);
  expect(pairLimited.total).toBe(expected.length);
  expect(pairLimited.pairs).toEqual(expected.slice(0, 5));

  const candidateLimited = await runJoin(
    device,
    'polygons',
    lefts,
    'polygons',
    rights,
    'intersects',
    {
      candidateCapacity: 6
    }
  );
  expect(candidateLimited.overflow).toBe(0);
  expect(candidateLimited.candidateOverflow).toBe(1);
  expect(candidateLimited.candidateCount).toBeGreaterThan(6);
  // Dropped candidates are the tail of the (left, right) order, so the output stays sorted.
  const expectedKeys = new Set(expected.map(pair => pair.join(',')));
  for (const pair of candidateLimited.pairs) {
    expect(expectedKeys.has(pair.join(','))).toBe(true);
  }
  expect(candidateLimited.pairs).toEqual(
    [...candidateLimited.pairs].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  );
  expect(candidateLimited.pairs.length).toBeLessThan(expected.length);

  const leafLimited = await runJoin(device, 'polygons', lefts, 'polygons', rights, 'intersects', {
    leafCapacity: 2
  });
  expect(leafLimited.overflow).toBe(0);
  expect(leafLimited.candidateOverflow).toBe(1);
});

it('GPUSpatialPredicateJoin updates per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'predicate-join-frames'});
  const buffers: Buffer[] = [];
  const lefts = [point(1, 1), point(8, 8)];
  const left = createGeometry(device, graph, 'left', 'points', lefts, buffers);
  const right = createGeometry(
    device,
    graph,
    'right',
    'polygons',
    [rectangle(0, 0, 4, 4)],
    buffers
  );
  const leftIds = createOutputBuffer(device, 4);
  const rightIds = createOutputBuffer(device, 4);
  const count = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  buffers.push(leftIds, rightIds, count, overflow);
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate: 'within',
      candidateCapacity: 4,
      pairs: {
        leftIds: importGraphBuffer(graph, 'l', leftIds, 'uint32', 4),
        rightIds: importGraphBuffer(graph, 'r', rightIds, 'uint32', 4),
        count: importGraphBuffer(graph, 'c', count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'o', overflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(count, 1)).toEqual([1]);
  expect((await readUint32(leftIds, 1))[0]).toBe(0);
  // buffers[0] holds the left positions: move the second point into the polygon.
  buffers[0].write(Float32Array.from([1, 1, 2, 2]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(count, 1)).toEqual([2]);
  expect(await readUint32(leftIds, 2)).toEqual([0, 1]);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUSpatialPredicateJoin validates its props', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'predicate-join-validation'});
  const buffers: Buffer[] = [];
  const geometry = createGeometry(device, graph, 'g', 'points', [point(0, 0)], buffers);
  expect(
    () =>
      new GPUSpatialPredicateJoin({
        left: geometry,
        right: geometry,
        predicate: 'dwithin',
        candidateCapacity: 1
      })
  ).toThrow(/distance|pairs/);
  expect(
    () =>
      new GPUSpatialPredicateJoin({
        left: geometry,
        right: geometry,
        predicate: 'intersects',
        candidateCapacity: 1
      })
  ).toThrow(/pairs, weights/);
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
