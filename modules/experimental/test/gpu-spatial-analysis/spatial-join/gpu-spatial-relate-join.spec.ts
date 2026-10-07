// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  formatGPUSpatialRelate,
  packGPUSpatialRelatePattern,
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUPointInPolygonJoin,
  GPUSpatialJoinCandidates,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildOracleArrays,
  line,
  point,
  polygonWithRings,
  rectangle,
  type OracleFeature
} from './spatial-predicate-oracle';
import {RELATE_KINDS, getRandomRelateSeeds, getRelateSide} from './spatial-relate-scenes';
import {createRandom} from '../segment-intersection/segment-intersection-oracle';
import {SHAPELY_RELATE_FIXTURES} from './shapely-relate-fixtures';

const INTERSECTS_PATTERNS = ['T********', '*T*******', '***T*****', '****T****'];

type RunOptions = {
  predicate: GPUSpatialPredicate;
  pattern?: string | string[];
  distance?: number;
  engine?: 'auto' | 'fast' | 'relate';
  how?: 'inner' | 'anti';
  relate?: boolean;
  prepared?: boolean;
  excludeSameRow?: boolean;
};

type RunResult = {
  pairs: [number, number][];
  matrices: string[];
  unmatched: number[];
  unmatchedOverflow: number;
  overflow: number;
  uncertainCount: number;
  candidateCount: number;
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

async function runJoin(
  device: Device,
  leftKind: OracleFeature['kind'],
  lefts: OracleFeature[],
  rightKind: OracleFeature['kind'],
  rights: OracleFeature[],
  options: RunOptions
): Promise<RunResult> {
  const graph = new GPUCommandGraph(device, {id: 'relate-join'});
  const buffers: Buffer[] = [];
  const left = createGeometry(device, graph, 'left', leftKind, lefts, buffers);
  const right = createGeometry(device, graph, 'right', rightKind, rights, buffers);
  const capacity = Math.max(lefts.length * rights.length, 1);
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const leftIds = output('left-ids', capacity);
  const rightIds = output('right-ids', capacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const relate = output('relate', capacity);
  const unmatchedIds = output('unmatched-ids', lefts.length);
  const unmatchedCount = output('unmatched-count', 1);
  const unmatchedOverflow = output('unmatched-overflow', 1);
  const candidateCount = output('candidate-count', 1);
  const uncertain = output('uncertain', 1);
  const isAnti = options.how === 'anti';
  let prepared: GPUSpatialJoinPrepared | undefined;
  if (options.prepared) {
    prepared = new GPUSpatialJoinPrepared({geometry: right});
    graph.add(prepared);
  }
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate: options.predicate,
      pattern: options.pattern,
      distance: options.distance,
      engine: options.engine,
      how: options.how,
      excludeSameRow: options.excludeSameRow,
      prepared,
      candidateCapacity: capacity,
      pairs: isAnti
        ? undefined
        : {
            leftIds: leftIds.view,
            rightIds: rightIds.view,
            count: count.view,
            overflow: overflow.view
          },
      relate: options.relate ? relate.view : undefined,
      unmatched: isAnti
        ? {ids: unmatchedIds.view, count: unmatchedCount.view, overflow: unmatchedOverflow.view}
        : undefined,
      candidateCount: candidateCount.view,
      uncertainCount: uncertain.view
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [countValue] = await readUint32(count.buffer, 1);
  const [unmatchedValue] = await readUint32(unmatchedCount.buffer, 1);
  const lefted = await readUint32(leftIds.buffer, capacity);
  const righted = await readUint32(rightIds.buffer, capacity);
  const matrices = await readUint32(relate.buffer, capacity);
  const result: RunResult = {
    pairs: lefted.slice(0, countValue).map((row, slot) => [row, righted[slot]]),
    matrices: matrices.slice(0, countValue).map(formatGPUSpatialRelate),
    unmatched: (await readUint32(unmatchedIds.buffer, lefts.length)).slice(0, unmatchedValue),
    unmatchedOverflow: (await readUint32(unmatchedOverflow.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    uncertainCount: (await readUint32(uncertain.buffer, 1))[0],
    candidateCount: (await readUint32(candidateCount.buffer, 1))[0]
  };
  compiled.destroy();
  prepared?.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

const SHAPELY_NAMES: Record<string, string> = {
  intersects: 'intersects',
  contains: 'contains',
  within: 'within',
  covers: 'covers',
  coveredBy: 'covered_by',
  touches: 'touches',
  crosses: 'crosses',
  overlaps: 'overlaps',
  equals: 'equals',
  containsProperly: 'contains_properly'
};

function getScene(leftKind: OracleFeature['kind'], rightKind: OracleFeature['kind']) {
  const [leftSeed, rightSeed] = getRandomRelateSeeds(leftKind, rightKind);
  return {
    lefts: getRelateSide(leftKind, leftSeed, true),
    rights: getRelateSide(rightKind, rightSeed, true),
    fixture: SHAPELY_RELATE_FIXTURES[`${leftKind}-${rightKind}`]
  };
}

/** Pairs where a Shapely bit string is `1`. */
function getShapelyPairs(bits: string, rightCount: number): [number, number][] {
  const pairs: [number, number][] = [];
  for (let index = 0; index < bits.length; index++) {
    if (bits[index] === '1') {
      pairs.push([Math.floor(index / rightCount), index % rightCount]);
    }
  }
  return pairs;
}

it('GPUSpatialPredicateJoin relate pattern emits Shapely DE-9IM matrices for all kinds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let total = 0;
  for (const leftKind of RELATE_KINDS) {
    for (const rightKind of RELATE_KINDS) {
      const {lefts, rights, fixture} = getScene(leftKind, rightKind);
      const label = `${leftKind}-${rightKind}`;
      const result = await runJoin(device, leftKind, lefts, rightKind, rights, {
        predicate: 'relate',
        pattern: INTERSECTS_PATTERNS,
        relate: true
      });
      const expectedPairs = getShapelyPairs(fixture.predicates.intersects, rights.length);
      expect(result.overflow, label).toBe(0);
      expect(result.pairs, label).toEqual(expectedPairs);
      const expectedMatrices = expectedPairs.map(([row, column]) => {
        const pair = row * rights.length + column;
        return fixture.relate.slice(pair * 9, pair * 9 + 9);
      });
      const mismatches = expectedPairs
        .map(([row, column], slot) => ({
          row,
          column,
          got: result.matrices[slot],
          want: expectedMatrices[slot]
        }))
        .filter(({got, want}) => got !== want);
      expect(
        mismatches.map(
          ({row, column, got, want}) =>
            `${row},${column} ${got} != ${want} ${JSON.stringify(lefts[row])} ${JSON.stringify(rights[column])}`
        ),
        label
      ).toEqual([]);
      // Dyadic scenes are certified, except point/polygon where the robust classifier decides.
      expect(result.uncertainCount, label).toBe(0);
      total += result.pairs.length;
    }
  }
  expect(total).toBeGreaterThan(500);
});

it('GPUSpatialPredicateJoin named predicates match Shapely for all kinds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const predicates: GPUSpatialPredicate[] = [
    'covers',
    'coveredBy',
    'touches',
    'crosses',
    'overlaps',
    'equals',
    'containsProperly'
  ];
  let total = 0;
  for (const leftKind of RELATE_KINDS) {
    for (const rightKind of RELATE_KINDS) {
      const {lefts, rights, fixture} = getScene(leftKind, rightKind);
      for (const predicate of predicates) {
        const result = await runJoin(device, leftKind, lefts, rightKind, rights, {predicate});
        const expected = getShapelyPairs(
          fixture.predicates[SHAPELY_NAMES[predicate]],
          rights.length
        );
        expect(result.pairs, `${leftKind}-${rightKind} ${predicate}`).toEqual(expected);
        expect(result.uncertainCount).toBe(0);
        total += expected.length;
      }
    }
  }
  expect(total).toBeGreaterThan(300);
});

it('GPUSpatialPredicateJoin intersects, contains and within agree with their relate masks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let total = 0;
  for (const leftKind of RELATE_KINDS) {
    for (const rightKind of RELATE_KINDS) {
      const {lefts, rights, fixture} = getScene(leftKind, rightKind);
      for (const predicate of ['intersects', 'contains', 'within'] as const) {
        const expected = getShapelyPairs(fixture.predicates[predicate], rights.length);
        // Short-circuiting kernel.
        const fast = await runJoin(device, leftKind, lefts, rightKind, rights, {predicate});
        // Same predicate through the relate engine, with the matrix of each pair.
        const viaRelate = await runJoin(device, leftKind, lefts, rightKind, rights, {
          predicate,
          relate: true
        });
        const label = `${leftKind}-${rightKind} ${predicate}`;
        expect(fast.pairs, `${label} fast`).toEqual(expected);
        expect(viaRelate.pairs, `${label} relate`).toEqual(expected);
        expect(viaRelate.matrices.length).toBe(expected.length);
        total += expected.length;
      }
    }
  }
  expect(total).toBeGreaterThan(300);
});

it("GPUSpatialPredicateJoin how: 'anti' emits unmatched left rows", async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let total = 0;
  for (const [leftKind, rightKind] of [
    ['points', 'polygons'],
    ['lines', 'lines'],
    ['polygons', 'polygons']
  ] as const) {
    const {lefts, rights, fixture} = getScene(leftKind, rightKind);
    for (const predicate of ['intersects', 'touches'] as const) {
      const result = await runJoin(device, leftKind, lefts, rightKind, rights, {
        predicate,
        how: 'anti'
      });
      const matched = new Set(
        getShapelyPairs(fixture.predicates[predicate], rights.length).map(([row]) => row)
      );
      const expected = lefts.map((_, row) => row).filter(row => !matched.has(row));
      expect(result.unmatched, `${leftKind}-${rightKind} ${predicate}`).toEqual(expected);
      expect(result.unmatchedOverflow).toBe(0);
      total += expected.length;
    }
  }
  expect(total).toBeGreaterThan(5);
});

it('GPUSpatialPredicateJoin relate handles Shapely reference cases on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const squares = [rectangle(0, 0, 2, 2), rectangle(2, 0, 2, 2), rectangle(0, 0, 2, 2)];
  const shared = await runJoin(device, 'polygons', [squares[0]], 'polygons', [squares[1]], {
    predicate: 'relate',
    pattern: INTERSECTS_PATTERNS,
    relate: true
  });
  // Shapely: shapely.relate(box(0, 0, 2, 2), box(2, 0, 4, 2)) == 'FF2F11212'.
  expect(shared.matrices).toEqual(['FF2F11212']);
  const identical = await runJoin(device, 'polygons', [squares[0]], 'polygons', [squares[2]], {
    predicate: 'equals'
  });
  expect(identical.pairs).toEqual([[0, 0]]);
  // A polygon filling the hole of another touches it all along the hole ring: 'FF2F112F2'.
  const ring = polygonWithRings(
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
  const filler = await runJoin(device, 'polygons', [ring], 'polygons', [rectangle(2, 2, 2, 2)], {
    predicate: 'relate',
    pattern: ['F***T****'],
    relate: true
  });
  expect(filler.pairs).toEqual([[0, 0]]);
  expect(filler.matrices[0]).toBe('FF2F112F2');
  const crossing = await runJoin(
    device,
    'lines',
    [line([0, 0], [4, 0])],
    'lines',
    [line([2, -1], [2, 1])],
    {predicate: 'crosses', relate: true}
  );
  expect(crossing.pairs).toEqual([[0, 0]]);
  expect(crossing.matrices).toEqual(['0F1FF0102']);
  const onBoundary = await runJoin(device, 'points', [point(0, 1)], 'polygons', [squares[0]], {
    predicate: 'touches',
    relate: true
  });
  expect(onBoundary.pairs).toEqual([[0, 0]]);
  expect(onBoundary.matrices).toEqual(['F0FFFF212']);
});

it('GPUSpatialPredicateJoin rejects invalid relate configurations', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'relate-validation'});
  const buffers: Buffer[] = [];
  const geometry = createGeometry(device, graph, 'g', 'points', [point(0, 0)], buffers);
  const ids = importGraphBuffer(graph, 'ids', createOutputBuffer(device, 1), 'uint32', 1);
  const scalar = importGraphBuffer(graph, 'scalar', createOutputBuffer(device, 1), 'uint32', 1);
  const base = {left: geometry, right: geometry, candidateCapacity: 1};
  const pairs = {leftIds: ids, rightIds: ids, count: scalar, overflow: scalar};
  expect(() => new GPUSpatialPredicateJoin({...base, predicate: 'relate', pairs} as never)).toThrow(
    /pattern/
  );
  expect(
    () =>
      new GPUSpatialPredicateJoin({...base, predicate: 'intersects', pattern: 'T********', pairs})
  ).toThrow(/pattern requires/);
  expect(
    () => new GPUSpatialPredicateJoin({...base, predicate: 'relate', pattern: 'F********', pairs})
  ).toThrow(/disjoint/);
  expect(
    () => new GPUSpatialPredicateJoin({...base, predicate: 'relate', pattern: 'T*', pairs})
  ).toThrow(/nine characters/);
  expect(() => new GPUSpatialPredicateJoin({...base, predicate: 'touches', how: 'anti'})).toThrow(
    /unmatched/
  );
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUSpatialPredicateJoin reuses a prepared right-hand side until invalidated', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'prepared-join'});
  const buffers: Buffer[] = [];
  const polygons = [rectangle(0, 0, 4, 4), rectangle(10, 10, 4, 4)];
  const right = createGeometry(device, graph, 'right', 'polygons', polygons, buffers);
  const left = createGeometry(
    device,
    graph,
    'left',
    'points',
    [point(1, 1), point(11, 11), point(20, 20)],
    buffers
  );
  const leftPositions = buffers[buffers.length - 1];
  const rightPositions = buffers[0];
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const leftIds = output('l', 8);
  const rightIds = output('r', 8);
  const count = output('c', 1);
  const overflow = output('o', 1);
  const prepared = new GPUSpatialJoinPrepared({geometry: right});
  graph.add(prepared);
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate: 'within',
      prepared,
      candidateCapacity: 8,
      pairs: {
        leftIds: leftIds.view,
        rightIds: rightIds.view,
        count: count.view,
        overflow: overflow.view
      }
    })
  );
  const compiled = graph.compile();
  const readPairs = async () => {
    const [matched] = await readUint32(count.buffer, 1);
    const lefted = await readUint32(leftIds.buffer, matched);
    const righted = await readUint32(rightIds.buffer, matched);
    return lefted.map((row, slot) => [row, righted[slot]]);
  };
  submitGraph(device, compiled, undefined);
  expect(await readPairs()).toEqual([
    [0, 0],
    [1, 1]
  ]);
  expect(prepared.encodedBuildCount).toBe(1);

  // Animated left features: queries see the new points, the index is not rebuilt.
  leftPositions.write(Float32Array.from([1, 1, 3, 3, 12, 12]));
  submitGraph(device, compiled, undefined);
  expect(await readPairs()).toEqual([
    [0, 0],
    [1, 0],
    [2, 1]
  ]);
  expect(prepared.encodedBuildCount).toBe(1);

  // The right-hand side changes: the static tree is stale until invalidated (a reused build, not a
  // result cache). Candidates still come from the old boxes, so the point in the moved polygon is missed.
  rightPositions.write(
    Float32Array.from([20, 20, 24, 20, 24, 24, 20, 24, 10, 10, 14, 10, 14, 14, 10, 14])
  );
  leftPositions.write(Float32Array.from([21, 21, 3, 3, 12, 12]));
  submitGraph(device, compiled, undefined);
  expect(await readPairs()).toEqual([[2, 1]]);
  expect(prepared.encodedBuildCount).toBe(1);
  prepared.invalidate();
  submitGraph(device, compiled, undefined);
  expect(await readPairs()).toEqual([
    [0, 0],
    [2, 1]
  ]);
  expect(prepared.encodedBuildCount).toBe(2);
  submitGraph(device, compiled, undefined);
  expect(prepared.encodedBuildCount).toBe(2);
  compiled.destroy();
  prepared.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUSpatialJoinCandidates emits sorted bounding-box candidates', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {lefts, rights} = getScene('polygons', 'lines');
  const bounds = (feature: OracleFeature) => {
    const vertices =
      feature.kind === 'points'
        ? [feature.vertex]
        : feature.kind === 'lines'
          ? feature.vertices
          : feature.polygons.flat(2);
    const xs = vertices.map(vertex => vertex[0]);
    const ys = vertices.map(vertex => vertex[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  };
  const margin = 0.5;
  const expected: [number, number][] = [];
  for (const [leftRow, left] of lefts.entries()) {
    for (const [rightRow, right] of rights.entries()) {
      const a = bounds(left);
      const b = bounds(right);
      if (
        a[0] - margin <= b[2] &&
        b[0] <= a[2] + margin &&
        a[1] - margin <= b[3] &&
        b[1] <= a[3] + margin
      ) {
        expected.push([leftRow, rightRow]);
      }
    }
  }
  expect(expected.length).toBeGreaterThan(20);
  for (const usePrepared of [false, true]) {
    const graph = new GPUCommandGraph(device, {id: 'candidates'});
    const buffers: Buffer[] = [];
    const left = createGeometry(device, graph, 'left', 'polygons', lefts, buffers);
    const right = createGeometry(device, graph, 'right', 'lines', rights, buffers);
    const capacity = lefts.length * rights.length;
    const output = (name: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const leftIds = output('l', capacity);
    const rightIds = output('r', capacity);
    const count = output('c', 1);
    const overflow = output('o', 1);
    const total = output('t', 1);
    const prepared = usePrepared ? new GPUSpatialJoinPrepared({geometry: right}) : undefined;
    if (prepared) {
      graph.add(prepared);
    }
    graph.add(
      new GPUSpatialJoinCandidates({
        left,
        right,
        prepared,
        distance: margin,
        pairs: {
          leftIds: leftIds.view,
          rightIds: rightIds.view,
          count: count.view,
          overflow: overflow.view,
          totalCount: total.view
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const [matched] = await readUint32(count.buffer, 1);
    const lefted = await readUint32(leftIds.buffer, matched);
    const righted = await readUint32(rightIds.buffer, matched);
    expect(
      lefted.map((row, slot) => [row, righted[slot]]),
      `prepared ${usePrepared}`
    ).toEqual(expected);
    expect((await readUint32(total.buffer, 1))[0]).toBe(expected.length);
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    compiled.destroy();
    prepared?.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
});

it('GPUPointInPolygonJoin reuses a prepared polygon tree for animated points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'pip-prepared'});
  const buffers: Buffer[] = [];
  const polygons = createGeometry(
    device,
    graph,
    'polygons',
    'polygons',
    [rectangle(0, 0, 4, 4), rectangle(10, 10, 4, 4)],
    buffers
  );
  if (polygons.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const pointBuffer = createInputBuffer(device, Float32Array.from([1, 1, 11, 11, 20, 20]));
  const assignments = createOutputBuffer(device, 3);
  const overflow = createOutputBuffer(device, 1);
  buffers.push(pointBuffer, assignments, overflow);
  const prepared = new GPUSpatialJoinPrepared({geometry: polygons, spatialSort: true});
  graph.add(prepared);
  graph.add(
    new GPUPointInPolygonJoin({
      points: importGraphBuffer(graph, 'points', pointBuffer, 'float32x2', 3),
      polygonPositions: polygons.positions,
      featureOffsets: polygons.featureOffsets,
      polygonOffsets: polygons.polygonOffsets,
      ringOffsets: polygons.ringOffsets,
      prepared,
      candidateCapacity: 8,
      pointFeatureIds: importGraphBuffer(graph, 'assignments', assignments, 'uint32', 3),
      overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(assignments, 3)).toEqual([0, 1, GPU_SPATIAL_JOIN_NO_FEATURE]);
  expect((await readUint32(overflow, 1))[0]).toBe(0);
  pointBuffer.write(Float32Array.from([12, 12, 2, 3, 13, 10]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(assignments, 3)).toEqual([1, 0, 1]);
  expect(prepared.encodedBuildCount).toBe(1);
  compiled.destroy();
  prepared.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

function stepFloat32(value: number, steps: number): number {
  const floats = new Float32Array([value]);
  const bits = new Int32Array(floats.buffer);
  bits[0] += value >= 0 ? steps : -steps;
  return floats[0];
}

it('GPUSpatialPredicateJoin relate is exact on non-dyadic near-degenerate lines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const exact = await runJoin(
    device,
    'lines',
    [line([0, 0], [4, 4])],
    'lines',
    [line([1, 1], [2, 2])],
    {predicate: 'covers'}
  );
  expect(exact.pairs).toEqual([[0, 0]]);
  expect(exact.uncertainCount).toBe(0);
  // The diagonal a-b passes through (x, x) for every f32 x, so a vertical line starting k ulps
  // above or below the diagonal point has an exact answer from the sign of k, although none of the
  // f32 products are exact. The old 2^-20 filter reported these as uncertain.
  const random = createRandom(5);
  const diagonal = line([Math.fround(0.1), Math.fround(0.1)], [Math.fround(0.7), Math.fround(0.7)]);
  const rights: OracleFeature[] = [];
  const offsets: number[] = [];
  for (let index = 0; index < 120; index++) {
    const x = Math.fround(0.15 + random() * 0.5);
    const k = Math.floor(random() * 5) - 2;
    offsets.push(k);
    // Above the diagonal (k > 0) the line starts off it; it always ends below it, at y = 0.05.
    rights.push(line([x, stepFloat32(x, k)], [x, Math.fround(0.05)]));
  }
  const touches = await runJoin(device, 'lines', [diagonal], 'lines', rights, {
    predicate: 'touches'
  });
  const crosses = await runJoin(device, 'lines', [diagonal], 'lines', rights, {
    predicate: 'crosses'
  });
  const expectedCrosses = offsets.flatMap((k, row) => (k > 0 ? [row] : []));
  const expectedTouches = offsets.flatMap((k, row) => (k === 0 ? [row] : []));
  expect(offsets.some(k => k === 0)).toBe(true);
  expect(offsets.some(k => k < 0)).toBe(true);
  expect(touches.pairs.map(pair => pair[1]).sort((a, b) => a - b)).toEqual(expectedTouches);
  expect(crosses.pairs.map(pair => pair[1]).sort((a, b) => a - b)).toEqual(expectedCrosses);
  expect(touches.uncertainCount).toBe(0);
  expect(crosses.uncertainCount).toBe(0);
});

it('GPUSpatialPredicateJoin relate is exact for lines against a diagonal polygon edge', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const f = Math.fround;
  const polygon = polygonWithRings([
    [f(0.1), f(0.1)],
    [f(0.7), f(0.7)],
    [f(0.7), f(0.1)]
  ]);
  const random = createRandom(11);
  const rights: OracleFeature[] = [];
  const offsets: number[] = [];
  for (let index = 0; index < 120; index++) {
    const x = f(0.2 + random() * 0.45);
    // +1 ulp is skipped: the outside piece is then below one ulp long and its f32 midpoint
    // rounds onto the edge (piece midpoints are not exact; orientation signs are).
    const k = [-2, -1, 0, 2, 3][Math.floor(random() * 5)];
    offsets.push(k);
    // Starts k ulps above the diagonal edge (k > 0: outside) and runs down into the polygon.
    rights.push(line([x, stepFloat32(x, k)], [x, f(0.12)]));
  }
  const rows = (pairs: [number, number][]) => pairs.map(pair => pair[1]).sort((a, b) => a - b);
  const only = (test: (k: number) => boolean) =>
    offsets.flatMap((k, row) => (test(k) ? [row] : []));
  const covers = await runJoin(device, 'polygons', [polygon], 'lines', rights, {
    predicate: 'covers'
  });
  const crosses = await runJoin(device, 'polygons', [polygon], 'lines', rights, {
    predicate: 'crosses'
  });
  const proper = await runJoin(device, 'polygons', [polygon], 'lines', rights, {
    predicate: 'containsProperly'
  });
  expect(offsets.some(k => k === 0)).toBe(true);
  expect(rows(covers.pairs)).toEqual(only(k => k <= 0));
  expect(rows(crosses.pairs)).toEqual(only(k => k > 0));
  expect(rows(proper.pairs)).toEqual(only(k => k < 0));
  expect(covers.uncertainCount + crosses.uncertainCount + proper.uncertainCount).toBe(0);
});

it('GPUSpatialPredicateJoin relate still counts non-finite input as uncertain', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runJoin(
    device,
    'lines',
    [line([0, 0], [4, 4])],
    'lines',
    [line([1, 1], [Infinity, 2]), line([1, 0], [1, 4])],
    {predicate: 'crosses'}
  );
  expect(result.pairs).toEqual([[0, 1]]);
});

type PerFrameSetting = {pattern?: string | string[]; distance?: number};

/** Runs one compiled join with per-frame pattern and distance views, once per setting. */
async function runPerFrame(
  device: Device,
  leftKind: OracleFeature['kind'],
  lefts: OracleFeature[],
  rightKind: OracleFeature['kind'],
  rights: OracleFeature[],
  predicate: GPUSpatialPredicate,
  engine: 'fast' | 'relate',
  settings: PerFrameSetting[]
): Promise<{pairs: [number, number][]; matrices: string[]}[]> {
  const graph = new GPUCommandGraph(device, {id: 'per-frame-join'});
  const buffers: Buffer[] = [];
  const left = createGeometry(device, graph, 'left', leftKind, lefts, buffers);
  const right = createGeometry(device, graph, 'right', rightKind, rights, buffers);
  const capacity = Math.max(lefts.length * rights.length, 1);
  const patternSlots = 4;
  const patternBuffer = new GPUParameterBuffer(device, {
    id: 'pattern',
    format: 'uint32',
    length: patternSlots * 2
  });
  const distanceBuffer = new GPUParameterBuffer(device, {
    id: 'distance',
    format: 'float32',
    length: 1
  });
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const leftIds = output('left-ids', capacity);
  const rightIds = output('right-ids', capacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const relate = output('relate', capacity);
  const isRelate = predicate === 'relate';
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate,
      pattern: isRelate ? patternBuffer.importToGraph(graph) : undefined,
      distance: predicate === 'dwithin' ? distanceBuffer.importToGraph(graph) : undefined,
      engine: isRelate ? undefined : engine,
      candidateCapacity: capacity,
      pairs: {
        leftIds: leftIds.view,
        rightIds: rightIds.view,
        count: count.view,
        overflow: overflow.view
      },
      relate: predicate === 'dwithin' ? undefined : relate.view
    })
  );
  const compiled = graph.compile();
  const results: {pairs: [number, number][]; matrices: string[]}[] = [];
  for (const setting of settings) {
    if (setting.pattern !== undefined) {
      patternBuffer.write(packGPUSpatialRelatePattern(setting.pattern, patternSlots));
    }
    if (setting.distance !== undefined) {
      distanceBuffer.write(Float32Array.of(setting.distance));
    }
    submitGraph(device, compiled, undefined);
    const [countValue] = await readUint32(count.buffer, 1);
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    const lefted = await readUint32(leftIds.buffer, capacity);
    const righted = await readUint32(rightIds.buffer, capacity);
    const matrices = await readUint32(relate.buffer, capacity);
    results.push({
      pairs: lefted.slice(0, countValue).map((row, slot) => [row, righted[slot]]),
      matrices: matrices.slice(0, countValue).map(formatGPUSpatialRelate)
    });
  }
  compiled.destroy();
  patternBuffer.destroy();
  distanceBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return results;
}

it('GPUSpatialPredicateJoin per-frame relate pattern equals the compile-time pattern', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {lefts, rights} = getScene('polygons', 'polygons');
  const settings: PerFrameSetting[] = [
    {pattern: 'T*T***T**'},
    {pattern: 'T*****FF*'},
    {pattern: ['T*F**F***', 'FT*******']},
    {pattern: INTERSECTS_PATTERNS},
    {pattern: 'T*T***T**'}
  ];
  const results = await runPerFrame(
    device,
    'polygons',
    lefts,
    'polygons',
    rights,
    'relate',
    'relate',
    settings
  );
  let total = 0;
  for (const [index, setting] of settings.entries()) {
    const expected = await runJoin(device, 'polygons', lefts, 'polygons', rights, {
      predicate: 'relate',
      pattern: setting.pattern,
      relate: true
    });
    expect(results[index].pairs, JSON.stringify(setting.pattern)).toEqual(expected.pairs);
    expect(results[index].matrices).toEqual(expected.matrices);
    total += expected.pairs.length;
  }
  expect(total).toBeGreaterThan(20);
  // The first and last settings repeat one pattern: a rewrite round-trips.
  expect(results[4]).toEqual(results[0]);
  expect(results[0].pairs).not.toEqual(results[1].pairs);
  // Packing rejects what the compile-time pattern rejects.
  expect(() => packGPUSpatialRelatePattern('FF*FF****', 2)).toThrow(/disjoint/);
  expect(() => packGPUSpatialRelatePattern(['T*****FF*', 'T*F**F***', '*T*******'], 2)).toThrow(
    /slots/
  );
});

it('GPUSpatialPredicateJoin per-frame dwithin distance equals the compile-time distance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [leftKind, rightKind] of [
    ['polygons', 'polygons'],
    ['lines', 'polygons'],
    ['points', 'lines']
  ] as const) {
    const {lefts, rights} = getScene(leftKind, rightKind);
    // The invalid values select nothing and a zero distance equals intersects.
    const distances = [0.5, 2, 0, 6, -1, Number.NaN, 2];
    for (const engine of ['fast', 'relate'] as const) {
      const results = await runPerFrame(
        device,
        leftKind,
        lefts,
        rightKind,
        rights,
        'dwithin',
        engine,
        distances.map(distance => ({distance}))
      );
      let total = 0;
      for (const [index, distance] of distances.entries()) {
        const label = `${leftKind}/${rightKind} ${engine} distance ${distance}`;
        if (distance < 0 || Number.isNaN(distance)) {
          expect(results[index].pairs, label).toEqual([]);
          continue;
        }
        const expected = await runJoin(device, leftKind, lefts, rightKind, rights, {
          predicate: 'dwithin',
          distance,
          engine
        });
        expect(results[index].pairs, label).toEqual(expected.pairs);
        total += expected.pairs.length;
      }
      expect(total, `${leftKind}/${rightKind} ${engine}`).toBeGreaterThan(0);
      expect(results[1].pairs.length).toBeGreaterThanOrEqual(results[0].pairs.length);
      expect(results[3].pairs.length).toBeGreaterThanOrEqual(results[1].pairs.length);
    }
  }
});
