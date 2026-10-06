// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
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
import {buildOracleArrays, type OracleFeature} from './spatial-predicate-oracle';
import {
  createRandomWalkLine,
  createSharedBoundaryLattice,
  createWobblyPolygon
} from './spatial-relate-realistic-scenes';
import {createRandom} from '../segment-intersection/segment-intersection-oracle';

/**
 * Relate-engine scaling benchmark on realistic (zip-code-like) polygons. Imported by the gated
 * `spatial-join-benchmark.spec.ts`, so it only runs with `LUMA_TEST_BROWSER_BENCHMARKS=true`.
 */

type Scenario = {
  name: string;
  leftKind: OracleFeature['kind'];
  lefts: OracleFeature[];
  rightKind: OracleFeature['kind'];
  rights: OracleFeature[];
  /** Candidate capacity; defaults to every pair. */
  capacity?: number;
};

type Row = {
  scenario: string;
  predicate: string;
  vertexPairs: number;
  candidates: number;
  matches: number;
  milliseconds: number;
  checksum: string;
};

const rows: Row[] = [];

function createScenarios(): Scenario[] {
  const random = createRandom(2026);
  const makeZips = (grid: number, minVertices: number, maxVertices: number) => {
    const zips: OracleFeature[] = [];
    for (let row = 0; row < grid; row++) {
      for (let column = 0; column < grid; column++) {
        const count = minVertices + Math.floor(random() * (maxVertices - minVertices + 1));
        zips.push(
          createWobblyPolygon(random, column * 2 + random(), row * 2 + random(), 1.1, count)
        );
      }
    }
    return zips;
  };
  const makeQueries = (count: number, minVertices: number, maxVertices: number, extent: number) =>
    Array.from({length: count}, () =>
      createWobblyPolygon(
        random,
        random() * extent,
        random() * extent,
        2.5,
        minVertices + Math.floor(random() * (maxVertices - minVertices + 1))
      )
    );
  const smallZips = makeZips(3, 40, 60);
  const smallQueries = makeQueries(2, 30, 50, 6);
  const zips = makeZips(5, 200, 1000);
  const queries = makeQueries(4, 50, 300, 10);
  const lines = Array.from({length: 6}, () =>
    createRandomWalkLine(random, random() * 10, random() * 10, 100, 0.4)
  );
  const smallRandom = createRandom(77);
  const smallFeatures = (count: number, extent: number, minSize: number, maxSize: number) =>
    Array.from({length: count}, (): OracleFeature => {
      const x = Math.fround(smallRandom() * extent);
      const y = Math.fround(smallRandom() * extent);
      const size = Math.fround(minSize + smallRandom() * (maxSize - minSize));
      return smallRandom() < 0.5
        ? {
            kind: 'polygons',
            polygons: [
              [
                [
                  [x, y],
                  [Math.fround(x + size), y],
                  [Math.fround(x + size), Math.fround(y + size)],
                  [x, Math.fround(y + size)]
                ]
              ]
            ]
          }
        : {
            kind: 'polygons',
            polygons: [
              [
                [
                  [x, y],
                  [Math.fround(x + size), y],
                  [x, Math.fround(y + size)]
                ]
              ]
            ]
          };
    });
  const mediumScenarios: Scenario[] = [8, 16, 32, 64].map(vertexCount => ({
    name: `medium 400x${vertexCount}v vs 40x${vertexCount}v`,
    leftKind: 'polygons',
    lefts: Array.from({length: 400}, () =>
      createWobblyPolygon(smallRandom, smallRandom() * 40, smallRandom() * 40, 1.2, vertexCount)
    ),
    rightKind: 'polygons',
    rights: Array.from({length: 40}, () =>
      createWobblyPolygon(smallRandom, smallRandom() * 40, smallRandom() * 40, 2.5, vertexCount)
    ),
    capacity: 1 << 14
  }));
  const manySmall = smallFeatures(3000, 50, 1, 3);
  const manyQueries = smallFeatures(300, 50, 3, 6);
  const longShared = createSharedBoundaryLattice(smallRandom, 3, 175, 2);
  const sharedSmall = createSharedBoundaryLattice(random, 4, 12, 2);
  const shared = createSharedBoundaryLattice(random, 5, 60, 2);
  return [
    {
      name: 'small 9x50v vs 2x40v',
      leftKind: 'polygons',
      lefts: smallZips,
      rightKind: 'polygons',
      rights: smallQueries
    },
    {
      name: 'zips 25x200-1000v vs 4x50-300v',
      leftKind: 'polygons',
      lefts: zips,
      rightKind: 'polygons',
      rights: queries
    },
    {
      name: 'zips 25x200-1000v vs 6 lines 100v',
      leftKind: 'polygons',
      lefts: zips,
      rightKind: 'lines',
      rights: lines
    },
    ...mediumScenarios,
    {
      name: 'equal+neighbours 9x700v vs same 9',
      leftKind: 'polygons',
      lefts: longShared.cells,
      rightKind: 'polygons',
      rights: longShared.cells
    },
    {
      name: 'one equal pair 700v',
      leftKind: 'polygons',
      lefts: [longShared.cells[4]],
      rightKind: 'polygons',
      rights: [longShared.cells[4]]
    },
    {
      name: 'one neighbour pair 700v',
      leftKind: 'polygons',
      lefts: [longShared.cells[4]],
      rightKind: 'polygons',
      rights: [longShared.cells[5]]
    },
    {
      name: 'many small 3000x4v vs 300x4v',
      leftKind: 'polygons',
      lefts: manySmall,
      rightKind: 'polygons',
      rights: manyQueries,
      capacity: 1 << 15
    },
    {
      name: 'zips 25x200-1000v vs 4x50-300v, capacity 65536',
      leftKind: 'polygons',
      lefts: zips,
      rightKind: 'polygons',
      rights: queries,
      capacity: 1 << 16
    },
    {
      name: 'shared-boundary small 16x~50v vs 4 blocks',
      leftKind: 'polygons',
      lefts: sharedSmall.cells,
      rightKind: 'polygons',
      rights: sharedSmall.blocks
    },
    {
      name: 'shared-boundary 25x240v vs 4 blocks 480v',
      leftKind: 'polygons',
      lefts: shared.cells,
      rightKind: 'polygons',
      rights: shared.blocks
    }
  ];
}

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

function countVertices(features: OracleFeature[]): number {
  let total = 0;
  for (const feature of features) {
    if (feature.kind === 'lines') {
      total += feature.vertices.length;
    } else if (feature.kind === 'polygons') {
      total += feature.polygons.flat().reduce((sum, ring) => sum + ring.length, 0);
    } else {
      total += 1;
    }
  }
  return total;
}

function hashNumbers(values: number[]): string {
  let hash = 2166136261;
  for (const value of values) {
    hash = Math.imul(hash ^ value, 16777619) >>> 0;
  }
  return hash.toString(16);
}

async function timePredicate(
  device: Device,
  scenario: Scenario,
  predicate: GPUSpatialPredicate,
  pattern: string | string[] | undefined,
  distance: number | undefined,
  wantsRelate: boolean
): Promise<void> {
  const graph = new GPUCommandGraph(device, {id: 'relate-benchmark'});
  const buffers: Buffer[] = [];
  const left = createGeometry(device, graph, 'left', scenario.leftKind, scenario.lefts, buffers);
  const right = createGeometry(
    device,
    graph,
    'right',
    scenario.rightKind,
    scenario.rights,
    buffers
  );
  const capacity = scenario.capacity ?? scenario.lefts.length * scenario.rights.length;
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
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      predicate,
      pattern,
      distance,
      candidateCapacity: capacity,
      pairs: {
        leftIds: leftIds.view,
        rightIds: rightIds.view,
        count: count.view,
        overflow: overflow.view
      },
      relate: wantsRelate ? relate.view : undefined
    })
  );
  const compiled = graph.compile();
  const run = async () => {
    submitGraph(device, compiled, undefined);
    return (await readUint32(count.buffer, 1))[0];
  };
  const start = performance.now();
  let matches = await run();
  const firstMilliseconds = performance.now() - start;
  let milliseconds = firstMilliseconds;
  // Slow runs are measured once; fast runs take the median of 5 after the first (compile) run.
  if (firstMilliseconds < 2000) {
    const timings: number[] = [];
    for (let index = 0; index < 5; index++) {
      const begin = performance.now();
      matches = await run();
      timings.push(performance.now() - begin);
    }
    timings.sort((a, b) => a - b);
    milliseconds = timings[2];
  }
  const ids = (await readUint32(leftIds.buffer, capacity)).slice(0, matches);
  const rightValues = (await readUint32(rightIds.buffer, capacity)).slice(0, matches);
  const matrices = wantsRelate ? (await readUint32(relate.buffer, capacity)).slice(0, matches) : [];
  rows.push({
    scenario: scenario.name,
    predicate: `${predicate}${wantsRelate ? '+matrix' : ''}${pattern ? ` ${Array.isArray(pattern) ? pattern.length : pattern}` : ''}`,
    vertexPairs: countVertices(scenario.lefts) * countVertices(scenario.rights),
    candidates: capacity,
    matches,
    milliseconds,
    checksum: hashNumbers([...ids, ...rightValues, ...matrices])
  });
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
}

/** Predicate, pattern, distance, whether the DE-9IM matrix output is requested (forces the relate engine). */
const BENCHMARK_PREDICATES: [
  GPUSpatialPredicate,
  string | string[] | undefined,
  number | undefined,
  boolean
][] = [
  ['intersects', undefined, undefined, false],
  ['contains', undefined, undefined, false],
  ['within', undefined, undefined, false],
  ['dwithin', undefined, 0.05, false],
  ['intersects', undefined, undefined, true],
  ['contains', undefined, undefined, true],
  ['touches', undefined, undefined, false],
  ['overlaps', undefined, undefined, false],
  ['relate', ['T********', '*T*******', '***T*****', '****T****'], undefined, true]
];

it('spatial relate benchmark: realistic polygons and lines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const only = (globalThis as {RELATE_BENCHMARK_SCENARIO?: string}).RELATE_BENCHMARK_SCENARIO;
  for (const scenario of createScenarios()) {
    if (only && !scenario.name.startsWith(only)) {
      continue;
    }
    for (const [predicate, pattern, distance, wantsRelate] of BENCHMARK_PREDICATES) {
      await timePredicate(device, scenario, predicate, pattern, distance, wantsRelate);
      const row = rows[rows.length - 1];
      // eslint-disable-next-line no-console
      console.log(
        `RELATE-BENCH ${row.scenario} | ${row.predicate} | candidates ${row.candidates} | matches ${row.matches} | ${row.milliseconds.toFixed(1)} ms | ${row.checksum}`
      );
    }
  }
}, 1_800_000);
