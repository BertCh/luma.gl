// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUPointInPolygonJoin,
  GPUSpatialJoinPrepared
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {HALF_OPEN_SCENES} from './half-open-fixtures';
import {
  buildPolygonFeatureArrays,
  createRandom,
  joinPointsInPolygons,
  NO_FEATURE,
  POLYGON_FEATURES,
  POLYGON_POINTS,
  type OraclePolygonFeature
} from './spatial-join-oracle';
import {createPolygonJoinRun} from './spatial-join-sort-utils';

type Point = [number, number];
type Rule = 'half-open' | 'closed' | 'open';

/** Compiles one join and reads ids, counts, overflow and candidate count. */
async function runJoin(
  device: Device,
  features: OraclePolygonFeature[],
  points: Point[],
  options: {edgeSlabs: boolean; boundaryRule: Rule; candidateCapacity?: number}
) {
  const run = createPolygonJoinRun(device, features, points, {
    spatialSort: false,
    candidateCapacity: options.candidateCapacity ?? Math.max(points.length * 4, 64),
    boundaryRule: options.boundaryRule,
    edgeSlabs: options.edgeSlabs
  });
  const result = await run.readResult();
  run.destroy();
  return result;
}

/** Jagged "circle" with many vertices: shells with strong y-variation and a hole. */
function createJaggedPolygon(
  centerX: number,
  centerY: number,
  radius: number,
  vertexCount: number,
  seed: number
): OraclePolygonFeature[number] {
  const random = createRandom(seed);
  const ring = (scale: number, count: number) =>
    Array.from({length: count}, (_, index) => {
      const angle = (index / count) * Math.PI * 2;
      const wobble = 1 + 0.25 * Math.sin(17 * angle) + 0.1 * (random() - 0.5);
      return [
        Math.fround(centerX + Math.cos(angle) * radius * scale * wobble),
        Math.fround(centerY + Math.sin(angle) * radius * scale * wobble)
      ];
    });
  return [ring(1, vertexCount), ring(0.3, Math.max(8, Math.floor(vertexCount / 50)))];
}

for (const boundaryRule of ['half-open', 'closed', 'open'] as const) {
  it(`edgeSlabs equals the walk and the oracle on the PIP fixtures (${boundaryRule})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const off = await runJoin(device, POLYGON_FEATURES, POLYGON_POINTS, {
      edgeSlabs: false,
      boundaryRule
    });
    const on = await runJoin(device, POLYGON_FEATURES, POLYGON_POINTS, {
      edgeSlabs: true,
      boundaryRule
    });
    expect(on).toEqual(off);
    const oracle = joinPointsInPolygons(POLYGON_POINTS, POLYGON_FEATURES, boundaryRule);
    expect(on.featureRows).toEqual(oracle.featureRows);
    expect(on.counts).toEqual(oracle.counts);
  }, 120000);
}

it('edgeSlabs keeps the shapely half-open owners', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of Object.values(HALF_OPEN_SCENES)) {
    const on = await runJoin(device, scene.features, scene.points, {
      edgeSlabs: true,
      boundaryRule: 'half-open'
    });
    const off = await runJoin(device, scene.features, scene.points, {
      edgeSlabs: false,
      boundaryRule: 'half-open'
    });
    expect(on).toEqual(off);
    expect(on.featureRows.map(row => (row === NO_FEATURE ? null : row))).toEqual(scene.owners);
  }
}, 120000);

/** Slabbed scene: many multi-vertex parts with holes, vertex-aligned and random points. */
function createMultipartScene(): {features: OraclePolygonFeature[]; points: Point[]} {
  const random = createRandom(77);
  const features: OraclePolygonFeature[] = [];
  for (let feature = 0; feature < 30; feature++) {
    const parts = 1 + (feature % 5) * 8;
    features.push(
      Array.from({length: parts}, (_, part) =>
        createJaggedPolygon(
          (feature % 6) * 25 + part * 0.7,
          Math.floor(feature / 6) * 25,
          6 + random() * 3,
          24 + Math.floor(random() * 80),
          feature * 100 + part
        )
      )
    );
  }
  const points: Point[] = [];
  for (let index = 0; index < 20000; index++) {
    points.push([Math.fround(random() * 150), Math.fround(random() * 125)]);
  }
  // Points exactly on vertices and on slab-aligned heights.
  for (const feature of features.slice(0, 10)) {
    for (const [x, y] of feature[0][0].slice(0, 40)) {
      points.push([x, y], [x + 1, y]);
    }
  }
  points.push([Number.NaN, 1], [1, Number.POSITIVE_INFINITY]);
  return {features, points};
}

for (const boundaryRule of ['half-open', 'closed', 'open'] as const) {
  it(`edgeSlabs is bitwise identical on slabbed multipart scenes (${boundaryRule})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {features, points} = createMultipartScene();
    const off = await runJoin(device, features, points, {edgeSlabs: false, boundaryRule});
    const on = await runJoin(device, features, points, {edgeSlabs: true, boundaryRule});
    expect(on.overflow).toBe(off.overflow);
    expect(on.candidateCount).toBe(off.candidateCount);
    expect(on.featureRows).toEqual(off.featureRows);
    expect(on.counts).toEqual(off.counts);
    expect(off.featureRows.some(row => row !== NO_FEATURE)).toBe(true);
  }, 300000);
}

it('edgeSlabs keeps malformed offsets UNCERTAIN: the uncertain count matches the walk', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Feature 1 has a ring whose end exceeds the vertex count after the arrays are patched.
  const features = [POLYGON_FEATURES[0], POLYGON_FEATURES[1]];
  const arrays = buildPolygonFeatureArrays(features);
  const counts: number[] = [];
  for (const edgeSlabs of [false, true]) {
    const graph = new GPUCommandGraph(device, {id: 'edge-slabs-malformed'});
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        format === 'uint32' ? values.length : values.length / 2
      );
    };
    const ringOffsets = Uint32Array.from(arrays.ringOffsets);
    ringOffsets[ringOffsets.length - 1] += 10_000;
    const ids = createOutputBuffer(device, 4);
    const overflow = createOutputBuffer(device, 1);
    const uncertain = createOutputBuffer(device, 1);
    buffers.push(ids, overflow, uncertain);
    graph.add(
      new GPUPointInPolygonJoin({
        id: 'edge-slabs-malformed',
        points: {
          kind: 'points',
          positions: input('points', Float32Array.of(0.5, 0.5, 5, 1, 6, 2, 9, 9), 'float32x2')
        },
        polygons: {
          kind: 'polygons',
          positions: input('positions', arrays.polygonPositions, 'float32x2'),
          featureOffsets: input('feature-offsets', arrays.featureOffsets, 'uint32'),
          polygonOffsets: input('polygon-offsets', arrays.polygonOffsets, 'uint32'),
          ringOffsets: input('ring-offsets', ringOffsets, 'uint32')
        },
        candidateCapacity: 64,
        edgeSlabs,
        output: {
          pointFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', 4),
          status: {
            overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
            uncertainCount: importGraphBuffer(graph, 'uncertain', uncertain, 'uint32', 1)
          }
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    counts.push((await readUint32(uncertain, 1))[0], ...(await readUint32(ids, 4)));
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
  expect(counts.slice(0, 5)).toEqual(counts.slice(5));
}, 120000);

/** Large-polygon scene: 5 polygons of 50,000 vertices and 1e6 points. */
function createLargeScene(): {features: OraclePolygonFeature[]; points: Point[]} {
  const features: OraclePolygonFeature[] = Array.from({length: 5}, (_, index) => [
    createJaggedPolygon(index * 30, 0, 12, 50_000, 500 + index)
  ]);
  const random = createRandom(901);
  const points: Point[] = Array.from({length: 1_000_000}, () => [
    Math.fround(random() * 150 - 15),
    Math.fround(random() * 30 - 15)
  ]);
  return {features, points};
}

/** Prepared join run: the index (BVH and slabs) is built once, encodings only query. */
function createPreparedRun(
  device: Device,
  features: OraclePolygonFeature[],
  points: Point[],
  options: {edgeSlabs: boolean; candidateCapacity: number}
) {
  const graph = new GPUCommandGraph(device, {id: 'edge-slabs-prepared'});
  const arrays = buildPolygonFeatureArrays(features);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      format === 'uint32' ? values.length : values.length / 2
    );
  };
  const polygons = {
    kind: 'polygons',
    positions: input('positions', arrays.polygonPositions, 'float32x2'),
    featureOffsets: input('feature-offsets', arrays.featureOffsets, 'uint32'),
    polygonOffsets: input('polygon-offsets', arrays.polygonOffsets, 'uint32'),
    ringOffsets: input('ring-offsets', arrays.ringOffsets, 'uint32')
  } as const;
  const ids = createOutputBuffer(device, points.length);
  const overflow = createOutputBuffer(device, 1);
  buffers.push(ids, overflow);
  const prepared = new GPUSpatialJoinPrepared({
    id: 'edge-slabs-prepared',
    geometry: polygons,
    edgeSlabs: options.edgeSlabs
  });
  graph.add(prepared);
  graph.add(
    new GPUPointInPolygonJoin({
      id: 'edge-slabs-prepared-join',
      points: {
        kind: 'points',
        positions: input('points', Float32Array.from(points.flat()), 'float32x2')
      },
      polygons,
      prepared,
      candidateCapacity: options.candidateCapacity,
      output: {
        pointFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', points.length),
        status: {overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  return {
    encodeAndWait: async () => {
      submitGraph(device, compiled, undefined);
      await readUint32(overflow, 1);
    },
    read: async () => {
      submitGraph(device, compiled, undefined);
      return [...(await readUint32(ids, points.length))];
    },
    destroy: () => {
      compiled.destroy();
      prepared.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

const median = (values: number[]) =>
  [...values].sort((left, right) => left - right)[Math.floor(values.length / 2)];

it('A/B: walk vs edge slabs, 5 polygons x 50k vertices x 1e6 points, plain and prepared', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {features, points} = createLargeScene();
  const candidateCapacity = 1_500_000;
  const plain = {
    walk: createPolygonJoinRun(device, features, points, {
      spatialSort: true,
      candidateCapacity,
      boundaryRule: 'half-open',
      edgeSlabs: false
    }),
    slabs: createPolygonJoinRun(device, features, points, {
      spatialSort: true,
      candidateCapacity,
      boundaryRule: 'half-open',
      edgeSlabs: true
    })
  };
  const prepared = {
    walk: createPreparedRun(device, features, points, {edgeSlabs: false, candidateCapacity}),
    slabs: createPreparedRun(device, features, points, {edgeSlabs: true, candidateCapacity})
  };
  const walkResult = await plain.walk.readResult();
  const slabResult = await plain.slabs.readResult();
  expect(slabResult.overflow).toBe(0);
  expect(slabResult).toEqual(walkResult);
  expect(await prepared.slabs.read()).toEqual(walkResult.featureRows);
  expect(await prepared.walk.read()).toEqual(walkResult.featureRows);
  const lines: string[] = [];
  const runs: Record<string, {encodeAndWait: () => Promise<void>}> = {
    'plain walk': plain.walk,
    'plain slabs': plain.slabs,
    'prepared walk': prepared.walk,
    'prepared slabs': prepared.slabs
  };
  const samples: Record<string, number[]> = Object.fromEntries(
    Object.keys(runs).map(name => [name, []])
  );
  for (const run of Object.values(runs)) {
    for (let index = 0; index < 2; index++) await run.encodeAndWait();
  }
  for (let repeat = 0; repeat < 5; repeat++) {
    for (const name of Object.keys(runs)) {
      const timings: number[] = [];
      for (let index = 0; index < 5; index++) {
        const start = performance.now();
        await runs[name].encodeAndWait();
        timings.push(performance.now() - start);
      }
      samples[name].push(median(timings));
    }
  }
  for (const name of Object.keys(runs)) {
    lines.push(`${name.padEnd(16)} ${median(samples[name]).toFixed(2)} ms`);
  }
  plain.walk.destroy();
  plain.slabs.destroy();
  prepared.walk.destroy();
  prepared.slabs.destroy();
  // eslint-disable-next-line no-console
  console.log(`EDGE-SLABS-AB\n${lines.join('\n')}`);
}, 900000);
