// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPULonLatPolygonPreparation,
  getGPULonLatPolygonPreparationParameterValues
} from '../../../src/gpu-spatial-analysis/lon-lat-polygons';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {getLinearInterpolationErrorBound} from '../line-segmentize/line-segmentize-oracle';
import {
  createLonLatPolygonFixtureViews,
  LON_LAT_POLYGON_FIXTURES,
  FIJI_TRANSMERIDIAN,
  ARCTIC_RING,
  GREAT_CIRCLE_60_DEGREE_EDGE,
  ANTARCTICA_NATURAL_EARTH,
  CHICAGO_SQUARE,
  AMBIGUOUS_180_EDGE
} from './fixtures';
import {
  prepareLonLatPolygons,
  pointInLonLatPolygonPrepared,
  interpolateGreatCircle,
  getLonLatFeatures,
  flattenLonLatFeatures,
  densifyRing,
  type LonLatPolygonData,
  type LonLatPreparationOptions,
  type LonLatPosition
} from './lon-lat-polygon-oracle';

function createFixture(
  device: Device,
  input: LonLatPolygonData,
  options: LonLatPreparationOptions = {},
  queryPoints?: LonLatPosition[]
) {
  const rounded = {...input, positions: Float64Array.from(input.positions, Math.fround)};
  const graph = new GPUCommandGraph(device);
  const source = createLonLatPolygonFixtureViews(device, graph, rounded);
  const buffers: Buffer[] = [];
  const outputBuffers: Record<string, Buffer> = {};
  const capacity = options.positionCapacity ?? 4096;
  function output<Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) {
    const buffer = createOutputBuffer(
      device,
      Math.max(1, length * (format === 'float32x2' ? 2 : 1))
    );
    buffers.push(buffer);
    outputBuffers[name] = buffer;
    return importGraphBuffer(graph, name, buffer, format, length);
  }
  const fields = {
    positions: output('positions', 'float32x2', capacity),
    featureOffsets: output('featureOffsets', 'uint32', input.featureOffsets.length),
    polygonOffsets: output('polygonOffsets', 'uint32', 2 * input.polygonOffsets.length - 1),
    ringOffsets: output('ringOffsets', 'uint32', 2 * input.ringOffsets.length - 1),
    ringFlags: output('ringFlags', 'uint32', input.ringOffsets.length - 1),
    status: {
      count: output('count', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      requiredCount: output('requiredCount', 'uint32', 1),
      invalidCount: output('invalidCount', 'uint32', 1),
      approximation: output('approximation', 'uint32', 1)
    }
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 4
  });
  const preparation = new GPULonLatPolygonPreparation({
    positions: source.positions,
    featureOffsets: source.featureOffsets,
    polygonOffsets: source.polygonOffsets,
    ringOffsets: source.ringOffsets,
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'none',
      edgeModel: options.edgeModel ?? 'linear'
    },
    parameters: parameters.importToGraph(graph),
    output: fields,
    maximumPieceCount: options.maximumPieceCount ?? 1024,
    antimeridian: options.antimeridian,
    poleRings: options.poleRings
  });
  graph.add(preparation);
  if (queryPoints) {
    const pointBuffer = createInputBuffer(device, Float32Array.from(queryPoints.flat()));
    buffers.push(pointBuffer);
    const points = importGraphBuffer(
      graph,
      'query-points',
      pointBuffer,
      'float32x2',
      queryPoints.length
    );
    const polygons = {
      kind: 'polygons' as const,
      positions: fields.positions,
      featureOffsets: fields.featureOffsets,
      polygonOffsets: fields.polygonOffsets,
      ringOffsets: fields.ringOffsets
    };
    graph.add(
      new GPUPointInPolygonJoin({
        id: 'prepared-join',
        spatialContext: preparation.spatialContext,
        points: {kind: 'points', positions: points},
        polygons,
        candidateCapacity: queryPoints.length * input.featureOffsets.length,
        output: {
          pointFeatureIds: output('pointFeatureIds', 'uint32', queryPoints.length),
          featureCounts: output('featureCounts', 'uint32', input.featureOffsets.length - 1),
          status: {overflow: output('joinOverflow', 'uint32', 1)}
        }
      })
    );
    graph.add(
      new GPUZonalStatistics({
        id: 'prepared-zonal',
        spatialContext: preparation.spatialContext,
        positions: points,
        features: {
          ...polygons,
          candidateCapacity: queryPoints.length * input.featureOffsets.length
        },
        output: {
          counts: output('zonalCounts', 'uint32', input.featureOffsets.length - 1),
          status: {overflow: output('zonalOverflow', 'uint32', 1)}
        }
      })
    );
  }
  const compiled = graph.compile();
  return {
    rounded,
    async run(maximumSegmentDegrees = options.maximumSegmentDegrees) {
      parameters.write(
        getGPULonLatPolygonPreparationParameterValues(options.edgeModel ?? 'linear', {
          maximumSegmentDegrees
        })
      );
      submitGraph(device, compiled, undefined);
      const result: Record<string, number[]> = {};
      for (const [name, buffer] of Object.entries(outputBuffers)) {
        result[name] =
          name === 'positions'
            ? await readFloat32(buffer, capacity * 2)
            : await readUint32(buffer, buffer.byteLength / 4);
      }
      return result;
    },
    destroy() {
      compiled.destroy();
      parameters.destroy();
      source.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  };
}

function expectPrepared(
  result: Record<string, number[]>,
  input: LonLatPolygonData,
  options: LonLatPreparationOptions = {}
) {
  const expected = prepareLonLatPolygons(input, options);
  for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets', 'ringFlags'] as const) {
    expect(result[name].slice(0, expected[name].length), name).toEqual(Array.from(expected[name]));
  }
  for (const [name, value] of Object.entries(expected.status)) {
    expect(result[name][0], name).toBe(value);
  }
  const tolerance = options.edgeModel === 'great-circle' ? 180e-6 / Math.PI : undefined;
  const rings = getLonLatFeatures(input).flat(2);
  const originals = rings.flat();
  const interiors =
    options.edgeModel !== 'great-circle' && (options.maximumSegmentDegrees ?? 0) > 0
      ? rings.flatMap(ring =>
          densifyRing(ring, 'linear', options.maximumSegmentDegrees, options.maximumPieceCount)
            .vertices.filter(vertex => vertex.fraction > 0)
            .map(vertex => ({
              ...vertex,
              start: ring[vertex.sourceRow],
              end: ring[(vertex.sourceRow + 1) % ring.length]
            }))
        )
      : [];
  for (let index = 0; index < expected.positions.length; index++) {
    const row = index - (index % 2);
    const longitude = expected.positions[row];
    const latitude = expected.positions[row + 1];
    const kept = originals.some(position => position[0] === longitude && position[1] === latitude);
    if (kept || Math.abs(latitude) === 90) {
      expect(result['positions'][index], `kept/cap position ${index}`).toBe(
        expected.positions[index]
      );
      continue;
    }
    const interior =
      Math.abs(longitude) !== 180
        ? interiors.find(
            vertex =>
              Math.abs(vertex.wrapped[0] - longitude) < 1e-10 &&
              Math.abs(vertex.wrapped[1] - latitude) < 1e-10
          )
        : undefined;
    // Seam vertices retain the audit's one-ULP bound. Linear densified interiors instead
    // budget fraction error <=5u (division <=2.5 ULP) and each lerp operation, u=2^-24:
    // longitude <=u*(|rawStep|+8*|wrappedStep|+|start|)/(1-8u), also covering mix.
    // Position 34 in Antarctica is interior (-10,-71), from (-60,-76)->(0,-70), t=5/6;
    // cancellation makes one ULP of the result smaller than the interpolation error.
    // fround(-60 + 60*fround(5*fround(1/6))) is -9.999998092651367 (error 2^-19);
    // the derived longitude bound is 3.5762803918268165e-5 degrees for this edge.
    const magnitude = Math.abs(expected.positions[index]);
    const ulp = magnitude === 0 ? 2 ** -149 : 2 ** (Math.floor(Math.log2(magnitude)) - 23);
    expect(
      Math.abs(result['positions'][index] - expected.positions[index]),
      `position ${index}`
    ).toBeLessThanOrEqual(
      interior
        ? getLinearInterpolationErrorBound(
            interior.start[index % 2],
            interior.end[index % 2],
            index % 2 === 0
          )
        : (tolerance ?? ulp)
    );
  }
  return expected;
}

for (const [name, input] of Object.entries(LON_LAT_POLYGON_FIXTURES)) {
  it(`prepares ${name} like the binary64 oracle with stable fixed topology`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(device, input);
    try {
      const result = await fixture.run();
      expectPrepared(result, fixture.rounded);
      if (name === 'ANTARCTICA_NATURAL_EARTH') {
        expect(result['positions'].slice(0, input.positions.length)).toEqual(
          Array.from(input.positions)
        );
        expect(result['ringFlags'][0]).toBe(0);
      }
      if (name === 'CHICAGO_SQUARE') {
        expect(result['ringOffsets'].slice(0, 3)).toEqual([0, 4, 4]);
      }
      if (name === 'FIJI_TRANSMERIDIAN') {
        const output = result['positions'].slice(0, 2 * result['count'][0]);
        for (let row = 0; row < fixture.rounded.positions.length / 2; row++) {
          expect(
            output.some(
              (longitude, index) =>
                index % 2 === 0 &&
                Object.is(longitude, fixture.rounded.positions[2 * row]) &&
                Object.is(output[index + 1], fixture.rounded.positions[2 * row + 1])
            )
          ).toBe(true);
        }
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('preserves fractional kept longitudes across both seam copies without an unwrapped f32 round trip', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = flattenLonLatFeatures([
    [
      [
        [
          [177.12345, -19.12345],
          [-178.12345, -19.12345],
          [-178.12345, -16.12345],
          [177.12345, -16.12345]
        ]
      ]
    ]
  ]);
  const fixture = createFixture(device, input);
  try {
    const result = await fixture.run();
    expectPrepared(result, fixture.rounded);
    for (const point of getLonLatFeatures(fixture.rounded).flat(3)) {
      expect(
        result['positions'].some(
          (longitude, index) =>
            index % 2 === 0 &&
            Object.is(longitude, point[0]) &&
            Object.is(result['positions'][index + 1], point[1])
        )
      ).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('densifies a 60-degree great-circle edge within 1e-6 radians of slerp and the corrected oracle deviation bound', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {edgeModel: 'great-circle'} as const;
  const fixture = createFixture(device, GREAT_CIRCLE_60_DEGREE_EDGE, options);
  try {
    const result = await fixture.run();
    expectPrepared(result, fixture.rounded, options);
    let maximumDeviation = 0;
    for (let piece = 0; piece < 600; piece++) {
      const first = interpolateGreatCircle([-30, 45], [30, 45], piece / 600);
      const last = interpolateGreatCircle([-30, 45], [30, 45], (piece + 1) / 600);
      const midpoint = interpolateGreatCircle([-30, 45], [30, 45], (piece + 0.5) / 600);
      const linearLatitude =
        first[1] + (last[1] - first[1]) * ((midpoint[0] - first[0]) / (last[0] - first[0]));
      maximumDeviation = Math.max(
        maximumDeviation,
        ((Math.abs(midpoint[1] - linearLatitude) * Math.PI) / 180) * 6371008.8
      );
    }
    expect(maximumDeviation).toBeGreaterThan(1.2);
    expect(maximumDeviation).toBeLessThanOrEqual(1.4);
    // A per-frame zero step changes the result and raises approximation without recompiling.
    expectPrepared(await fixture.run(0), fixture.rounded, {...options, maximumSegmentDegrees: 0});
    expectPrepared(await fixture.run(0.1), fixture.rounded, options);
  } finally {
    fixture.destroy();
  }
});

for (const [name, input, options] of [
  // Capacity four fits all intermediate Fiji streams, so merge-only overflow stays exact.
  ['overflow', FIJI_TRANSMERIDIAN, {positionCapacity: 4}],
  ['clamp', GREAT_CIRCLE_60_DEGREE_EDGE, {edgeModel: 'great-circle', maximumPieceCount: 4}],
  ['reject crossing', FIJI_TRANSMERIDIAN, {antimeridian: 'reject'}],
  ['reject pole', ARCTIC_RING, {poleRings: 'reject'}],
  ['pole edge densification', ANTARCTICA_NATURAL_EARTH, {maximumSegmentDegrees: 10}],
  ['linear densification', CHICAGO_SQUARE, {maximumSegmentDegrees: 0.02}],
  [
    'invalid complete feature',
    flattenLonLatFeatures([
      getLonLatFeatures(FIJI_TRANSMERIDIAN)[0].concat(getLonLatFeatures(AMBIGUOUS_180_EDGE)[0])
    ]),
    {}
  ],
  [
    'invalid latitude',
    flattenLonLatFeatures([
      [
        [
          [
            [0, 91],
            [1, 0],
            [0, 1]
          ]
        ]
      ]
    ]),
    {}
  ],
  [
    'nonfinite',
    flattenLonLatFeatures([
      [
        [
          [
            [NaN, 1],
            [1, 0],
            [0, 1]
          ]
        ]
      ]
    ]),
    {}
  ],
  [
    'equatorial pole',
    flattenLonLatFeatures([
      [
        [
          [
            [-180, -10],
            [-90, 10],
            [0, -10],
            [90, 10]
          ]
        ]
      ]
    ]),
    {}
  ],
  [
    'multiple features',
    flattenLonLatFeatures([
      ...getLonLatFeatures(FIJI_TRANSMERIDIAN),
      ...getLonLatFeatures(CHICAGO_SQUARE)
    ]),
    {}
  ]
] as [string, LonLatPolygonData, LonLatPreparationOptions][]) {
  it(`preparation handles ${name} and keeps offsets valid`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(device, input, options);
    try {
      const result = await fixture.run();
      expectPrepared(result, fixture.rounded, options);
      expect(
        result['ringOffsets'].every(
          (offset, index, offsets) =>
            offset <= result['count'][0] && (index === 0 || offset >= offsets[index - 1])
        )
      ).toBe(true);
    } finally {
      fixture.destroy();
    }
  });
}

for (const [name, input] of [
  ['Fiji', FIJI_TRANSMERIDIAN],
  ['Arctic', ARCTIC_RING]
] as const) {
  it(`prepared ${name} feeds point-in-polygon and zonal counts on a seam/pole grid`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const points: LonLatPosition[] = [
      -180, -179.999, -179, -178, -90, 0, 90, 177, 179, 179.999, 180
    ].flatMap(longitude =>
      [-20, -18, -16, 69, 71, 75, 89].map(
        latitude => [Math.fround(longitude), latitude] as LonLatPosition
      )
    );
    const fixture = createFixture(device, input, {}, points);
    try {
      const result = await fixture.run();
      const prepared = expectPrepared(result, fixture.rounded);
      const expected = points.map(point => pointInLonLatPolygonPrepared(point, prepared));
      expect(result['pointFeatureIds'].map(feature => feature !== 0xffffffff)).toEqual(expected);
      const count = expected.filter(Boolean).length;
      expect(result['featureCounts'][0]).toBe(count);
      expect(result['zonalCounts'][0]).toBe(count);
      expect(result['joinOverflow'][0]).toBe(0);
      expect(result['zonalOverflow'][0]).toBe(0);
    } finally {
      fixture.destroy();
    }
  });
}

for (const [name, input, options] of [
  ['dense', FIJI_TRANSMERIDIAN, {positionCapacity: 3}],
  ['empty capacity', FIJI_TRANSMERIDIAN, {positionCapacity: 0}],
  ['cap', ARCTIC_RING, {positionCapacity: 4}],
  [
    'main right clip',
    flattenLonLatFeatures([
      [
        [
          [
            [177, -19],
            [-178, -19],
            [177, -16]
          ]
        ]
      ]
    ]),
    {positionCapacity: 3}
  ],
  [
    'east left clip',
    flattenLonLatFeatures([
      [
        [
          [
            [177, -19],
            [-178, -19],
            [-178, -16]
          ]
        ]
      ]
    ]),
    {positionCapacity: 3}
  ],
  [
    'great-circle dense',
    GREAT_CIRCLE_60_DEGREE_EDGE,
    {
      positionCapacity: 3,
      edgeModel: 'great-circle',
      maximumSegmentDegrees: 10
    }
  ]
] as [string, LonLatPolygonData, LonLatPreparationOptions][]) {
  it(`reports ${name} intermediate overflow and succeeds after one resize`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(device, input, options);
    let requiredCount = 0;
    try {
      const result = await fixture.run();
      const exact = prepareLonLatPolygons(fixture.rounded, {
        ...options,
        positionCapacity: undefined
      });
      expect(result['overflow'][0]).toBe(1);
      requiredCount = result['requiredCount'][0];
      expect(requiredCount).toBeGreaterThanOrEqual(exact.status.requiredCount);
      expect(requiredCount).toBeGreaterThan(options.positionCapacity!);
      expect(result['count'][0]).toBeLessThanOrEqual(options.positionCapacity!);
      expect(result['ringOffsets'][0]).toBe(0);
      expect(result['ringOffsets'].at(-1)).toBe(result['count'][0]);
      expect(
        result['ringOffsets'].every(
          (offset, index, offsets) =>
            offset <= result['count'][0] && (index === 0 || offset >= offsets[index - 1])
        )
      ).toBe(true);
      // Re-encoding must reset all diagnostics, including intermediate overflow totals.
      expect(await fixture.run()).toEqual(result);
    } finally {
      fixture.destroy();
    }
    const resized = createFixture(device, input, {...options, positionCapacity: requiredCount});
    try {
      const result = await resized.run();
      expectPrepared(result, resized.rounded, {...options, positionCapacity: requiredCount});
      expect(result['overflow'][0]).toBe(0);
    } finally {
      resized.destroy();
    }
  });
}
