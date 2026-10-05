// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView, type GraphVectorView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPU_SPATIAL_JOIN_NO_FEATURE} from '../../../src/geospatial/spatial-join';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsExtentStatistic,
  type GPUZonalStatisticsOutput,
  type GPUZonalStatisticsSumOrder
} from '../../../src/geospatial/zonal-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildPolygonFeatureArrays,
  createRandom,
  joinPointsInPolygons,
  POLYGON_FEATURES,
  type OraclePolygonFeature
} from '../spatial-join/spatial-join-oracle';
import {UNCERTAINTY_FEATURES, UNCERTAINTY_POINTS} from '../spatial-join/uncertainty-scene';
import {
  computeExtent,
  computePolygonFeatureArea,
  computeZonalStatistics,
  type ZonalOracleResult
} from './zonal-statistics-oracle';

type Point = [number, number];
type OutputKey = Exclude<keyof GPUZonalStatisticsOutput, 'extentStatistic'>;

const UINT32_OUTPUTS: OutputKey[] = [
  'counts',
  'valueCounts',
  'overflow',
  'uncertainCount',
  'pointFeatureRows'
];

type Fixture = {
  graph: GPUCommandGraph;
  compiled: ReturnType<GPUCommandGraph['compile']>;
  buffers: Buffer[];
  inputs: Record<string, Buffer>;
  outputs: Partial<Record<OutputKey, Buffer>>;
  featureCount: number;
  pointCount: number;
  run(): void;
  readFloat(key: OutputKey, length?: number): Promise<number[]>;
  readUint(key: OutputKey, length?: number): Promise<number[]>;
  destroy(): void;
};

type FixtureOptions = {
  features?: OraclePolygonFeature[];
  points: Point[];
  values?: number[];
  weights?: number[];
  areas?: number[];
  sumOrder?: GPUZonalStatisticsSumOrder;
  outputKeys: OutputKey[];
  extentStatistic?: GPUZonalStatisticsExtentStatistic;
  candidateCapacity?: number;
  includeBoundary?: boolean;
};

function createPolygonFixture(device: Device, options: FixtureOptions): Fixture {
  const features = options.features ?? POLYGON_FEATURES;
  const featureCount = features.length;
  const pointCount = options.points.length;
  const graph = new GPUCommandGraph(device, {id: 'zonal-graph'});
  const buffers: Buffer[] = [];
  const inputs: Record<string, Buffer> = {};
  const addInput = (name: string, values: Float32Array | Uint32Array): Buffer => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    inputs[name] = buffer;
    return buffer;
  };
  const arrays = buildPolygonFeatureArrays(features);
  const points = importGraphBuffer(
    graph,
    'points',
    addInput('points', Float32Array.from(options.points.flat())),
    'float32x2',
    pointCount
  );
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array | undefined,
    format: Format,
    length: number
  ) =>
    values ? importGraphBuffer(graph, name, addInput(name, values), format, length) : undefined;
  const outputs: Partial<Record<OutputKey, Buffer>> = {};
  const output: Record<string, unknown> = {extentStatistic: options.extentStatistic};
  for (const key of options.outputKeys) {
    const length =
      key === 'extent'
        ? 2
        : key === 'overflow' || key === 'uncertainCount'
          ? 1
          : key === 'pointFeatureRows'
            ? pointCount
            : featureCount;
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs[key] = buffer;
    output[key] = importGraphBuffer(
      graph,
      `out-${key}`,
      buffer,
      UINT32_OUTPUTS.includes(key) ? 'uint32' : 'float32',
      length
    );
  }
  graph.add(
    new GPUZonalStatistics({
      id: 'zonal',
      features: {
        kind: 'polygons',
        polygonPositions: importGraphBuffer(
          graph,
          'polygon-positions',
          addInput('polygonPositions', arrays.polygonPositions),
          'float32x2',
          arrays.polygonPositions.length / 2
        ),
        featureOffsets: view(
          'feature-offsets',
          arrays.featureOffsets,
          'uint32',
          arrays.featureOffsets.length
        )!,
        polygonOffsets: view(
          'polygon-offsets',
          arrays.polygonOffsets,
          'uint32',
          arrays.polygonOffsets.length
        )!,
        ringOffsets: view('ring-offsets', arrays.ringOffsets, 'uint32', arrays.ringOffsets.length)!,
        candidateCapacity: options.candidateCapacity ?? Math.max(pointCount * featureCount, 4),
        includeBoundary: options.includeBoundary
      },
      points,
      values: view(
        'values',
        options.values && Float32Array.from(options.values),
        'float32',
        pointCount
      ),
      weights: view(
        'weights',
        options.weights && Float32Array.from(options.weights),
        'float32',
        pointCount
      ),
      areas: view(
        'areas',
        options.areas && Float32Array.from(options.areas),
        'float32',
        featureCount
      ),
      sumOrder: options.sumOrder,
      output: output as GPUZonalStatisticsOutput
    })
  );
  const compiled = graph.compile();
  const readOutput = async (key: OutputKey, length: number | undefined, float: boolean) => {
    const buffer = outputs[key]!;
    const rows =
      length ??
      (key === 'extent'
        ? 2
        : key === 'overflow' || key === 'uncertainCount'
          ? 1
          : key === 'pointFeatureRows'
            ? pointCount
            : featureCount);
    return float ? readFloat32(buffer, rows) : readUint32(buffer, rows);
  };
  return {
    graph,
    compiled,
    buffers,
    inputs,
    outputs,
    featureCount,
    pointCount,
    run: () => submitGraph(device, compiled, undefined),
    readFloat: (key, length) => readOutput(key, length, true),
    readUint: (key, length) => readOutput(key, length, false),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Compares rows with a per-row absolute tolerance; NaN must match NaN. */
function expectRows(
  actual: number[],
  expected: number[],
  relativeTolerance: number,
  scales?: number[]
): void {
  expect(actual.length).toBe(expected.length);
  for (const [row, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(actual[row], `row ${row}`).toBeNaN();
    } else {
      const scale = scales ? scales[row] : Math.abs(value);
      expect(
        Math.abs(actual[row] - value),
        `row ${row}: ${actual[row]} vs ${value}`
      ).toBeLessThanOrEqual(relativeTolerance * scale + 1e-30);
    }
  }
}

type RandomScene = {
  features: OraclePolygonFeature[];
  points: Point[];
  values: number[];
  weights: number[];
};

/** Random grid-aligned rectangles (some holed or multi-part) and 1/8-grid points. */
function createRandomScene(seed: number, featureCount: number, pointCount: number): RandomScene {
  const random = createRandom(seed);
  const grid = (count: number, step: number) => Math.floor(random() * count) * step;
  const rectangle = (minX: number, minY: number, maxX: number, maxY: number, reverse: boolean) => {
    const ring = [
      [minX, minY],
      [maxX, minY],
      [maxX, maxY],
      [minX, maxY]
    ];
    return reverse ? ring.reverse() : ring;
  };
  const features: OraclePolygonFeature[] = [];
  for (let row = 0; row < featureCount; row++) {
    const minX = grid(112, 0.25);
    const minY = grid(112, 0.25);
    const width = 2 + grid(24, 0.25);
    const height = 2 + grid(24, 0.25);
    const rings = [rectangle(minX, minY, minX + width, minY + height, random() < 0.5)];
    if (random() < 0.4) {
      rings.push(
        rectangle(minX + 0.5, minY + 0.5, minX + width - 0.75, minY + height - 0.75, random() < 0.5)
      );
    }
    const feature: OraclePolygonFeature = [rings];
    if (random() < 0.3) {
      feature.push([rectangle(minX + width + 1, minY, minX + width + 3, minY + 2, false)]);
    }
    features.push(feature);
  }
  const points: Point[] = [];
  const values: number[] = [];
  const weights: number[] = [];
  for (let index = 0; index < pointCount; index++) {
    points.push([grid(320, 0.125) - 1, grid(320, 0.125) - 1]);
    const kind = random();
    values.push(
      kind < 0.05
        ? Number.NaN
        : kind < 0.07
          ? Number.POSITIVE_INFINITY
          : kind < 0.09
            ? Number.NEGATIVE_INFINITY
            : Math.fround(random() * 20 - 4)
    );
    const weightKind = random();
    weights.push(weightKind < 0.04 ? Number.NaN : Math.fround(random() * 3));
  }
  points[0] = [Number.NaN, 1];
  return {features, points, values, weights};
}

function getAreas(features: OraclePolygonFeature[]): number[] {
  return features.map(computePolygonFeatureArea);
}

function getOracleRows(scene: RandomScene) {
  return joinPointsInPolygons(scene.points, scene.features, true);
}

const ALL_OUTPUTS: OutputKey[] = [
  'counts',
  'valueCounts',
  'sums',
  'weightSums',
  'means',
  'minima',
  'maxima',
  'densities',
  'featureAreas',
  'pointFeatureRows',
  'overflow'
];

async function expectMatchesOracle(
  fixture: Fixture,
  oracle: ZonalOracleResult,
  options: {weighted: boolean; areas: number[]; rows: number[]; sumTolerance?: number}
): Promise<void> {
  const tolerance = options.sumTolerance ?? 2e-5;
  expect(await fixture.readUint('counts')).toEqual(oracle.counts);
  expect(await fixture.readUint('valueCounts')).toEqual(oracle.valueCounts);
  expectRows(await fixture.readFloat('sums'), oracle.sums, tolerance, oracle.absoluteSums);
  if (options.weighted) {
    expectRows(
      await fixture.readFloat('weightSums'),
      oracle.weightSums,
      tolerance,
      oracle.weightSums
    );
  }
  const meanScales = oracle.absoluteSums.map(
    (absoluteSum, row) =>
      absoluteSum /
      Math.max(options.weighted ? oracle.weightSums[row] : oracle.valueCounts[row], 1e-30)
  );
  expectRows(await fixture.readFloat('means'), oracle.means, tolerance, meanScales);
  expect(await fixture.readFloat('minima')).toEqual(oracle.minima.map(value => value));
  expect(await fixture.readFloat('maxima')).toEqual(oracle.maxima.map(value => value));
  expectRows(await fixture.readFloat('densities'), oracle.densities, 1e-6);
  expectRows(await fixture.readFloat('featureAreas'), options.areas, 1e-6);
  expect(await fixture.readUint('pointFeatureRows')).toEqual(options.rows);
  expect(await fixture.readUint('overflow')).toEqual([0]);
}

for (const sumOrder of ['atomic', 'sorted'] as const) {
  for (const weighted of [false, true]) {
    it(`GPUZonalStatistics matches the oracle on random polygons (${sumOrder}, ${weighted ? 'weighted' : 'unweighted'})`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) {
        return;
      }
      const scene = createRandomScene(11, 12, 3000);
      const joined = getOracleRows(scene);
      const areas = getAreas(scene.features);
      const fixture = createPolygonFixture(device, {
        ...scene,
        weights: weighted ? scene.weights : undefined,
        sumOrder,
        outputKeys: weighted ? ALL_OUTPUTS : ALL_OUTPUTS.filter(key => key !== 'weightSums')
      });
      fixture.run();
      const oracle = computeZonalStatistics({
        featureRows: joined.featureRows,
        featureCount: 12,
        values: scene.values,
        weights: weighted ? scene.weights : undefined,
        areas
      });
      expect(oracle.counts).toEqual(joined.counts);
      expect(oracle.counts.some(count => count > 50)).toBe(true);
      await expectMatchesOracle(fixture, oracle, {weighted, areas, rows: joined.featureRows});
      fixture.destroy();
    });
  }
}

it('GPUZonalStatistics sorted sums are bitwise reproducible and match atomic sums', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(23, 12, 3000);
  const options = {
    ...scene,
    outputKeys: ['sums', 'weightSums', 'means', 'counts'] as OutputKey[]
  };
  const sorted = createPolygonFixture(device, {...options, sumOrder: 'sorted'});
  sorted.run();
  const firstSums = await sorted.readUint('sums');
  const firstWeightSums = await sorted.readUint('weightSums');
  const firstMeans = await sorted.readUint('means');
  for (let repeat = 0; repeat < 3; repeat++) {
    // Scribble over the outputs to prove they are rewritten, then encode the same graph again.
    sorted.outputs.sums!.write(new Uint32Array(12).fill(0x3f800000));
    sorted.run();
    expect(await sorted.readUint('sums')).toEqual(firstSums);
    expect(await sorted.readUint('weightSums')).toEqual(firstWeightSums);
    expect(await sorted.readUint('means')).toEqual(firstMeans);
  }
  const atomic = createPolygonFixture(device, {...options, sumOrder: 'atomic'});
  atomic.run();
  const oracle = computeZonalStatistics({
    featureRows: getOracleRows(scene).featureRows,
    featureCount: 12,
    values: scene.values,
    weights: scene.weights
  });
  expectRows(await sorted.readFloat('sums'), oracle.sums, 2e-5, oracle.absoluteSums);
  expectRows(await atomic.readFloat('sums'), oracle.sums, 2e-5, oracle.absoluteSums);
  expectRows(await sorted.readFloat('weightSums'), oracle.weightSums, 2e-5, oracle.weightSums);
  sorted.destroy();
  atomic.destroy();
});

for (const statistic of [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'density'
] as GPUZonalStatisticsExtentStatistic[]) {
  it(`GPUZonalStatistics publishes the ${statistic} extent without its own output view`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createRandomScene(5, 12, 2000);
    const joined = getOracleRows(scene);
    const areas = getAreas(scene.features);
    const oracle = computeZonalStatistics({
      featureRows: joined.featureRows,
      featureCount: 12,
      values: scene.values,
      weights: scene.weights,
      areas
    });
    const statistics = {
      count: oracle.counts,
      sum: oracle.sums,
      mean: oracle.means,
      minimum: oracle.minima,
      maximum: oracle.maxima,
      density: oracle.densities
    };
    const fixture = createPolygonFixture(device, {
      ...scene,
      outputKeys: ['extent'],
      extentStatistic: statistic
    });
    fixture.run();
    const expected = computeExtent(statistics[statistic], oracle.counts);
    const actual = await fixture.readFloat('extent');
    if (['count', 'minimum', 'maximum'].includes(statistic)) {
      expect(actual).toEqual(expected);
    } else {
      expectRows(actual, expected, 1e-4, [Math.abs(expected[0]) + 1, Math.abs(expected[1]) + 1]);
    }
    fixture.destroy();
  });
}

it('GPUZonalStatistics computes polygon areas for holes, multipolygons, and far-from-origin rings', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const offset = 100000;
  const farSquare = (size: number): number[][] => [
    [offset, offset],
    [offset + size, offset],
    [offset + size, offset + size],
    [offset, offset + size]
  ];
  const features: OraclePolygonFeature[] = [
    ...POLYGON_FEATURES,
    [[farSquare(3.5)]],
    [
      [
        farSquare(8),
        farSquare(2)
          .map(([x, y]) => [x + 1, y + 1])
          .reverse()
      ]
    ],
    [
      [
        [
          [0, 0],
          [Number.NaN, 5],
          [6, 0],
          [6, 4],
          [0, 4]
        ]
      ]
    ]
  ];
  const fixture = createPolygonFixture(device, {
    features,
    points: [[0.5, 0.5]],
    outputKeys: ['featureAreas']
  });
  fixture.run();
  const expected = features.map(computePolygonFeatureArea);
  expect(expected.slice(0, 4)).toEqual([15, 16, 8, 0]);
  expect(expected[4]).toBe(12.25);
  expect(expected[5]).toBe(60);
  expectRows(await fixture.readFloat('featureAreas'), expected, 1e-6);
  fixture.destroy();
});

it('GPUZonalStatistics handles empty features, all-NaN values, and empty extents', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Fixture features: [holed square, square, multipolygon, empty]. Feature 1 only has NaN values.
  const points: Point[] = [
    [0.5, 0.5],
    [3, 3],
    [6, 1],
    [7, 3],
    [11, 1],
    [20, 20]
  ];
  const values = [2, 4, Number.NaN, Number.NEGATIVE_INFINITY, 8, 100];
  const outputKeys: OutputKey[] = [
    'counts',
    'valueCounts',
    'sums',
    'means',
    'minima',
    'maxima',
    'densities'
  ];
  const fixture = createPolygonFixture(device, {points, values, outputKeys});
  fixture.run();
  expect(await fixture.readUint('counts')).toEqual([2, 2, 1, 0]);
  expect(await fixture.readUint('valueCounts')).toEqual([2, 0, 1, 0]);
  expect(await fixture.readFloat('sums')).toEqual([6, 0, 8, 0]);
  const means = await fixture.readFloat('means');
  expect(means[0]).toBe(3);
  expect(means[1]).toBeNaN();
  expect(means[2]).toBe(8);
  expect(means[3]).toBeNaN();
  const minima = await fixture.readFloat('minima');
  expect([minima[0], minima[2]]).toEqual([2, 8]);
  expect(minima[1]).toBeNaN();
  expect(minima[3]).toBeNaN();
  // No areas were computed or given: densities need areas, so GPU areas run (feature 3 area 0).
  const densities = await fixture.readFloat('densities');
  expect(densities[0]).toBe(Math.fround(2 / 15));
  expect(densities[1]).toBe(Math.fround(2 / 16));
  expect(densities[3]).toBeNaN();
  fixture.destroy();

  // No point lands in any feature: the extent is [0, 0] for count and for value statistics.
  for (const extentStatistic of ['count', 'minimum'] as const) {
    const empty = createPolygonFixture(device, {
      points: [
        [50, 50],
        [60, 60]
      ],
      values: [1, 2],
      outputKeys: ['extent', 'counts'],
      extentStatistic
    });
    empty.outputs.extent!.write(new Float32Array([7, 9]));
    empty.run();
    expect(await empty.readFloat('extent')).toEqual([0, 0]);
    expect(await empty.readUint('counts')).toEqual([0, 0, 0, 0]);
    empty.destroy();
  }
  // Points are assigned but every value is NaN: the minimum extent has no qualifying feature.
  const nan = createPolygonFixture(device, {
    points: [[0.5, 0.5]],
    values: [Number.NaN],
    outputKeys: ['extent'],
    extentStatistic: 'minimum'
  });
  nan.run();
  expect(await nan.readFloat('extent')).toEqual([0, 0]);
  nan.destroy();
});

it('GPUZonalStatistics reports join overflow when candidateCapacity is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(3, 12, 500);
  const fixture = createPolygonFixture(device, {
    ...scene,
    candidateCapacity: 8,
    outputKeys: ['counts', 'overflow']
  });
  fixture.run();
  expect(await fixture.readUint('overflow')).toEqual([1]);
  fixture.destroy();
});

it('GPUZonalStatistics counts near-collinear points and reports zero uncertain pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const oracle = joinPointsInPolygons(UNCERTAINTY_POINTS, UNCERTAINTY_FEATURES, false);
  const fixture = createPolygonFixture(device, {
    features: UNCERTAINTY_FEATURES,
    points: UNCERTAINTY_POINTS,
    outputKeys: ['counts', 'pointFeatureRows', 'overflow', 'uncertainCount']
  });
  fixture.run();
  expect(await fixture.readUint('pointFeatureRows')).toEqual(oracle.featureRows);
  expect(await fixture.readUint('counts')).toEqual(oracle.counts);
  expect(await fixture.readUint('overflow')).toEqual([0]);
  expect(await fixture.readUint('uncertainCount')).toEqual([0]);
  fixture.destroy();
});

it('GPUZonalStatistics writes zero uncertainCount for feature rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createRowsFixture(device, {
    rows: [0, 1, 1, GPU_SPATIAL_JOIN_NO_FEATURE],
    featureCount: 2,
    outputKeys: ['counts', 'uncertainCount']
  });
  // Poison the output so the test proves the contributor writes it.
  fixture.outputs.uncertainCount!.write(new Uint32Array([7]));
  fixture.run();
  expect(await readUint32(fixture.outputs.counts!, 2)).toEqual([1, 2]);
  expect(await readUint32(fixture.outputs.uncertainCount!, 1)).toEqual([0]);
  fixture.destroy();
});

it('GPUZonalStatistics updates per frame without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const scene = createRandomScene(41, 12, 1000);
    const fixture = createPolygonFixture(device, {
      ...scene,
      sumOrder,
      outputKeys: ['counts', 'sums', 'weightSums', 'minima', 'maxima']
    });
    fixture.run();
    const first = await fixture.readFloat('sums');

    const changed = createRandomScene(42, 12, 1000);
    // Keep the polygons; replace points, values, and weights.
    fixture.inputs['points'].write(Float32Array.from(changed.points.flat()));
    fixture.inputs['values'].write(Float32Array.from(changed.values));
    fixture.inputs['weights'].write(Float32Array.from(changed.weights));
    fixture.run();
    const oracle = computeZonalStatistics({
      featureRows: joinPointsInPolygons(changed.points, scene.features, true).featureRows,
      featureCount: 12,
      values: changed.values,
      weights: changed.weights
    });
    expect(await fixture.readUint('counts')).toEqual(oracle.counts);
    const sums = await fixture.readFloat('sums');
    expect(sums).not.toEqual(first);
    expectRows(sums, oracle.sums, 2e-5, oracle.absoluteSums);
    expect(await fixture.readFloat('minima')).toEqual(oracle.minima);
    expect(await fixture.readFloat('maxima')).toEqual(oracle.maxima);
    fixture.destroy();
  }
});

function createRowsFixture(
  device: Device,
  options: {
    rows: number[];
    featureCount: number;
    values?: number[];
    weights?: number[];
    areas?: number[];
    sumOrder?: GPUZonalStatisticsSumOrder;
    rowChunks?: number[];
    valueChunks?: number[];
    outputKeys: OutputKey[];
    extentStatistic?: GPUZonalStatisticsExtentStatistic;
  }
) {
  const {rows, featureCount} = options;
  const graph = new GPUCommandGraph(device, {id: 'rows-graph'});
  const buffers: Buffer[] = [];
  const makeColumn = <Format extends 'uint32' | 'float32'>(
    name: string,
    data: Uint32Array | Float32Array,
    format: Format,
    chunkLengths?: number[]
  ): GraphDataView<Format> | GraphVectorView<Format> => {
    if (!chunkLengths) {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, data.length);
    }
    let start = 0;
    const chunks = chunkLengths.map((length, index) => {
      const slice = data.slice(start, start + length);
      start += length;
      const buffer = createInputBuffer(device, slice.length ? slice : new Uint32Array(1));
      buffers.push(buffer);
      return importGraphBuffer(graph, `${name}-${index}`, buffer, format, length);
    });
    return createVectorView(name, format, chunks);
  };
  const outputs: Partial<Record<OutputKey, Buffer>> = {};
  const output: Record<string, unknown> = {extentStatistic: options.extentStatistic};
  for (const key of options.outputKeys) {
    const length =
      key === 'extent' ? 2 : key === 'overflow' || key === 'uncertainCount' ? 1 : featureCount;
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs[key] = buffer;
    output[key] = importGraphBuffer(
      graph,
      `out-${key}`,
      buffer,
      UINT32_OUTPUTS.includes(key) ? 'uint32' : 'float32',
      length
    );
  }
  const values =
    options.values &&
    makeColumn('values', Float32Array.from(options.values), 'float32', options.valueChunks);
  const weights =
    options.weights &&
    makeColumn('weights', Float32Array.from(options.weights), 'float32', options.valueChunks);
  const areas =
    options.areas &&
    (makeColumn('areas', Float32Array.from(options.areas), 'float32') as GraphDataView<'float32'>);
  graph.add(
    new GPUZonalStatistics({
      id: 'zonal',
      features: {
        kind: 'feature-rows',
        pointFeatureRows: makeColumn('rows', Uint32Array.from(rows), 'uint32', options.rowChunks),
        featureCount
      },
      values,
      weights,
      areas,
      sumOrder: options.sumOrder,
      output: output as GPUZonalStatisticsOutput
    })
  );
  const compiled = graph.compile();
  return {
    outputs,
    run: () => submitGraph(device, compiled, undefined),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it('GPUZonalStatistics accepts chunked feature rows with differently chunked values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(77);
  const featureCount = 9;
  const pointCount = 1500;
  const rows = Array.from({length: pointCount}, () => {
    const draw = random();
    return draw < 0.1
      ? GPU_SPATIAL_JOIN_NO_FEATURE
      : draw < 0.15
        ? featureCount + 3
        : Math.floor(random() * featureCount);
  });
  const values = Array.from({length: pointCount}, () => {
    const draw = random();
    return draw < 0.05
      ? Number.NaN
      : draw < 0.07
        ? Number.POSITIVE_INFINITY
        : Math.fround(random() * 10 - 2);
  });
  const weights = Array.from({length: pointCount}, () => Math.fround(random() * 2));
  const areas = [4, 0, 2.5, Number.NaN, 8, 1, 16, -3, 0.5];
  const oracle = computeZonalStatistics({featureRows: rows, featureCount, values, weights, areas});
  const fixture = createRowsFixture(device, {
    rows,
    featureCount,
    values,
    weights,
    areas,
    rowChunks: [500, 0, 700, 300],
    valueChunks: [1000, 250, 250],
    outputKeys: [
      'counts',
      'valueCounts',
      'sums',
      'weightSums',
      'means',
      'minima',
      'maxima',
      'densities',
      'overflow'
    ]
  });
  fixture.run();
  const read = async (key: OutputKey, float: boolean) =>
    float
      ? readFloat32(fixture.outputs[key]!, featureCount)
      : readUint32(fixture.outputs[key]!, featureCount);
  expect(await read('counts', false)).toEqual(oracle.counts);
  expect(await read('valueCounts', false)).toEqual(oracle.valueCounts);
  expectRows(await read('sums', true), oracle.sums, 2e-5, oracle.absoluteSums);
  expectRows(await read('weightSums', true), oracle.weightSums, 2e-5, oracle.weightSums);
  expect(await read('minima', true)).toEqual(oracle.minima);
  expect(await read('maxima', true)).toEqual(oracle.maxima);
  expectRows(
    await read('means', true),
    oracle.means,
    2e-5,
    oracle.absoluteSums.map((sum, row) => sum / Math.max(oracle.weightSums[row], 1e-30))
  );
  expectRows(await read('densities', true), oracle.densities, 1e-6);
  expect(oracle.densities[1]).toBeNaN();
  expect(oracle.densities[3]).toBeNaN();
  expect(oracle.densities[7]).toBeNaN();
  expect(await readUint32(fixture.outputs.overflow!, 1)).toEqual([0]);
  fixture.destroy();
});

it('GPUZonalStatistics sorted mode works with caller feature rows and many features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(5);
  const featureCount = 600;
  const pointCount = 4000;
  const rows = Array.from({length: pointCount}, () =>
    random() < 0.1 ? GPU_SPATIAL_JOIN_NO_FEATURE : Math.floor(random() * featureCount)
  );
  const values = Array.from({length: pointCount}, () => Math.fround(random() * 10 + 1));
  const oracle = computeZonalStatistics({featureRows: rows, featureCount, values});
  const fixture = createRowsFixture(device, {
    rows,
    featureCount,
    values,
    sumOrder: 'sorted',
    outputKeys: ['sums', 'means', 'counts']
  });
  fixture.run();
  const sums = await readUint32(fixture.outputs.sums!, featureCount);
  expectRows(
    await readFloat32(fixture.outputs.sums!, featureCount),
    oracle.sums,
    2e-5,
    oracle.absoluteSums
  );
  fixture.run();
  expect(await readUint32(fixture.outputs.sums!, featureCount)).toEqual(sums);
  expectRows(
    await readFloat32(fixture.outputs.means!, featureCount),
    oracle.means,
    2e-5,
    oracle.absoluteSums.map((sum, row) => sum / Math.max(oracle.valueCounts[row], 1))
  );
  expect(await readUint32(fixture.outputs.counts!, featureCount)).toEqual(oracle.counts);
  fixture.destroy();
});

it('GPUZonalStatistics handles zero points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const graph = new GPUCommandGraph(device, {id: 'empty-graph'});
    const buffers: Buffer[] = [];
    const named: Record<string, Buffer> = {};
    const input = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      named[name] = buffer;
      return importGraphBuffer(graph, name, buffer, format, length);
    };
    const output = {
      counts: input('counts', 'uint32', 3),
      sums: input('sums', 'float32', 3),
      means: input('means', 'float32', 3),
      extent: input('extent', 'float32', 2)
    };
    graph.add(
      new GPUZonalStatistics({
        id: 'zonal',
        features: {
          kind: 'feature-rows',
          pointFeatureRows: input('rows', 'uint32', 0),
          featureCount: 3
        },
        values: input('values', 'float32', 0),
        sumOrder,
        output: {...output, extentStatistic: 'sum'}
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const {
      counts: countsBuffer,
      sums: sumsBuffer,
      means: meansBuffer,
      extent: extentBuffer
    } = named;
    expect(await readUint32(countsBuffer, 3)).toEqual([0, 0, 0]);
    expect(await readFloat32(sumsBuffer, 3)).toEqual([0, 0, 0]);
    for (const mean of await readFloat32(meansBuffer, 3)) expect(mean).toBeNaN();
    expect(await readFloat32(extentBuffer, 2)).toEqual([0, 0]);
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
