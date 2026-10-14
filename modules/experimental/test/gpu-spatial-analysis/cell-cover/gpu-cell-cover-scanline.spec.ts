// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUCellCover,
  type GPUCellCoverContainment
} from '../../../src/gpu-spatial-analysis/cell-cover';
import {joinCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtureJson from './fixtures/cell-cover-thin.json';
import {
  coverQuadbinOnCPU,
  createStarPolygon,
  flattenCoverFeatures,
  roundFeatures,
  type CoverFeature
} from './cell-cover-oracle';

// Oracles
// - H3: fixtures/cell-cover-thin.json, written by test/oracles/cells-cover-thin.py with h3-py
//   `h3shape_to_cells` (cell center inside the polygon). Thin strips, a buffered random-walk river,
//   a comb, polygons with holes and an island, multipolygons with far-apart parts and a strip at
//   the latitude limit, at resolutions 5, 7 and 9. Cells whose center lies within the fixture's
//   `centerMargin` of the boundary are listed as ambiguous and may differ (float32 centers).
// - Quadbin: `coverQuadbinOnCPU` (the bit-exact CPU mirror of the kernel) for every containment.
// - 'scanline' and 'auto' must equal 'bounds' exactly (cells, order, feature rows, core flags),
//   because the scanline stage only prunes candidates (see cell-cover-scanline.ts).

type RunOptions = {
  family: 'quadbin' | 'h3';
  resolution: number;
  containment?: GPUCellCoverContainment;
  candidateCapacity: number;
  outputCapacity: number;
  candidateEnumeration: 'bounds' | 'scanline' | 'auto';
  scanlineSlotCapacity?: number;
  edgeSlabs?: boolean;
};

type Run = {
  cells: bigint[];
  featureRows: number[];
  core: number[];
  count: number;
  total: number;
  overflow: number;
  candidateOverflow: number;
  candidateCount: number;
  /** Wall-clock milliseconds of one submit to read back, for the A/B logs. */
  milliseconds: number;
};

async function runCover(
  device: Device,
  features: CoverFeature[],
  options: RunOptions
): Promise<Run> {
  const arrays = flattenCoverFeatures(features);
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values.length ? values : new Uint32Array(2));
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const {outputCapacity} = options;
  const out = {
    featureIds: output(outputCapacity),
    cells: output(2 * outputCapacity),
    core: output(outputCapacity),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1),
    candidateCount: output(1),
    total: output(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-cover-scanline-graph'});
  graph.add(
    new GPUCellCover({
      family: options.family,
      resolution: options.resolution,
      containment: options.containment,
      positions: importGraphBuffer(
        graph,
        'positions',
        input(arrays.positions),
        'float32x2',
        arrays.positions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'feature-offsets',
        input(arrays.featureOffsets),
        'uint32',
        arrays.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygon-offsets',
        input(arrays.polygonOffsets),
        'uint32',
        arrays.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        input(arrays.ringOffsets),
        'uint32',
        arrays.ringOffsets.length
      ),
      candidateCapacity: options.candidateCapacity,
      candidateEnumeration: options.candidateEnumeration,
      scanlineSlotCapacity: options.scanlineSlotCapacity,
      edgeSlabs: options.edgeSlabs,
      output: {
        featureIds: importGraphBuffer(graph, 'out-ids', out.featureIds, 'uint32', outputCapacity),
        cells: importGraphBuffer(graph, 'out-cells', out.cells, 'uint32x2', outputCapacity),
        core: importGraphBuffer(graph, 'out-core', out.core, 'uint32', outputCapacity),
        status: {
          count: importGraphBuffer(graph, 'out-count', out.count, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'out-total', out.total, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'out-overflow', out.overflow, 'uint32', 1),
          candidateOverflow: importGraphBuffer(
            graph,
            'out-candidate-overflow',
            out.candidateOverflow,
            'uint32',
            1
          ),
          candidateCount: importGraphBuffer(
            graph,
            'out-candidate-count',
            out.candidateCount,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  const start = performance.now();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const milliseconds = performance.now() - start;
  const [overflow] = await readUint32(out.overflow, 1);
  const [candidateOverflow] = await readUint32(out.candidateOverflow, 1);
  const [candidateCount] = await readUint32(out.candidateCount, 1);
  const [total] = await readUint32(out.total, 1);
  const ids = await readUint32(out.featureIds, outputCapacity);
  const words = await readUint32(out.cells, 2 * outputCapacity);
  const core = await readUint32(out.core, outputCapacity);
  const run: Run = {
    cells: [],
    featureRows: ids.slice(0, count),
    core: core.slice(0, count),
    count,
    total,
    overflow,
    candidateOverflow,
    candidateCount,
    milliseconds
  };
  for (let row = 0; row < count; row++) {
    run.cells.push(joinCellKey(words[2 * row], words[2 * row + 1]));
  }
  compiled.destroy?.();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return run;
}

function expectSameRun(actual: Run, expected: Run, label: string) {
  expect(actual.overflow, `${label} overflow`).toBe(expected.overflow);
  expect(actual.total, `${label} total`).toBe(expected.total);
  expect(actual.count, `${label} count`).toBe(expected.count);
  expect(actual.featureRows, `${label} feature rows`).toEqual(expected.featureRows);
  expect(
    actual.cells.map(cell => cell.toString(16)),
    `${label} cells`
  ).toEqual(expected.cells.map(cell => cell.toString(16)));
  expect(actual.core, `${label} core`).toEqual(expected.core);
}

// ---------------------------------------------------------------------------------------------
// Quadbin: scanline equals bounds equals the CPU oracle.
// ---------------------------------------------------------------------------------------------

type Vertex = [number, number];

function createStrip(from: Vertex, to: Vertex, width: number): number[][] {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const nx = (-(to[1] - from[1]) / length) * (width / 2);
  const ny = ((to[0] - from[0]) / length) * (width / 2);
  return [
    [from[0] + nx, from[1] + ny],
    [to[0] + nx, to[1] + ny],
    [to[0] - nx, to[1] - ny],
    [from[0] - nx, from[1] - ny]
  ];
}

function createBox([west, south, east, north]: number[]): number[][] {
  return [
    [west, south],
    [east, south],
    [east, north],
    [west, north]
  ];
}

function createComb(
  origin: Vertex,
  teeth: number,
  pitch: number,
  toothWidth: number,
  height: number
) {
  const bar = height / 8;
  const ring: number[][] = [
    [origin[0], origin[1]],
    [origin[0] + (teeth - 1) * pitch + toothWidth, origin[1]],
    [origin[0] + (teeth - 1) * pitch + toothWidth, origin[1] + bar]
  ];
  for (let tooth = teeth - 1; tooth >= 0; tooth--) {
    const left = origin[0] + tooth * pitch;
    ring.push(
      [left + toothWidth, origin[1] + bar],
      [left + toothWidth, origin[1] + height],
      [left, origin[1] + height],
      [left, origin[1] + bar]
    );
  }
  ring.push([origin[0], origin[1] + bar]);
  return ring;
}

function createSparseScene(): CoverFeature[] {
  return roundFeatures([
    // Thin diagonal strip (1e-3 degrees wide, 6 degrees long).
    [[createStrip([0.3, 0.2], [6.1, 5.9], 1e-3)]],
    // Wider strip.
    [[createStrip([20.2, 40.1], [26.4, 44.7], 0.05)]],
    // Comb.
    [[createComb([30, 10], 12, 0.5, 0.04, 3)]],
    // Rectangle with two holes.
    [
      [
        createBox([-30, 10, -27, 13]),
        createBox([-29.6, 10.4, -28.8, 11.2]).reverse(),
        createBox([-28.2, 11.8, -27.4, 12.6]).reverse()
      ]
    ],
    // Multipolygon with far-apart parts.
    [
      [createBox([-60, -30, -59.8, -29.8])],
      [createBox([-45, -20, -44.6, -19.7])],
      [createStrip([-52, -28], [-50.5, -26.5], 0.02)]
    ],
    // Tiny polygon inside one tile.
    [[createBox([5.0001, 5.0001, 5.0003, 5.0003])]],
    // Star (dense) for a mix of choices in one run.
    [[createStarPolygon(7, [-100.2, 40.4], 1.4, 17)]],
    // Empty feature.
    [],
    // Feature with a non-finite vertex: keeps the bounding box.
    [
      [
        [
          [8, 8],
          [8.5, NaN],
          [9, 8.4],
          [8.6, 8.9]
        ]
      ]
    ]
  ]);
}

it('GPUCellCover quadbin scanline equals bounds and the CPU oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = createSparseScene();
  const arrays = flattenCoverFeatures(features);
  for (const resolution of [9, 11, 13]) {
    for (const containment of ['center', 'full', 'intersects'] as const) {
      const label = `z${resolution} ${containment}`;
      const expected = coverQuadbinOnCPU(arrays, resolution, containment);
      const base = {
        family: 'quadbin',
        resolution,
        containment,
        candidateCapacity: 2_000_000,
        outputCapacity: Math.max(expected.cells.length + 64, 64)
      } as const;
      const bounds = await runCover(device, features, {...base, candidateEnumeration: 'bounds'});
      expect(bounds.candidateOverflow, `${label} bounds overflow`).toBe(0);
      expect(
        bounds.cells.map(cell => cell.toString(16)),
        `${label} bounds vs oracle`
      ).toEqual(expected.cells.map(cell => cell.toString(16)));
      for (const enumeration of ['scanline', 'auto'] as const) {
        for (const edgeSlabs of [false, true]) {
          const actual = await runCover(device, features, {
            ...base,
            candidateEnumeration: enumeration,
            edgeSlabs
          });
          expectSameRun(actual, bounds, `${label} ${enumeration} slabs ${edgeSlabs}`);
          expect(actual.candidateOverflow).toBe(0);
          // The scanline enumeration never visits more cells than the boxes.
          expect(actual.candidateCount).toBeLessThanOrEqual(bounds.candidateCount);
        }
      }
      const forced = await runCover(device, features, {
        ...base,
        candidateEnumeration: 'scanline'
      });
      if (resolution >= 11) {
        // Thin and sparse features: far fewer candidates than the sum of the boxes.
        expect(forced.candidateCount * 3).toBeLessThan(bounds.candidateCount);
      }
    }
  }
});

it('GPUCellCover scanline falls back per feature to the bounding box', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = createSparseScene();
  const arrays = flattenCoverFeatures(features);
  const expected = coverQuadbinOnCPU(arrays, 12, 'center');
  const base = {
    family: 'quadbin',
    resolution: 12,
    candidateCapacity: 2_000_000,
    outputCapacity: expected.cells.length + 64
  } as const;
  const bounds = await runCover(device, features, {...base, candidateEnumeration: 'bounds'});
  // A stage too small for any feature (rows plus slots) keeps every bounding box.
  const tiny = await runCover(device, features, {
    ...base,
    candidateEnumeration: 'scanline',
    scanlineSlotCapacity: 4
  });
  expectSameRun(tiny, bounds, 'tiny stage');
  expect(tiny.candidateCount).toBe(bounds.candidateCount);
  // A stage that holds only the first feature that fits: still identical output.
  const small = await runCover(device, features, {
    ...base,
    candidateEnumeration: 'scanline',
    scanlineSlotCapacity: 600
  });
  expectSameRun(small, bounds, 'small stage');
  expect(small.candidateCount).toBeLessThanOrEqual(bounds.candidateCount);
});

// ---------------------------------------------------------------------------------------------
// H3: scanline and auto equal bounds and h3-py.
// ---------------------------------------------------------------------------------------------

type ThinCase = {
  name: string;
  resolution: number;
  polygons: {shell: [number, number][]; holes: [number, number][][]}[];
  cells: string[];
  ambiguous: string[];
  boundsCandidates: number;
  pentagonCells: number;
};

const fixture = fixtureJson as unknown as {
  tolerances: {centerMargin: number};
  cases: ThinCase[];
};

/** Cases whose bounding-box enumeration fits `candidateCapacity` and one storage binding. */
const BOUNDS_FEASIBLE_LIMIT = 4_000_000;

function getFeature(testCase: ThinCase): CoverFeature {
  return testCase.polygons.map(polygon => [polygon.shell, ...polygon.holes]);
}

function expectMatchesH3(run: Run, testCase: ThinCase, label: string) {
  const expected = new Set(testCase.cells);
  const ambiguous = new Set(testCase.ambiguous);
  const actual = run.cells.map(cell => cell.toString(16));
  expect(new Set(actual).size, `${label} duplicate cells`).toBe(actual.length);
  const unexpected = actual.filter(cell => !expected.has(cell) && !ambiguous.has(cell));
  const missing = testCase.cells.filter(cell => !new Set(actual).has(cell) && !ambiguous.has(cell));
  expect(unexpected, `${label} cells h3 does not have`).toEqual([]);
  expect(missing, `${label} cells h3 has`).toEqual([]);
}

it('GPUCellCover h3 scanline equals bounds and h3-py on thin, sparse and large polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const report: string[] = [];
  for (const testCase of fixture.cases) {
    const label = `${testCase.name} res ${testCase.resolution}`;
    const features = [getFeature(testCase)];
    const feasible = testCase.boundsCandidates <= BOUNDS_FEASIBLE_LIMIT;
    // Rows of the box must fit the default stage; the far-apart multipolygon needs the maximum.
    const candidateCapacity = feasible ? testCase.boundsCandidates + 16 : 1_000_000;
    const base = {
      family: 'h3',
      resolution: testCase.resolution,
      candidateCapacity,
      outputCapacity: testCase.cells.length + testCase.ambiguous.length + 256,
      scanlineSlotCapacity: Math.min(1 << 17, candidateCapacity)
    } as const;
    const scanline = await runCover(device, features, {...base, candidateEnumeration: 'scanline'});
    expect(scanline.candidateOverflow, `${label} scanline candidate overflow`).toBe(0);
    expect(scanline.overflow, `${label} scanline overflow`).toBe(0);
    expectMatchesH3(scanline, testCase, `${label} scanline`);
    const auto = await runCover(device, features, {...base, candidateEnumeration: 'auto'});
    expectSameRun(auto, scanline, `${label} auto`);
    if (feasible) {
      const bounds = await runCover(device, features, {...base, candidateEnumeration: 'bounds'});
      expect(bounds.candidateOverflow, `${label} bounds candidate overflow`).toBe(0);
      expectSameRun(scanline, bounds, `${label} scanline vs bounds`);
      expectMatchesH3(bounds, testCase, `${label} bounds`);
      expect(scanline.candidateCount).toBeLessThanOrEqual(bounds.candidateCount);
      report.push(
        `${label}: candidates ${scanline.candidateCount} of ${bounds.candidateCount}, ` +
          `${scanline.milliseconds.toFixed(1)} ms vs ${bounds.milliseconds.toFixed(1)} ms`
      );
    } else {
      // The bounding box cannot be enumerated: the scanline stage is the only way to cover it.
      const bounds = await runCover(device, features, {...base, candidateEnumeration: 'bounds'});
      expect(bounds.candidateOverflow, `${label} bounds overflow expected`).toBe(1);
      expect(scanline.candidateCount * 1000).toBeLessThan(testCase.boundsCandidates);
      report.push(
        `${label}: candidates ${scanline.candidateCount} of ${testCase.boundsCandidates} (bounds overflows)`
      );
    }
  }
  // eslint-disable-next-line no-console
  console.log(`H3 scanline enumeration:\n${report.join('\n')}`);
});

it('GPUCellCover h3 scanline keeps one cell per lattice point across features', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One run holding every feasible resolution-7 case plus a feature with a non-finite vertex.
  const cases = fixture.cases.filter(
    testCase => testCase.resolution === 7 && testCase.boundsCandidates <= BOUNDS_FEASIBLE_LIMIT
  );
  const features: CoverFeature[] = [
    ...cases.map(getFeature),
    [
      [
        [
          [12.3, 41.2],
          [12.4, Number.NaN],
          [12.5, 41.3],
          [12.35, 41.4]
        ]
      ]
    ]
  ];
  const base = {
    family: 'h3',
    resolution: 7,
    candidateCapacity: 3_000_000,
    outputCapacity: 20_000
  } as const;
  const bounds = await runCover(device, features, {...base, candidateEnumeration: 'bounds'});
  const scanline = await runCover(device, features, {...base, candidateEnumeration: 'scanline'});
  const auto = await runCover(device, features, {...base, candidateEnumeration: 'auto'});
  expect(bounds.candidateOverflow).toBe(0);
  expectSameRun(scanline, bounds, 'mixed scanline');
  expectSameRun(auto, bounds, 'mixed auto');
});

it('GPUCellCover scanline fits polygons the bounding box cannot enumerate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const testCase = fixture.cases.find(
    item => item.name === 'multipolygon far apart' && item.resolution === 9
  )!;
  const features = [getFeature(testCase)];
  const base = {
    family: 'h3',
    resolution: 9,
    candidateCapacity: 200_000,
    outputCapacity: 2048,
    scanlineSlotCapacity: 1 << 17
  } as const;
  expect(testCase.boundsCandidates).toBeGreaterThan(2 ** 24);
  const bounds = await runCover(device, features, {...base, candidateEnumeration: 'bounds'});
  expect(bounds.candidateOverflow).toBe(1);
  const scanline = await runCover(device, features, {...base, candidateEnumeration: 'scanline'});
  expect(scanline.candidateOverflow).toBe(0);
  expectMatchesH3(scanline, testCase, 'far apart');
  // A small stage cannot take the feature; it then overflows like the bounding box.
  const small = await runCover(device, features, {
    ...base,
    candidateEnumeration: 'scanline',
    scanlineSlotCapacity: 64
  });
  expect(small.candidateOverflow).toBe(1);
});

// ---------------------------------------------------------------------------------------------
// A/B harness (logs only; run by hand): bounds against scanline.
// ---------------------------------------------------------------------------------------------

function createScatteredMultiPolygon(
  seed: number,
  origin: Vertex,
  extent: number,
  partCount: number,
  partSize: number
): CoverFeature {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return Array.from({length: partCount}, () => {
    const west = origin[0] + random() * extent;
    const south = origin[1] + random() * extent;
    return [createBox([west, south, west + partSize, south + partSize * (0.5 + random())])];
  });
}

it('GPUCellCover scanline A/B: diagonal strip and scattered multipolygon', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows: string[] = [];
  const measure = async (
    label: string,
    features: CoverFeature[],
    options: Omit<RunOptions, 'candidateEnumeration'>
  ) => {
    const results: Record<string, Run> = {};
    for (const enumeration of ['bounds', 'scanline', 'auto'] as const) {
      let best: Run | undefined;
      // Three runs, the best kept: the first compiles pipelines and warms the buffers.
      for (let attempt = 0; attempt < 3; attempt++) {
        const run = await runCover(device, features, {
          ...options,
          candidateEnumeration: enumeration
        });
        if (!best || run.milliseconds < best.milliseconds) {
          best = run;
        }
      }
      results[enumeration] = best!;
    }
    const {bounds, scanline, auto} = results;
    if (bounds.candidateOverflow === 0) {
      expectSameRun(scanline, bounds, `${label} scanline`);
      expectSameRun(auto, bounds, `${label} auto`);
    }
    rows.push(
      `${label}: bounds ${bounds.candidateCount} candidates ${bounds.milliseconds.toFixed(1)} ms` +
        `${bounds.candidateOverflow ? ' (overflow)' : ''}; scanline ${scanline.candidateCount} ` +
        `candidates ${scanline.milliseconds.toFixed(1)} ms; auto ${auto.candidateCount} ` +
        `candidates ${auto.milliseconds.toFixed(1)} ms; cells ${scanline.total}`
    );
  };
  // Strip 0.01 degrees wide. At Quadbin z14 the 20 degree diagonal fits the bounding box (about 0.4M
  // tiles); at H3 res 9 only a 1.5 degree strip does (about 5M points), so the full 20 degree strip
  // runs scanline only (the bounds run reports its candidate overflow).
  const strip20 = roundFeatures([[[createStrip([0.1, 0.1], [14.2, 14.2], 0.01)]]]);
  const strip15 = roundFeatures([[[createStrip([0.1, 0.1], [1.2, 1.2], 0.01)]]]);
  await measure('quadbin z14 strip 0.01 x 20', strip20, {
    family: 'quadbin',
    resolution: 14,
    candidateCapacity: 2_000_000,
    outputCapacity: 200_000,
    scanlineSlotCapacity: 1 << 16
  });
  await measure('h3 res 9 strip 0.01 x 1.5', strip15, {
    family: 'h3',
    resolution: 9,
    candidateCapacity: 12_000_000,
    outputCapacity: 200_000,
    scanlineSlotCapacity: 1 << 14
  });
  await measure('h3 res 9 strip 0.01 x 20', strip20, {
    family: 'h3',
    resolution: 9,
    candidateCapacity: 12_000_000,
    outputCapacity: 400_000,
    scanlineSlotCapacity: 1 << 17
  });
  // 1000 small parts scattered over a few degrees.
  const scatteredQuadbin = roundFeatures([createScatteredMultiPolygon(11, [2, 2], 5, 1000, 0.02)]);
  const scatteredH3 = roundFeatures([createScatteredMultiPolygon(11, [2, 2], 1.4, 1000, 0.004)]);
  await measure('quadbin z14 1000 parts in 5 x 5', scatteredQuadbin, {
    family: 'quadbin',
    resolution: 14,
    candidateCapacity: 2_000_000,
    outputCapacity: 400_000,
    scanlineSlotCapacity: 1 << 16
  });
  await measure('h3 res 9 1000 parts in 1.4 x 1.4', scatteredH3, {
    family: 'h3',
    resolution: 9,
    candidateCapacity: 12_000_000,
    outputCapacity: 400_000,
    scanlineSlotCapacity: 1 << 16
  });
  // eslint-disable-next-line no-console
  console.log(`GPUCellCover scanline A/B (wall clock, best of 3):\n${rows.join('\n')}`);
  expect(rows.length).toBe(5);
});
