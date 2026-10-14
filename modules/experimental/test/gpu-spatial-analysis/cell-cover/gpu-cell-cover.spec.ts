// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellToLatLng, getPentagons} from 'h3-js';
import {
  FIJI_SPLIT,
  FIJI_TRANSMERIDIAN,
  ANTARCTICA_NATURAL_EARTH
} from '../lon-lat-polygons/fixtures';
import {getCoverFixtureFeatures} from './cell-cover-oracle';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUCellCover,
  type GPUCellCoverContainment
} from '../../../src/gpu-spatial-analysis/cell-cover';
import {bigIntToH3, joinCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  countCoverDisagreements,
  getDistanceToFeatureBoundary,
  listCoverDisagreements,
  coverH3OnCPU,
  coverQuadbinCenterFloat64,
  coverQuadbinOnCPU,
  createRectangleRing,
  createStarPolygon,
  flattenCoverFeatures,
  roundFeatures,
  type CoverFeature,
  type CoverResult
} from './cell-cover-oracle';
import {isCellInsideFeature} from './cell-cover-core-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type CoverRun = CoverResult & {
  count: number;
  overflow: number;
  candidateOverflow: number;
  candidateCount: number;
  invalidCount: number;
  uncertainCount: number;
  total: number;
  featureIds: number[];
  core: number[];
};

async function runCover(
  device: Device,
  features: CoverFeature[],
  options: {
    family: 'quadbin' | 'h3';
    resolution: number;
    containment?: GPUCellCoverContainment;
    candidateCapacity: number;
    outputCapacity: number;
    featureIds?: number[];
    candidateEnumeration?: 'bounds' | 'auto' | 'scanline';
    edgeSlabs?: boolean;
    edgeSlabEntryCapacity?: number;
  }
): Promise<CoverRun> {
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
  const outputBuffers = {
    featureIds: output(outputCapacity),
    cells: output(2 * outputCapacity),
    core: output(outputCapacity),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1),
    candidateCount: output(1),
    invalidCount: output(1),
    uncertainCount: output(1),
    total: output(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-cover-graph'});
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
      featureIds: options.featureIds
        ? importGraphBuffer(
            graph,
            'feature-ids',
            input(Uint32Array.from(options.featureIds)),
            'uint32',
            options.featureIds.length
          )
        : undefined,
      candidateCapacity: options.candidateCapacity,
      candidateEnumeration: options.candidateEnumeration,
      edgeSlabs: options.edgeSlabs,
      edgeSlabEntryCapacity: options.edgeSlabEntryCapacity,
      output: {
        featureIds: importGraphBuffer(
          graph,
          'out-ids',
          outputBuffers.featureIds,
          'uint32',
          outputCapacity
        ),
        cells: importGraphBuffer(
          graph,
          'out-cells',
          outputBuffers.cells,
          'uint32x2',
          outputCapacity
        ),
        core: importGraphBuffer(graph, 'out-core', outputBuffers.core, 'uint32', outputCapacity),
        status: {
          candidateCount: importGraphBuffer(
            graph,
            'candidate-count',
            outputBuffers.candidateCount,
            'uint32',
            1
          ),
          invalidCount: importGraphBuffer(
            graph,
            'invalid-count',
            outputBuffers.invalidCount,
            'uint32',
            1
          ),
          uncertainCount: importGraphBuffer(
            graph,
            'uncertain-count',
            outputBuffers.uncertainCount,
            'uint32',
            1
          ),
          count: importGraphBuffer(graph, 'out-count', outputBuffers.count, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'out-total', outputBuffers.total, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'out-overflow', outputBuffers.overflow, 'uint32', 1),
          candidateOverflow: importGraphBuffer(
            graph,
            'out-candidate-overflow',
            outputBuffers.candidateOverflow,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(outputBuffers.count, 1);
  const [overflow] = await readUint32(outputBuffers.overflow, 1);
  const [candidateOverflow] = await readUint32(outputBuffers.candidateOverflow, 1);
  const [total] = await readUint32(outputBuffers.total, 1);
  const ids = await readUint32(outputBuffers.featureIds, outputCapacity);
  const words = await readUint32(outputBuffers.cells, 2 * outputCapacity);
  const core = await readUint32(outputBuffers.core, outputCapacity);
  const result: CoverRun = {
    core: core.slice(0, count),
    count,
    overflow,
    candidateOverflow,
    candidateCount: (await readUint32(outputBuffers.candidateCount, 1))[0],
    invalidCount: (await readUint32(outputBuffers.invalidCount, 1))[0],
    uncertainCount: (await readUint32(outputBuffers.uncertainCount, 1))[0],
    total,
    featureIds: ids.slice(0, count),
    featureRows: [],
    cells: []
  };
  for (let row = 0; row < count; row++) {
    result.cells.push(joinCellKey(words[2 * row], words[2 * row + 1]));
    result.featureRows.push(ids[row]);
  }
  compiled.destroy?.();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function createQuadbinScene(): CoverFeature[] {
  return roundFeatures([
    // Convex-ish star in Switzerland.
    [[createStarPolygon(3, [8.31, 47.13], 2.4, 19)]],
    // Rectangle with a hole, central US.
    [
      [
        createRectangleRing([-104.37, 36.11, -93.13, 43.71]),
        createRectangleRing([-100.91, 38.23, -96.57, 41.37]).reverse()
      ]
    ],
    // Empty feature: no polygons.
    [],
    // Multi-polygon feature in the southern hemisphere.
    [
      [createStarPolygon(21, [150.31, -33.91], 1.7, 13)],
      [createStarPolygon(22, [145.07, -37.83], 1.3, 11)]
    ],
    // Zero-area bounding box (all vertices equal).
    [
      [
        [
          [10.123, 20.456],
          [10.123, 20.456],
          [10.123, 20.456]
        ]
      ]
    ],
    // High latitude.
    [[createStarPolygon(31, [20.3, 70.2], 3.1, 15)]]
  ]);
}

it('GPUCellCover quadbin matches the CPU oracle for every containment', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = createQuadbinScene();
  const arrays = flattenCoverFeatures(features);
  for (const resolution of [3, 6, 8, 10]) {
    for (const containment of ['center', 'full', 'intersects'] as const) {
      const expected = coverQuadbinOnCPU(arrays, resolution, containment);
      const label = `res ${resolution} ${containment}`;
      const actual = await runCover(device, features, {
        family: 'quadbin',
        resolution,
        containment,
        candidateCapacity: 200000,
        outputCapacity: Math.max(expected.cells.length + 16, 16)
      });
      expect(actual.overflow, `${label} overflow`).toBe(0);
      expect(actual.total, `${label} total`).toBe(expected.cells.length);
      expect(actual.count, `${label} count`).toBe(expected.cells.length);
      expect(actual.featureRows, `${label} features`).toEqual(expected.featureRows);
      expect(
        actual.cells.map(cell => cell.toString(16)),
        `${label} cells`
      ).toEqual(expected.cells.map(cell => cell.toString(16)));
    }
  }
});

it('GPUCellCover quadbin center agrees with the independent f64 oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = createQuadbinScene();
  const arrays = flattenCoverFeatures(features);
  const report: string[] = [];
  let worst = 0;
  for (const resolution of [3, 6, 8, 10]) {
    const exact = coverQuadbinCenterFloat64(arrays, resolution);
    const actual = await runCover(device, features, {
      family: 'quadbin',
      resolution,
      candidateCapacity: 400000,
      outputCapacity: exact.cells.length + 64
    });
    const disagreements = countCoverDisagreements(actual, exact);
    worst = Math.max(worst, disagreements);
    report.push(`res ${resolution}: ${disagreements} of ${exact.cells.length}`);
  }
  // eslint-disable-next-line no-console
  console.log(`quadbin center GPU vs f64 oracle disagreements: ${report.join('; ')}`);
  expect(worst).toBeLessThanOrEqual(3);
});

it('GPUCellCover reports overflow, clamps count and uses feature ids', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = createQuadbinScene().slice(0, 2);
  const expected = coverQuadbinOnCPU(flattenCoverFeatures(features), 9, 'center');
  const capacity = 40;
  expect(expected.cells.length).toBeGreaterThan(capacity);
  const actual = await runCover(device, features, {
    family: 'quadbin',
    resolution: 9,
    candidateCapacity: 400000,
    outputCapacity: capacity,
    featureIds: [700, 900]
  });
  expect(actual.overflow).toBe(1);
  expect(actual.candidateOverflow).toBe(0);
  expect(actual.total).toBe(expected.cells.length);
  expect(actual.count).toBe(capacity);
  expect(actual.cells.map(cell => cell.toString(16))).toEqual(
    expected.cells.slice(0, capacity).map(cell => cell.toString(16))
  );
  expect(actual.featureRows).toEqual(
    expected.featureRows.slice(0, capacity).map(row => [700, 900][row])
  );
  // Candidate capacity overflow: only the first candidates are tested.
  const limited = await runCover(device, features, {
    family: 'quadbin',
    resolution: 9,
    candidateCapacity: 50,
    outputCapacity: 4096
  });
  expect(limited.overflow).toBe(0);
  expect(limited.candidateOverflow).toBe(1);
  const limitedExpected = coverQuadbinOnCPU(flattenCoverFeatures(features), 9, 'center', 50);
  expect(limited.cells).toEqual(limitedExpected.cells);
});

it('GPUCellCover h3 center matches h3-js polygonToCells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const [pentagonLat, pentagonLng] = cellToLatLng(getPentagons(5)[0]);
  const createScene = (scale: number) =>
    roundFeatures([
      [[createStarPolygon(5, [8.5, 47.2], 0.6 * scale, 13)]],
      [
        [
          createRectangleRing([-100.5, 39.5, -100.5 + 1.5 * scale, 39.5 + 1.2 * scale]),
          createRectangleRing([
            -100.5 + 0.5 * scale,
            39.5 + 0.4 * scale,
            -100.5 + 0.9 * scale,
            39.5 + 0.8 * scale
          ]).reverse()
        ]
      ],
      [[createStarPolygon(7, [20, 70], 0.8 * scale, 11)]],
      [[createStarPolygon(11, [pentagonLng, pentagonLat], 0.7 * scale, 15)]],
      []
    ]);
  const report: string[] = [];
  let totalDisagreements = 0;
  let nearEdge = 0;
  let otherCause = 0;
  for (const resolution of [3, 4, 5, 6, 7, 8, 9, 10]) {
    const features = createScene(resolution >= 9 ? 0.12 : 1);
    const expected = coverH3OnCPU(features, resolution);
    const actual = await runCover(device, features, {
      family: 'h3',
      resolution,
      candidateCapacity: 4_000_000,
      outputCapacity: expected.cells.length + 256
    });
    expect(actual.overflow, `res ${resolution} overflow`).toBe(0);
    const keys = actual.featureRows.map((row, index) => `${row}:${actual.cells[index]}`);
    expect(new Set(keys).size, `res ${resolution} unique`).toBe(keys.length);
    const disagreements = countCoverDisagreements(actual, expected);
    totalDisagreements += disagreements;
    // Cause: a center within 5e-5 degrees of a polygon edge (f32 center vs f64 center), else the
    // f32 forward index returned a neighbouring cell for the lattice point nearest the center.
    for (const row of listCoverDisagreements(actual, expected)) {
      const [centreLat, centreLng] = cellToLatLng(bigIntToH3(row.cell));
      const distance = getDistanceToFeatureBoundary(flattenCoverFeatures(features), row.feature, [
        centreLng,
        centreLat
      ]);
      if (distance < 5e-5) {
        nearEdge++;
      } else {
        otherCause++;
      }
    }
    report.push(`res ${resolution}: ${disagreements} of ${expected.cells.length}`);
  }
  // eslint-disable-next-line no-console
  console.log(
    `h3 center GPU vs h3-js disagreements: ${report.join('; ')}; near-edge centers ${nearEdge}, other ${otherCause}`
  );
  expect(totalDisagreements).toBeLessThanOrEqual(20);
});

it('GPUCellCover h3 center keeps the cells of tall high-latitude polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // D-07: a longitude spacing taken at the poleward edge lost 448 of 1756 res-3 cells over 0-85
  // degrees; the spacing now comes from the most equatorward latitude. Truth: h3-js polygonToCells.
  const features = roundFeatures([
    [[createRectangleRing([-20, 0, 20, 80])]],
    [[createRectangleRing([-141, 41.7, -52, 83.1])]],
    [[createRectangleRing([30, -82, 45, -5])]]
  ]);
  for (const resolution of [2, 3]) {
    const expected = coverH3OnCPU(features, resolution);
    const actual = await runCover(device, features, {
      family: 'h3',
      resolution,
      candidateCapacity: 4_000_000,
      outputCapacity: expected.cells.length + 256
    });
    expect(actual.overflow, `res ${resolution} overflow`).toBe(0);
    const disagreements = listCoverDisagreements(actual, expected);
    // Only f32 center-on-edge ties may differ (same allowance as the scene above, per cell).
    for (const row of disagreements) {
      const [centreLat, centreLng] = cellToLatLng(bigIntToH3(row.cell));
      const distance = getDistanceToFeatureBoundary(flattenCoverFeatures(features), row.feature, [
        centreLng,
        centreLat
      ]);
      expect(distance, `res ${resolution} cell ${row.cell}`).toBeLessThan(5e-5);
    }
  }
});

it('GPUCellCover core flag is sound, conservative and nonzero (quadbin and h3)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases: {family: 'quadbin' | 'h3'; resolution: number; features: CoverFeature[]}[] = [
    {family: 'quadbin', resolution: 8, features: createQuadbinScene().slice(0, 4)},
    {family: 'quadbin', resolution: 10, features: createQuadbinScene().slice(0, 1)},
    {
      family: 'h3',
      resolution: 5,
      features: roundFeatures([
        [[createStarPolygon(5, [8.5, 47.2], 1.2, 13)]],
        [
          [
            createRectangleRing([-100.5, 39.5, -97.5, 42]),
            createRectangleRing([-99.5, 40.3, -98.7, 41.1]).reverse()
          ]
        ]
      ])
    }
  ];
  for (const {family, resolution, features} of cases) {
    const label = `${family} res ${resolution}`;
    for (const containment of family === 'h3'
      ? (['center'] as const)
      : (['center', 'intersects'] as const)) {
      const run = await runCover(device, features, {
        family,
        resolution,
        containment,
        candidateCapacity: 400000,
        outputCapacity: 20000
      });
      expect(run.overflow, `${label} overflow`).toBe(0);
      expect(run.count, `${label} count`).toBeGreaterThan(20);
      expect(run.core.every(flag => flag === 0 || flag === 1)).toBe(true);
      const coreCount = run.core.reduce((sum, flag) => sum + flag, 0);
      // Interior cells exist, and border cells (near the boundary or the hole) stay unflagged.
      expect(coreCount, `${label} ${containment} core`).toBeGreaterThan(0);
      expect(coreCount, `${label} ${containment} border`).toBeLessThan(run.count);
      for (let row = 0; row < run.count; row++) {
        if (run.core[row]) {
          expect(
            isCellInsideFeature(features[run.featureRows[row]], family, run.cells[row]),
            `${label} ${containment} core cell ${run.cells[row].toString(16)} is inside`
          ).toBe(true);
        }
      }
      // Cells in the output keep their keys: the core bit never leaks into them.
      for (const cell of run.cells) {
        expect(cell >> 63n).toBe(0n);
      }
    }
  }
});

/** Features with many edges: random stars, a comb with horizontal and vertical edges, holes. */
function createDenseScene(): CoverFeature[] {
  const comb: number[][] = [[-20, 10]];
  for (let tooth = 0; tooth < 40; tooth++) {
    const x = -20 + tooth * 0.5;
    comb.push([x, 10], [x, 12 + (tooth % 3)], [x + 0.25, 12 + (tooth % 3)], [x + 0.25, 10]);
  }
  comb.push([0, 8], [-20, 8]);
  return roundFeatures([
    [[createStarPolygon(3, [8.31, 47.13], 2.4, 400)]],
    [
      [
        createStarPolygon(9, [-100, 40], 4, 300),
        createStarPolygon(10, [-100, 40], 1.2, 120).reverse()
      ],
      [createStarPolygon(12, [-90, 38], 1.5, 60)]
    ],
    [[comb]],
    [],
    [[createStarPolygon(15, [150.3, -33.9], 1.7, 12)]],
    // A feature with a non-finite vertex never uses slabs; it must still cover as before.
    [[[...createStarPolygon(21, [20.3, 55.2], 2.5, 50), [Number.NaN, 55]]]]
  ]);
}

it('GPUCellCover edge slabs give the same cells and core flags as testing every edge', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = createDenseScene();
  const cases: {
    family: 'quadbin' | 'h3';
    resolution: number;
    containment: GPUCellCoverContainment;
  }[] = [
    {family: 'quadbin', resolution: 9, containment: 'center'},
    {family: 'quadbin', resolution: 9, containment: 'full'},
    {family: 'quadbin', resolution: 10, containment: 'intersects'},
    {family: 'h3', resolution: 4, containment: 'center'}
  ];
  for (const {family, resolution, containment} of cases) {
    const label = `${family} res ${resolution} ${containment}`;
    const options = {
      family,
      resolution,
      containment,
      candidateCapacity: 2_000_000,
      outputCapacity: 60000
    };
    const reference = await runCover(device, features, {...options, edgeSlabs: false});
    expect(reference.overflow, `${label} reference overflow`).toBe(0);
    expect(reference.count, `${label} reference count`).toBeGreaterThan(50);
    // Default capacity, then a tiny capacity so that most features fall back mid-way.
    for (const edgeSlabEntryCapacity of [undefined, 900]) {
      const slabbed = await runCover(device, features, {
        ...options,
        edgeSlabs: true,
        edgeSlabEntryCapacity
      });
      const tag = `${label} capacity ${edgeSlabEntryCapacity ?? 'default'}`;
      expect(slabbed.overflow, `${tag} overflow`).toBe(0);
      expect(slabbed.total, `${tag} total`).toBe(reference.total);
      expect(slabbed.featureRows, `${tag} features`).toEqual(reference.featureRows);
      expect(
        slabbed.cells.map(cell => cell.toString(16)),
        `${tag} cells`
      ).toEqual(reference.cells.map(cell => cell.toString(16)));
      expect(slabbed.core, `${tag} core`).toEqual(reference.core);
    }
  }
});

// With edge slabs the counters move to their own `-status` pass (the count kernel would bind 9).
for (const edgeSlabs of [false, true]) {
  it(`GPUCellCover rejects antimeridian-spanning features and counts clamped polar features (edge slabs ${edgeSlabs})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Feature 0 is ordinary, feature 1 crosses the antimeridian (its bounding box is 340 degrees
    // wide, edge longitude difference 340), feature 2 reaches past the Web Mercator latitude limit.
    const features: CoverFeature[] = [
      [[createRectangleRing([0, 0, 2, 2])]],
      [[createRectangleRing([170, 0, -170, 2])]],
      [[createRectangleRing([0, 84, 2, 86])]]
    ];
    const arrays = flattenCoverFeatures(features);
    const buffers: Buffer[] = [];
    const input = (values: Float32Array | Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const output = (length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return buffer;
    };
    const capacity = 256;
    const featureIds = output(capacity);
    const cells = output(2 * capacity);
    const count = output(1);
    const overflow = output(1);
    const candidateOverflow = output(1);
    const invalidCount = output(1);
    const uncertainCount = output(1);
    const graph = new GPUCommandGraph(device, {id: 'cell-cover-status'});
    const view = (
      name: string,
      values: Float32Array | Uint32Array,
      format: 'float32x2' | 'uint32'
    ) =>
      importGraphBuffer(
        graph,
        name,
        input(values),
        format,
        format === 'float32x2' ? values.length / 2 : values.length
      );
    graph.add(
      new GPUCellCover({
        family: 'quadbin',
        resolution: 4,
        // A res-4 tile is 22.5 degrees wide, so the 2-degree feature 0 contains no tile center;
        // 'intersects' gives it cells, which the assertion below needs.
        containment: 'intersects',
        positions: view('positions', arrays.positions, 'float32x2') as never,
        featureOffsets: view('feature-offsets', arrays.featureOffsets, 'uint32') as never,
        polygonOffsets: view('polygon-offsets', arrays.polygonOffsets, 'uint32') as never,
        ringOffsets: view('ring-offsets', arrays.ringOffsets, 'uint32') as never,
        candidateCapacity: 4096,
        edgeSlabs,
        output: {
          featureIds: importGraphBuffer(graph, 'out-ids', featureIds, 'uint32', capacity),
          cells: importGraphBuffer(graph, 'out-cells', cells, 'uint32x2', capacity),
          status: {
            count: importGraphBuffer(graph, 'out-count', count, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'out-overflow', overflow, 'uint32', 1),
            candidateOverflow: importGraphBuffer(
              graph,
              'out-candidate',
              candidateOverflow,
              'uint32',
              1
            ),
            invalidCount: importGraphBuffer(graph, 'out-invalid', invalidCount, 'uint32', 1),
            uncertainCount: importGraphBuffer(graph, 'out-uncertain', uncertainCount, 'uint32', 1)
          }
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect((await readUint32(invalidCount, 1))[0]).toBe(1);
    expect((await readUint32(uncertainCount, 1))[0]).toBe(1);
    const written = (await readUint32(count, 1))[0];
    const ids = (await readUint32(featureIds, capacity)).slice(0, written);
    expect(ids.includes(0)).toBe(true);
    expect(ids.includes(1)).toBe(false);
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  });
}

for (const candidateEnumeration of ['bounds', 'auto', 'scanline'] as const) {
  it(`GPUCellCover split Fiji equals transmeridian h3-js cover (${candidateEnumeration})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const features = getCoverFixtureFeatures(FIJI_SPLIT);
    // Fixed prepared topology can include empty polygons/rings.
    features[0].push([[]]);
    const result = await runCover(device, features, {
      family: 'h3',
      resolution: 3,
      candidateEnumeration,
      candidateCapacity: 100000,
      outputCapacity: 1000
    });
    const expected = coverH3OnCPU(getCoverFixtureFeatures(FIJI_TRANSMERIDIAN), 3);
    expect(result.invalidCount).toBe(0);
    expect(result.candidateOverflow).toBe(0);
    expect(result.overflow).toBe(0);
    expect(countCoverDisagreements(result, expected)).toBe(0);
    expect(new Set(result.cells).size).toBe(result.count);
    expect(result.count).toBeGreaterThan(0);
  });
}

it('GPUCellCover accepts full-width Antarctica and reports candidates and polar uncertainty', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const features = getCoverFixtureFeatures(ANTARCTICA_NATURAL_EARTH);
  features.push([[[]]]);
  const result = await runCover(device, features, {
    family: 'quadbin',
    resolution: 4,
    candidateEnumeration: 'bounds',
    candidateCapacity: 256,
    outputCapacity: 256
  });
  expect(result.invalidCount).toBe(0);
  expect(result.uncertainCount).toBe(1);
  // Bounds [-180, -90, 180, -70] span all 16 columns and Mercator rows 12 through 15.
  expect(result.candidateCount).toBe(64);
  expect(result.candidateOverflow).toBe(0);
  expect(
    countCoverDisagreements(result, coverQuadbinOnCPU(flattenCoverFeatures(features), 4, 'center'))
  ).toBe(0);
  expect(
    countCoverDisagreements(result, coverQuadbinCenterFloat64(flattenCoverFeatures(features), 4))
  ).toBe(0);
  // The three southern rows have centers below every coastal vertex: 3 * 16 distinct cells.
  expect(result.cells).toHaveLength(48);
  expect(new Set(result.cells).size).toBe(result.cells.length);
  expect(result.featureRows.every(feature => feature === 0)).toBe(true);
});

it('GPUCellCover rejects unsplit transmeridian Fiji', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runCover(device, getCoverFixtureFeatures(FIJI_TRANSMERIDIAN), {
    family: 'h3',
    resolution: 3,
    candidateCapacity: 1000,
    outputCapacity: 1000
  });
  expect(result.invalidCount).toBe(1);
  expect(result.candidateCount).toBe(0);
  expect(result.count).toBe(0);
});
