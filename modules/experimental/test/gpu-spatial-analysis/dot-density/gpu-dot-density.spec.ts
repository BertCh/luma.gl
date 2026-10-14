// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUDotDensityParameterValues,
  GPUDotDensity,
  GPURandomPointsInPolygon,
  type GPUDotDensitySettings
} from '../../../src/gpu-spatial-analysis/dot-density';
import {generateDotCountsOnCPU, type DotDensityCPUResult} from './dot-density-cpu';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createPolygonColumns,
  isInSceneHole,
  SCENE_FEATURES,
  type SceneFeature
} from './dot-density-scene';
import shapelyFixtureJson from '../../oracles/fixtures/geometry-dot-density.json';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * Dots are sampled in f32 inside GPU-triangulated polygons, so a dot on a polygon edge can be off
 * by a few ulp of the coordinate (below 100, so about 1e-5). The grid lookup of the shapely
 * fixture accepts a dot within this distance of an inside cell.
 */
const EDGE_TOLERANCE = 1e-4;

type ShapelyFeature = {
  xs: number[];
  ys: number[];
  inside: boolean[][];
  area: number;
  boundingBoxFill: number;
  rings: number[][][][];
};

/** Shapely `contains` lookup written by test/oracles/geometry-dot-density.py. */
const SHAPELY_FIXTURE = shapelyFixtureJson as unknown as {
  shapely: string;
  features: ShapelyFeature[];
};

function findCell(edges: number[], value: number): number {
  for (let cell = 0; cell < edges.length - 1; cell++) {
    if (value >= edges[cell] && value < edges[cell + 1]) {
      return cell;
    }
  }
  return -1;
}

/** Whether a point is inside the shapely-contained cells of a feature, within EDGE_TOLERANCE. */
function isInsideShapelyFeature(feature: ShapelyFeature, x: number, y: number): boolean {
  for (const deltaX of [-EDGE_TOLERANCE, 0, EDGE_TOLERANCE]) {
    for (const deltaY of [-EDGE_TOLERANCE, 0, EDGE_TOLERANCE]) {
      const column = findCell(feature.xs, x + deltaX);
      const row = findCell(feature.ys, y + deltaY);
      if (column >= 0 && row >= 0 && feature.inside[row][column]) {
        return true;
      }
    }
  }
  return false;
}

type Mask = {weights: Float32Array; width: number; height: number};

type GPUResult = DotDensityCPUResult & {failedCount: number; positions: Float32Array};

type Fixture = {
  run(settings: GPUDotDensitySettings): Promise<GPUResult>;
  destroy(): void;
};

type FixtureProps = {
  features?: SceneFeature[];
  values?: Float32Array;
  counts?: Uint32Array;
  categoryCount?: number;
  capacity: number;
  maximumAttempts?: number;
  mask?: Mask;
};

function createFixture(device: Device, props: FixtureProps): Fixture {
  const polygons = createPolygonColumns(props.features ?? SCENE_FEATURES);
  const featureCount = polygons.featureOffsets.length - 1;
  const categoryCount = props.categoryCount ?? 1;
  const slotCount = featureCount * categoryCount;
  const {capacity} = props;
  const graph = new GPUCommandGraph(device, {id: 'dot-density-graph'});
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
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'dot-parameters',
    format: 'uint32',
    length: 8
  });
  const outputs = {
    positions: output(capacity * 2),
    ids: output(capacity),
    count: output(1),
    overflow: output(1),
    total: output(1),
    categories: output(capacity),
    failed: output(1),
    slotCounts: output(slotCount),
    slotOffsets: output(slotCount)
  };
  const polygonViews = {
    positions: importGraphBuffer(
      graph,
      'polygon-positions',
      input(polygons.positions),
      'float32x2',
      polygons.positions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      input(polygons.featureOffsets),
      'uint32',
      polygons.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      input(polygons.polygonOffsets),
      'uint32',
      polygons.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      input(polygons.ringOffsets),
      'uint32',
      polygons.ringOffsets.length
    )
  };
  const mask = props.mask
    ? {
        weights: importGraphBuffer(
          graph,
          'mask',
          input(props.mask.weights),
          'float32',
          props.mask.weights.length
        ),
        width: props.mask.width,
        height: props.mask.height
      }
    : undefined;
  const sharedOutput = {
    positions: importGraphBuffer(graph, 'o-positions', outputs.positions, 'float32x2', capacity),
    dots: {
      ids: importGraphBuffer(graph, 'o-ids', outputs.ids, 'uint32', capacity),
      status: {
        count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'o-total', outputs.total, 'uint32', 1)
      }
    },
    status: {failedCount: importGraphBuffer(graph, 'o-failed', outputs.failed, 'uint32', 1)},
    slotCounts: importGraphBuffer(graph, 'o-slot-counts', outputs.slotCounts, 'uint32', slotCount),
    slotOffsets: importGraphBuffer(
      graph,
      'o-slot-offsets',
      outputs.slotOffsets,
      'uint32',
      slotCount
    )
  };
  const parameters = parameterBuffer.importToGraph(graph);
  if (props.values) {
    graph.add(
      new GPUDotDensity({
        id: 'dots',
        ...polygonViews,
        values: importGraphBuffer(graph, 'values', input(props.values), 'float32', slotCount),
        categoryCount,
        parameters,
        maximumAttempts: props.maximumAttempts,
        mask,
        output: {
          ...sharedOutput,
          categories: importGraphBuffer(
            graph,
            'o-categories',
            outputs.categories,
            'uint32',
            capacity
          )
        }
      })
    );
  } else {
    graph.add(
      new GPURandomPointsInPolygon({
        id: 'points',
        ...polygonViews,
        counts: importGraphBuffer(graph, 'counts', input(props.counts!), 'uint32', featureCount),
        parameters,
        maximumAttempts: props.maximumAttempts,
        mask,
        output: sharedOutput
      })
    );
  }
  const compiled = graph.compile();
  return {
    async run(settings) {
      parameterBuffer.write(getGPUDotDensityParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      return {
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        requiredCount: (await readUint32(outputs.total, 1))[0],
        failedCount: (await readUint32(outputs.failed, 1))[0],
        slotCounts: Uint32Array.from(await readUint32(outputs.slotCounts, slotCount)),
        slotOffsets: Uint32Array.from(await readUint32(outputs.slotOffsets, slotCount)),
        positions: Float32Array.from(await readFloat32(outputs.positions, capacity * 2)),
        featureIds: Uint32Array.from(await readUint32(outputs.ids, capacity)),
        categories: Uint32Array.from(await readUint32(outputs.categories, capacity))
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectMatchesOracle(
  actual: GPUResult,
  expected: DotDensityCPUResult,
  checkCategories: boolean
): void {
  expect(Array.from(actual.slotCounts)).toEqual(Array.from(expected.slotCounts));
  expect(Array.from(actual.slotOffsets)).toEqual(Array.from(expected.slotOffsets));
  expect(actual.requiredCount).toBe(expected.requiredCount);
  expect(actual.count).toBe(expected.count);
  expect(actual.overflow).toBe(expected.overflow);
  expect(Array.from(actual.featureIds.slice(0, actual.count))).toEqual(
    Array.from(expected.featureIds)
  );
  if (checkCategories) {
    expect(Array.from(actual.categories.slice(0, actual.count))).toEqual(
      Array.from(expected.categories)
    );
  }
  // Rows past the count are cleared.
  for (let dot = actual.count; dot < actual.featureIds.length; dot++) {
    expect(actual.featureIds[dot]).toBe(0xffffffff);
    expect(Number.isNaN(actual.positions[2 * dot])).toBe(true);
  }
}

/** Counts of dots with a NaN position among the first `count` dots. */
function countMissingDots(actual: GPUResult): number {
  let missing = 0;
  for (let dot = 0; dot < actual.count; dot++) {
    if (Number.isNaN(actual.positions[2 * dot])) {
      missing++;
    }
  }
  return missing;
}

/** Whether a point is inside or on the triangle, within a small tolerance. */
function isInTriangle(
  point: [number, number],
  triangle: [number, number][],
  tolerance: number
): boolean {
  const sides = [0, 1, 2].map(index => {
    const [ax, ay] = triangle[index];
    const [bx, by] = triangle[(index + 1) % 3];
    const length = Math.hypot(bx - ax, by - ay);
    return ((bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)) / length;
  });
  return sides.every(side => side >= -tolerance) || sides.every(side => side <= tolerance);
}

const VALUES = new Float32Array([370.3, 120.6, 80.2, 45.5, 0, Number.NaN]);

/** The shapely fixture features as scene features (rings with `[x, y]` vertices). */
const FIXTURE_SCENE: SceneFeature[] = SHAPELY_FIXTURE.features.map(
  feature => feature.rings as SceneFeature
);

it('GPUDotDensity matches the CPU count oracle, places every dot and avoids holes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const fixture = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 2048
  });
  try {
    for (const settings of [
      {seed: 5, dotsPerUnit: 1},
      {seed: 6, dotsPerUnit: 2.5}
    ]) {
      const actual = await fixture.run(settings);
      const expected = generateDotCountsOnCPU({
        ...polygons,
        values: VALUES,
        categoryCount: 2,
        parameters: getGPUDotDensityParameterValues(settings),
        capacity: 2048
      });
      expectMatchesOracle(actual, expected, true);
      expect(actual.count).toBeGreaterThan(500);
      // Exact area-weighted sampling places every dot of a valid polygon.
      expect(actual.failedCount).toBe(0);
      expect(countMissingDots(actual)).toBe(0);
      for (let dot = 0; dot < actual.count; dot++) {
        expect(isInSceneHole(actual.positions[2 * dot], actual.positions[2 * dot + 1])).toBe(false);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('D-16: dots of a 5% fill C shape, a holed multipolygon and a same-winding hole are all placed inside (shapely)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(SHAPELY_FIXTURE.shapely).toMatch(/^2\./);
  const [cShape, holed, sameWinding] = SHAPELY_FIXTURE.features;
  // The C shape is the 5% bounding-box fill case: 32-attempt rejection lost about 19% of its dots.
  expect(cShape.boundingBoxFill).toBeLessThan(0.06);
  const counts = new Uint32Array([4000, 3000, 2000]);
  const fixture = createFixture(device, {
    features: FIXTURE_SCENE,
    counts,
    capacity: 16384
  });
  try {
    const actual = await fixture.run({seed: 21});
    expect(actual.count).toBe(9000);
    expect(Array.from(actual.slotCounts)).toEqual([4000, 3000, 2000]);
    // Rule 9: counts per feature are exact because no dot is dropped.
    expect(actual.failedCount).toBe(0);
    expect(countMissingDots(actual)).toBe(0);
    const features = [cShape, holed, sameWinding];
    // Per grid cell: dots must be in shapely-contained cells and spread in proportion to area.
    const cellCounts = features.map(feature => feature.inside.map(row => row.map(() => 0)));
    for (let dot = 0; dot < actual.count; dot++) {
      const feature = features[actual.featureIds[dot]];
      const x = actual.positions[2 * dot];
      const y = actual.positions[2 * dot + 1];
      expect(isInsideShapelyFeature(feature, x, y), `dot ${dot} (${x}, ${y})`).toBe(true);
      const column = findCell(feature.xs, x);
      const row = findCell(feature.ys, y);
      if (column >= 0 && row >= 0) {
        cellCounts[actual.featureIds[dot]][row][column]++;
      }
    }
    for (let featureIndex = 0; featureIndex < features.length; featureIndex++) {
      const feature = features[featureIndex];
      const total = counts[featureIndex];
      for (let row = 0; row < feature.ys.length - 1; row++) {
        for (let column = 0; column < feature.xs.length - 1; column++) {
          if (!feature.inside[row][column]) {
            continue;
          }
          const cellArea =
            (feature.xs[column + 1] - feature.xs[column]) * (feature.ys[row + 1] - feature.ys[row]);
          const expectedDots = (total * cellArea) / feature.area;
          // Binomial: 6 standard deviations plus a few edge dots.
          const limit = 6 * Math.sqrt(expectedDots) + 3;
          expect(
            Math.abs(cellCounts[featureIndex][row][column] - expectedDots),
            `feature ${featureIndex} cell ${row},${column}`
          ).toBeLessThan(limit);
        }
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('GPURandomPointsInPolygon keeps points stable when a count grows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const small = createFixture(device, {
    features: FIXTURE_SCENE,
    counts: new Uint32Array([100, 50, 20]),
    capacity: 512
  });
  const large = createFixture(device, {
    features: FIXTURE_SCENE,
    counts: new Uint32Array([300, 50, 20]),
    capacity: 512
  });
  try {
    const before = await small.run({seed: 8});
    const after = await large.run({seed: 8});
    // Feature 0 is the first slot, so its first 100 points are a stable prefix.
    for (let index = 0; index < 200; index++) {
      expect(after.positions[index]).toBe(before.positions[index]);
    }
  } finally {
    small.destroy();
    large.destroy();
  }
});

it('GPUDotDensity keeps dot positions stable when the dot value changes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 4096
  });
  try {
    const coarse = await fixture.run({seed: 11, dotsPerUnit: 0.5});
    const fine = await fixture.run({seed: 11, dotsPerUnit: 3});
    let comparedDots = 0;
    for (let slot = 0; slot < 6; slot++) {
      expect(fine.slotCounts[slot]).toBeGreaterThanOrEqual(coarse.slotCounts[slot]);
      for (let rank = 0; rank < coarse.slotCounts[slot]; rank++) {
        const coarseDot = coarse.slotOffsets[slot] + rank;
        const fineDot = fine.slotOffsets[slot] + rank;
        // Bitwise equal: same Philox draws on the same triangulation.
        expect(fine.positions[2 * fineDot]).toBe(coarse.positions[2 * coarseDot]);
        expect(fine.positions[2 * fineDot + 1]).toBe(coarse.positions[2 * coarseDot + 1]);
        comparedDots++;
      }
    }
    expect(comparedDots).toBeGreaterThan(200);
    // Replay of the first frame on the same compiled graph is bitwise identical.
    const replay = await fixture.run({seed: 11, dotsPerUnit: 0.5});
    expect(new Uint32Array(replay.positions.buffer)).toEqual(
      new Uint32Array(coarse.positions.buffer)
    );
  } finally {
    fixture.destroy();
  }
});

it('GPUDotDensity clamps to capacity and applies a mask, counting masked-out dots as failed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const small = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 100
  });
  try {
    const settings = {seed: 2, dotsPerUnit: 1};
    const actual = await small.run(settings);
    const expected = generateDotCountsOnCPU({
      ...polygons,
      values: VALUES,
      categoryCount: 2,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 100
    });
    expect(actual.count).toBe(100);
    expect(actual.overflow).toBe(1);
    expect(actual.requiredCount).toBeGreaterThan(100);
    expectMatchesOracle(actual, expected, true);
    expect(actual.failedCount).toBe(0);
  } finally {
    small.destroy();
  }

  const weights = new Float32Array(100);
  for (let row = 0; row < 10; row++) {
    weights.fill(0.75, row * 10 + 5, row * 10 + 10);
  }
  const mask = {weights, width: 10, height: 10};
  const maskedValues = new Float32Array([300, 50, 40]);
  const masked = createFixture(device, {
    values: maskedValues,
    capacity: 1024,
    maximumAttempts: 8,
    mask
  });
  try {
    const settings = {
      seed: 4,
      dotsPerUnit: 1,
      maskExtent: [0, 0, 1, 1] as const
    };
    const actual = await masked.run(settings);
    const expected = generateDotCountsOnCPU({
      ...polygons,
      values: maskedValues,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 1024
    });
    expectMatchesOracle(actual, expected, false);
    // Features 1 and 2 lie outside the mask: every one of their dots fails. Feature 0 loses a dot
    // with probability (1 - 0.75 * 0.48)^8 (half of the square, 96% outside the hole) = 3%, so 300
    // dots lose 9 +- 3 of them; 40 extra failures is a 10 sigma bound.
    const outsideMask = expected.slotCounts[1] + expected.slotCounts[2];
    expect(actual.failedCount).toBeGreaterThanOrEqual(outsideMask);
    expect(actual.failedCount).toBeLessThan(outsideMask + 40);
    expect(countMissingDots(actual)).toBe(actual.failedCount);
    for (let dot = 0; dot < actual.count; dot++) {
      const x = actual.positions[2 * dot];
      if (!Number.isNaN(x)) {
        expect(actual.featureIds[dot]).toBe(0);
        expect(x).toBeGreaterThanOrEqual(5);
      }
    }
  } finally {
    masked.destroy();
  }
});

it('GPURandomPointsInPolygon matches the count oracle and places every point of a thin sliver', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const counts = new Uint32Array([500, 300, 64]);
  const fixture = createFixture(device, {
    counts,
    capacity: 1024
  });
  try {
    const settings = {seed: 77};
    const actual = await fixture.run(settings);
    const expected = generateDotCountsOnCPU({
      ...polygons,
      counts,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 1024
    });
    expectMatchesOracle(actual, expected, false);
    expect(actual.count).toBe(864);
    // The sliver covers 0.1% of its bounding box, where 3 rejection attempts almost always failed.
    expect(actual.failedCount).toBe(0);
    for (let dot = 0; dot < 500; dot++) {
      const x = actual.positions[2 * dot];
      const y = actual.positions[2 * dot + 1];
      expect(isInSceneHole(x, y)).toBe(false);
      expect(x >= 0 && x <= 10 && y >= 0 && y <= 10).toBe(true);
    }
    const sliver: [number, number][] = [
      [40, 0],
      [50, 10],
      [49.98, 10]
    ];
    for (let dot = 800; dot < 864; dot++) {
      const point: [number, number] = [actual.positions[2 * dot], actual.positions[2 * dot + 1]];
      expect(isInTriangle(point, sliver, 1e-4), `sliver dot ${dot}`).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});

it('D-16: a feature that cannot be triangulated reports all of its dots as failed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Feature 1 is a collinear shell (zero area); feature 0 is a normal square.
  const features: SceneFeature[] = [
    [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ]
      ]
    ],
    [
      [
        [
          [10, 0],
          [15, 0],
          [20, 0]
        ]
      ]
    ]
  ];
  const fixture = createFixture(device, {
    features,
    counts: new Uint32Array([30, 20]),
    capacity: 64
  });
  try {
    const actual = await fixture.run({seed: 1});
    expect(actual.count).toBe(50);
    expect(actual.failedCount).toBe(20);
    expect(countMissingDots(actual)).toBe(20);
    for (let dot = 0; dot < 30; dot++) {
      expect(Number.isNaN(actual.positions[2 * dot])).toBe(false);
    }
    for (let dot = 30; dot < 50; dot++) {
      expect(Number.isNaN(actual.positions[2 * dot])).toBe(true);
    }
  } finally {
    fixture.destroy();
  }
});
