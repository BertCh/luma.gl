// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/index';
import {GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW} from '../../../src/gpu-spatial-analysis/group-geometry/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/*
 * Scene and oracle values come from shapely 2.1.2 (GEOS): minimum_rotated_rectangle /
 * oriented_envelope area, minimum_bounding_radius, minimum_bounding_circle centre, and the
 * maximum pairwise point distance, over each feature's points (float32-rounded). Generator:
 * scratchpad build/B/gen.py. Features: blob, axis-aligned square (equal-area tie), triangle,
 * two-ring multi-polygon, single point, two-point line.
 */
const POSITIONS = Float32Array.from([
  1019.2999877929688, 2013.199951171875, 1022.0999755859375, 2016.199951171875, 1015.5999755859375,
  2012.5999755859375, 1016.2000122070312, 2013.199951171875, 1020.0999755859375, 2016.699951171875,
  1017.0999755859375, 2014.5999755859375, 1016.9000244140625, 2014.5999755859375,
  1007.7000122070312, 2010.699951171875, 1001.7000122070312, 2006.800048828125, 999.0999755859375,
  2008.699951171875, 998.7000122070312, 2005.199951171875, 992.7000122070312, 2004.0,
  992.5999755859375, 2002.0, 986.0999755859375, 1999.199951171875, 988.4000244140625,
  1998.5999755859375, 979.0999755859375, 1992.0, 981.7999877929688, 1992.5, 983.7999877929688,
  1992.5999755859375, 978.0, 1988.4000244140625, 978.5999755859375, 1988.5999755859375,
  978.2000122070312, 1985.5, 980.5999755859375, 1986.5999755859375, 983.2999877929688,
  1987.9000244140625, 981.4000244140625, 1984.4000244140625, 993.2000122070312, 1990.300048828125,
  993.9000244140625, 1988.199951171875, 994.4000244140625, 1988.699951171875, 997.2999877929688,
  1991.300048828125, 998.0, 1990.0, 1008.5999755859375, 1999.699951171875, 1009.7000122070312,
  1999.699951171875, 1012.0, 1999.9000244140625, 1013.2000122070312, 2003.0, 1017.9000244140625,
  2006.0, 1017.2999877929688, 2005.9000244140625, 1020.5999755859375, 2007.5999755859375,
  1022.2999877929688, 2009.4000244140625, 1020.0, 2008.5, 1023.2000122070312, 2014.5,
  1016.4000244140625, 2011.199951171875, 1100.0, 2000.0, 1110.0, 2000.0, 1110.0, 2010.0, 1100.0,
  2010.0, 1200.0, 2000.0, 1212.0, 2003.0, 1203.0, 2011.0, 1304.699951171875, 2001.0, 1298.0,
  2002.300048828125, 1295.5999755859375, 2001.5, 1297.0, 2001.0, 1295.699951171875,
  2000.9000244140625, 1296.699951171875, 1999.699951171875, 1296.699951171875, 1998.300048828125,
  1298.300048828125, 1997.800048828125, 1299.699951171875, 1996.199951171875, 1303.199951171875,
  1997.5, 1303.199951171875, 1999.5999755859375, 1304.0, 2000.0, 1342.5999755859375,
  2035.0999755859375, 1341.300048828125, 2034.300048828125, 1340.4000244140625, 2033.9000244140625,
  1338.5999755859375, 2032.699951171875, 1337.300048828125, 2031.0, 1337.699951171875,
  2030.300048828125, 1336.5999755859375, 2028.4000244140625, 1337.4000244140625, 2026.5,
  1337.5999755859375, 2025.5999755859375, 1338.5, 2025.0999755859375, 1343.0, 2032.0,
  1343.199951171875, 2033.699951171875, 1400.0, 2000.0, 1500.0, 2000.0, 1503.0, 2004.0
]);
const RING_OFFSETS = Uint32Array.from([0, 40, 44, 47, 59, 71, 72, 74]);
const FEATURE_RING_OFFSETS = Uint32Array.from([0, 1, 2, 3, 5, 6, 7]);
const FEATURE_COUNT = 6;
const ORACLE = [
  {
    area: 783.8763679617563,
    envArea: 783.8763679617563,
    radius: 26.78480846087427,
    cx: 1000.1787490583587,
    cy: 2000.8088566100098,
    diameter: 53.56952292698544,
    bounds: [978.0, 1984.4000244140625, 1023.2000122070312, 2016.699951171875]
  },
  {
    area: 100.0,
    envArea: 100.0,
    radius: 7.0710678118654755,
    cx: 1104.9999999999998,
    cy: 2005.0000000000007,
    diameter: 14.142135623730951,
    bounds: [1100.0, 2000.0, 1110.0, 2010.0]
  },
  {
    area: 122.9999999978154,
    envArea: 122.9999999978154,
    radius: 6.903451619253195,
    cx: 1205.2560975609756,
    cy: 2004.4756097560974,
    diameter: 12.36931687685298,
    bounds: [1200.0, 2000.0, 1212.0, 2011.0]
  },
  {
    area: 452.52050174826167,
    envArea: 452.52050174826167,
    radius: 29.41533125171251,
    cx: 1319.6499633789062,
    cy: 2016.7000122070315,
    diameter: 58.83066250342502,
    bounds: [1295.5999755859375, 1996.199951171875, 1343.199951171875, 2035.0999755859375]
  },
  {
    area: 0.0,
    envArea: 0.0,
    radius: 0.0,
    cx: 1400.0,
    cy: 2000.0,
    diameter: 0,
    bounds: [1400.0, 2000.0, 1400.0, 2000.0]
  },
  {
    area: 0.0,
    envArea: 0.0,
    radius: 2.5,
    cx: 1501.5000000000005,
    cy: 2002.0000000000002,
    diameter: 5.0,
    bounds: [1500.0, 2000.0, 1503.0, 2004.0]
  }
];

type Result = {
  corners: number[];
  sizes: number[];
  circles: number[];
  lines: number[];
  diameters: number[];
  bounds: number[];
  hullSizes: number[];
  overflow: number;
};

async function runMinimumBounds(
  device: Device,
  mode: 'geometry' | 'labels',
  maximumHullVertices?: number
): Promise<Result> {
  const graph = new GPUCommandGraph(device, {id: 'minimum-bounds-graph'});
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
  const rows = POSITIONS.length / 2;
  const out = {
    corners: output(FEATURE_COUNT * 8),
    sizes: output(FEATURE_COUNT * 4),
    circles: output(FEATURE_COUNT * 3),
    lines: output(FEATURE_COUNT * 4),
    diameters: output(FEATURE_COUNT),
    bounds: output(FEATURE_COUNT * 4),
    hullSizes: output(FEATURE_COUNT),
    overflow: output(1)
  };
  const geometry =
    mode === 'geometry'
      ? {
          ringOffsets: importGraphBuffer(graph, 'rings', input(RING_OFFSETS), 'uint32', 8),
          featureRingOffsets: importGraphBuffer(
            graph,
            'features',
            input(FEATURE_RING_OFFSETS),
            'uint32',
            7
          )
        }
      : (() => {
          const labels = new Uint32Array(rows);
          for (let feature = 0; feature < FEATURE_COUNT; feature++) {
            for (
              let ring = FEATURE_RING_OFFSETS[feature];
              ring < FEATURE_RING_OFFSETS[feature + 1];
              ring++
            ) {
              labels.fill(feature, RING_OFFSETS[ring], RING_OFFSETS[ring + 1]);
            }
          }
          return {
            labels: importGraphBuffer(graph, 'labels', input(labels), 'uint32', rows),
            groupCount: FEATURE_COUNT
          };
        })();
  graph.add(
    new GPUMinimumBounds({
      id: 'mb',
      positions: importGraphBuffer(graph, 'positions', input(POSITIONS), 'float32x2', rows),
      ...geometry,
      maximumHullVertices,
      output: {
        rectangleCorners: importGraphBuffer(graph, 'o-corners', out.corners, 'float32x2', 24),
        rectangleSizes: importGraphBuffer(graph, 'o-sizes', out.sizes, 'float32x4', 6),
        circles: importGraphBuffer(graph, 'o-circles', out.circles, 'float32x3', 6),
        longestLines: importGraphBuffer(graph, 'o-lines', out.lines, 'float32x4', 6),
        diameters: importGraphBuffer(graph, 'o-diameters', out.diameters, 'float32', 6),
        bounds: importGraphBuffer(graph, 'o-bounds', out.bounds, 'float32x4', 6),
        hullSizes: importGraphBuffer(graph, 'o-hull-sizes', out.hullSizes, 'uint32', 6),
        overflow: importGraphBuffer(graph, 'o-overflow', out.overflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  const start = performance.now();
  submitGraph(device, compiled, undefined);
  const result: Result = {
    corners: await readFloat32(out.corners, FEATURE_COUNT * 8),
    sizes: await readFloat32(out.sizes, FEATURE_COUNT * 4),
    circles: await readFloat32(out.circles, FEATURE_COUNT * 3),
    lines: await readFloat32(out.lines, FEATURE_COUNT * 4),
    diameters: await readFloat32(out.diameters, FEATURE_COUNT),
    bounds: await readFloat32(out.bounds, FEATURE_COUNT * 4),
    hullSizes: await readUint32(out.hullSizes, FEATURE_COUNT),
    overflow: (await readUint32(out.overflow, 1))[0]
  };
  console.log(`GPUMinimumBounds ${mode}: ${(performance.now() - start).toFixed(1)} ms`);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function getPolygonArea(corners: number[], feature: number): number {
  let area = 0;
  for (let corner = 0; corner < 4; corner++) {
    const next = (corner + 1) % 4;
    const x0 = corners[feature * 8 + corner * 2] - 1000;
    const y0 = corners[feature * 8 + corner * 2 + 1] - 2000;
    const x1 = corners[feature * 8 + next * 2] - 1000;
    const y1 = corners[feature * 8 + next * 2 + 1] - 2000;
    area += x0 * y1 - x1 * y0;
  }
  return area / 2;
}

function expectClose(actual: number, expected: number, relative: number, absolute: number) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(absolute + relative * Math.abs(expected));
}

for (const mode of ['geometry', 'labels'] as const) {
  it(`GPUMinimumBounds (${mode}) matches shapely rectangle, circle, diameter and envelope`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const actual = await runMinimumBounds(device, mode);
    expect(actual.overflow).toBe(0);
    for (let feature = 0; feature < FEATURE_COUNT; feature++) {
      const expected = ORACLE[feature];
      const label = `feature ${feature}`;
      // Rectangle: area (not corner order), counter-clockwise corners.
      expectClose(actual.sizes[feature * 4 + 3], expected.area, 2e-4, 2e-3);
      expectClose(
        actual.sizes[feature * 4] * actual.sizes[feature * 4 + 1],
        expected.area,
        2e-4,
        2e-3
      );
      expect(getPolygonArea(actual.corners, feature), label).toBeGreaterThanOrEqual(-1e-3);
      expectClose(getPolygonArea(actual.corners, feature), expected.area, 2e-4, 2e-2);
      // Circle.
      expectClose(actual.circles[feature * 3 + 2], expected.radius, 1e-4, 1e-3);
      expectClose(actual.circles[feature * 3], expected.cx, 0, 2e-3);
      expectClose(actual.circles[feature * 3 + 1], expected.cy, 0, 2e-3);
      // Longest line and diameter.
      expectClose(actual.diameters[feature], expected.diameter, 1e-4, 1e-3);
      const [x0, y0, x1, y1] = actual.lines.slice(feature * 4, feature * 4 + 4);
      expectClose(Math.hypot(x1 - x0, y1 - y0), expected.diameter, 1e-4, 1e-3);
      // Envelope is exact: extremes are input coordinates.
      expect(actual.bounds.slice(feature * 4, feature * 4 + 4), label).toEqual(expected.bounds);
    }
    // Sanity against a silently failed compile (all zeros).
    expect(actual.hullSizes[0]).toBeGreaterThan(6);
    expect(actual.hullSizes.slice(1, 3)).toEqual([4, 3]);
    expect(actual.hullSizes.slice(4)).toEqual([1, 2]);
    // The square is a tie between its four edges: the lowest edge (smallest vertex, first CCW
    // edge, direction +x) wins, so angle 0 and width = height = 10.
    expect(actual.sizes.slice(4, 8).map(value => Math.round(value * 1e3) / 1e3)).toEqual([
      10, 10, 0, 100
    ]);
    // Degenerate features: a point has equal corners and zero radius; a line has zero height.
    expect(actual.circles[4 * 3 + 2]).toBe(0);
    expect(actual.sizes[4 * 4 + 3]).toBe(0);
    expect(actual.corners.slice(32, 40)).toEqual([1400, 2000, 1400, 2000, 1400, 2000, 1400, 2000]);
    expectClose(actual.sizes[5 * 4 + 1], 0, 0, 1e-3);
  });
}

it('GPUMinimumBounds flags hulls over maximumHullVertices and writes NaN for them', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const actual = await runMinimumBounds(device, 'geometry', 5);
  expect(actual.overflow & GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW).toBe(
    GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW
  );
  expect(actual.hullSizes[0]).toBeGreaterThan(5);
  expect(Number.isNaN(actual.circles[2])).toBe(true);
  expect(Number.isNaN(actual.diameters[0])).toBe(true);
  expect(Number.isNaN(actual.sizes[3])).toBe(true);
  expect(Number.isNaN(actual.corners[0])).toBe(true);
  expectClose(actual.circles[1 * 3 + 2], ORACLE[1].radius, 1e-4, 1e-3);
});

it('GPUMinimumBounds calipers on large hulls match a brute-force CPU scan', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Three groups with 60 to 220 hull vertices: a circle, a rotated thin ellipse (long aspect,
  // so the rectangle's extremes sweep a lot per edge) and a skewed superellipse.
  const groups: [number, number][][] = [];
  const ring = (count: number, make: (angle: number) => [number, number]) => {
    const points: [number, number][] = [];
    for (let k = 0; k < count; k++) {
      const [x, y] = make((2 * Math.PI * k) / count);
      points.push([Math.fround(x), Math.fround(y)]);
    }
    return points;
  };
  groups.push(ring(60, a => [500 + 40 * Math.cos(a), 700 + 40 * Math.sin(a)]));
  groups.push(
    ring(220, a => {
      const x = 90 * Math.cos(a);
      const y = 4 * Math.sin(a);
      return [1000 + x * 0.8 - y * 0.6, 300 + x * 0.6 + y * 0.8];
    })
  );
  groups.push(
    ring(130, a => {
      const c = Math.cos(a);
      const s = Math.sin(a);
      const x = Math.sign(c) * Math.abs(c) ** 0.6 * 30;
      const y = Math.sign(s) * Math.abs(s) ** 0.6 * 12;
      return [-200 + x + 0.3 * y, 40 + y];
    })
  );
  const flat = new Float32Array(groups.flat().flatMap(point => point));
  const offsets = [0];
  for (const group of groups) {
    offsets.push(offsets[offsets.length - 1] + group.length);
  }
  const graph = new GPUCommandGraph(device, {id: 'minimum-bounds-large'});
  const buffers: Buffer[] = [];
  const rows = flat.length / 2;
  const groupCount = groups.length;
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
  const sizes = output(groupCount * 4);
  const lines = output(groupCount * 4);
  const diameters = output(groupCount);
  const overflow = output(1);
  graph.add(
    new GPUMinimumBounds({
      id: 'mb-large',
      positions: importGraphBuffer(graph, 'positions', input(flat), 'float32x2', rows),
      ringOffsets: importGraphBuffer(
        graph,
        'rings',
        input(Uint32Array.from(offsets)),
        'uint32',
        offsets.length
      ),
      output: {
        rectangleSizes: importGraphBuffer(graph, 'o-sizes', sizes, 'float32x4', groupCount),
        longestLines: importGraphBuffer(graph, 'o-lines', lines, 'float32x4', groupCount),
        diameters: importGraphBuffer(graph, 'o-diameters', diameters, 'float32', groupCount),
        overflow: importGraphBuffer(graph, 'o-overflow', overflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const actualSizes = await readFloat32(sizes, groupCount * 4);
  const actualDiameters = await readFloat32(diameters, groupCount);
  const actualLines = await readFloat32(lines, groupCount * 4);
  expect((await readUint32(overflow, 1))[0]).toBe(0);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  groups.forEach((points, group) => {
    // Brute force over all points: minimum-area rectangle aligned with any point pair edge of
    // the hull is the same as over all point pairs that are hull edges; scanning every pair
    // direction is a superset that contains the optimum and never goes below it.
    let bestArea = Infinity;
    let diameter = 0;
    for (let i = 0; i < points.length; i++) {
      for (let j = 0; j < points.length; j++) {
        diameter = Math.max(
          diameter,
          Math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
        );
      }
      const next = points[(i + 1) % points.length];
      const dx = next[0] - points[i][0];
      const dy = next[1] - points[i][1];
      const length = Math.hypot(dx, dy);
      const ux = dx / length;
      const uy = dy / length;
      let minAlong = Infinity;
      let maxAlong = -Infinity;
      let minAcross = Infinity;
      let maxAcross = -Infinity;
      for (const [x, y] of points) {
        const along = x * ux + y * uy;
        const across = -x * uy + y * ux;
        minAlong = Math.min(minAlong, along);
        maxAlong = Math.max(maxAlong, along);
        minAcross = Math.min(minAcross, across);
        maxAcross = Math.max(maxAcross, across);
      }
      bestArea = Math.min(bestArea, (maxAlong - minAlong) * (maxAcross - minAcross));
    }
    expectClose(actualSizes[group * 4 + 3], bestArea, 2e-4, 1e-2);
    expectClose(actualDiameters[group], diameter, 1e-5, 1e-3);
    const [x0, y0, x1, y1] = actualLines.slice(group * 4, group * 4 + 4);
    expectClose(Math.hypot(x1 - x0, y1 - y0), diameter, 1e-5, 1e-3);
  });
});
