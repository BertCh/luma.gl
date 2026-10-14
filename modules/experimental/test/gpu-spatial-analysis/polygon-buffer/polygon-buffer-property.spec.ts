// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createBufferGraph,
  createBufferResult,
  getPolygonsArea,
  isInsidePolygons,
  readResultPolygons
} from './polygon-buffer-test-utils';
import fixtureJson from './fixtures/buffer-properties.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/buffer-properties.json, written by test/oracles/geometry-properties.py with
// shapely.buffer(distance, quad_segs=8, cap_style='round', join_style='round'): six random
// polylines with switchbacks closer than 2 d and four random concave polygons. The fixture states the
// area tolerance and its derivation (both polygonal round joins lie below the true buffer by at most
// sagittaRatio * d^2 * turnSum, so their areas differ by at most that much), plus the
// polygon and hole counts, which are stable under d * (1 +/- 0.02), and sample points away from the
// boundary.

type BufferCase = {
  kind: 'line' | 'polygon';
  distance: number;
  ring: number[][];
  area: number;
  polygonCount: number;
  holeCount: number;
  areaTolerance: number;
  samples: [number, number, number][];
};

const fixture: {quadSegments: number; cases: BufferCase[]} = fixtureJson as unknown as {
  quadSegments: number;
  cases: BufferCase[];
};

async function runCase(testCase: BufferCase) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'polygon-buffer-property'});
  const graphBuffers = createBufferGraph(device, graph);
  const result = createBufferResult(graphBuffers, 'result', 8192, 128, 16384);
  const coordinates = testCase.ring.flat();
  const vertexCount = testCase.ring.length;
  const positions = graphBuffers.input(
    'positions',
    Float32Array.from(coordinates),
    'float32x2',
    vertexCount
  );
  const parameters = graphBuffers.input(
    'parameters',
    Float32Array.of(testCase.distance, 5, 0, 0),
    'float32',
    4
  );
  const geometry =
    testCase.kind === 'line'
      ? {
          kind: 'lines' as const,
          positions,
          lineOffsets: graphBuffers.input(
            'line-offsets',
            Uint32Array.of(0, vertexCount),
            'uint32',
            2
          )
        }
      : {
          kind: 'polygons' as const,
          positions,
          featureOffsets: graphBuffers.input('feature-offsets', Uint32Array.of(0, 1), 'uint32', 2),
          polygonOffsets: graphBuffers.input('polygon-offsets', Uint32Array.of(0, 1), 'uint32', 2),
          ringOffsets: graphBuffers.input(
            'ring-offsets',
            Uint32Array.of(0, vertexCount),
            'uint32',
            2
          )
        };
  graph.add(
    new GPUBufferSurface({
      geometry,
      parameters,
      joinStyle: 'round',
      capStyle: 'round',
      quadSegments: fixture.quadSegments,
      vertexTolerance: 1e-5,
      capacity: {intersections: 16384, nodedSegments: 16384},
      output: result.output
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const polygons = await readResultPolygons(result);
  const [overflow] = await readUint32(result.overflow, 1);
  compiled.destroy();
  for (const buffer of graphBuffers.buffers) {
    buffer.destroy();
  }
  return {polygons, overflow};
}

for (const [index, testCase] of fixture.cases.entries()) {
  it(`GPUBufferSurface matches shapely.buffer (round, quad 8): ${testCase.kind} ${index}`, async () => {
    const run = await runCase(testCase);
    if (!run) {
      return;
    }
    expect(run.overflow).toBe(0);
    // Relative area error below the derived polygonal-approximation tolerance.
    const area = getPolygonsArea(run.polygons);
    expect(
      Math.abs(area - testCase.area),
      `area ${area} vs shapely ${testCase.area} (relative ${Math.abs(area - testCase.area) / testCase.area})`
    ).toBeLessThanOrEqual(testCase.areaTolerance);
    expect(run.polygons.length, 'polygon count').toBe(testCase.polygonCount);
    expect(
      run.polygons.reduce((total, rings) => total + rings.length - 1, 0),
      'hole count'
    ).toBe(testCase.holeCount);
    // The buffer contains every input vertex (each is at distance >= d inside the boundary).
    const vertexCount =
      testCase.kind === 'polygon' ? testCase.ring.length - 1 : testCase.ring.length;
    for (const [x, y] of testCase.ring.slice(0, vertexCount)) {
      expect(isInsidePolygons(x, y, run.polygons), `vertex ${x}, ${y}`).toBe(true);
    }
    // Membership away from the shapely boundary (> 2 sagittas of either approximation).
    const mismatches = testCase.samples.filter(
      ([x, y, expected]) => isInsidePolygons(x, y, run.polygons) !== (expected === 1)
    );
    expect(mismatches).toEqual([]);
  });
}
