// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Runs the real GPUDelaunayTessellation WGSL on the CPU (wgsl_reflect's interpreter, see
// segment-intersection/wgsl-graph-executor.ts) on small inputs: the parallel insertion and flip
// kernels, the gates, the canonical output sort. Oracles: T = 2n - 2 - h and the exact empty-circle
// check of delaunay-reference.ts (BigInt on the f32 inputs).

import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation';
import {getDelaunayStepCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  checkTriangulation,
  countHullBoundarySites,
  createUniformFixture,
  digestTriangleSet,
  mulberry32,
  SCIPY_FIXTURES,
  toTriangleRows
} from './delaunay-reference';

// The stable sorts are upstream radix/bitonic kernels with workgroup barriers: CPU stand-ins.
useCPUExecutorTestHooks();

function runDelaunayOnCPU(
  coordinates: Float32Array,
  triangleCapacity: number,
  maximumSteps?: number,
  continueOnGPU = false
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'delaunay-cpu'});
  const pointCount = coordinates.length / 2;
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const positions = createTransientView(graph, 'positions', 'float32x2', pointCount);
  const triangles = createTransientView(graph, 'triangles', 'uint32x3', triangleCapacity);
  const status = {
    count: scalar('count'),
    requiredCount: scalar('required-count'),
    overflow: scalar('overflow'),
    invalidCount: scalar('invalid-count'),
    duplicateCount: scalar('duplicate-count'),
    converged: scalar('converged'),
    iterationCount: scalar('iteration-count'),
    iterationLimitReached: scalar('iteration-limit-reached')
  };
  const nodes = new GPUDelaunayTessellation({
    positions,
    output: {triangles, status}
  }).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(positions, coordinates);
  executeCommandNodesOnCPU(
    maximumSteps === undefined
      ? nodes
      : nodes.filter(node => {
          if (node.id.includes('-continue-')) {
            return continueOnGPU;
          }
          const step = /-step-(\d+)-/.exec(node.id);
          return !step || Number(step[1]) < maximumSteps;
        }),
    memory
  );
  const read = (view: typeof status.count) => memory.readUint32(view)[0];
  const written = read(status.count);
  const activeSteps = [...memory.workgroupCounts.entries()].filter(
    ([nodeId, count]) => nodeId.endsWith('-flip-evaluate') && count > 0
  ).length;
  return {
    written,
    required: read(status.requiredCount),
    overflow: read(status.overflow),
    invalid: read(status.invalidCount),
    duplicates: read(status.duplicateCount),
    converged: read(status.converged),
    iterationCount: read(status.iterationCount),
    iterationLimitReached: read(status.iterationLimitReached),
    indices: Array.from(memory.readUint32(triangles, 3 * written)),
    activeSteps
  };
}

function expectCanonicalOrder(indices: number[]): void {
  for (let row = 0; row < indices.length / 3; row++) {
    const [a, b, c] = indices.slice(3 * row, 3 * row + 3);
    expect(a).toBeLessThan(b);
    expect(a).toBeLessThan(c);
    if (row > 0) {
      const [previousA, previousB] = indices.slice(3 * row - 3, 3 * row - 1);
      expect(previousA < a || (previousA === a && previousB < b)).toBe(true);
    }
  }
}

const CLEAN = {degenerate: 0, circleViolations: 0, duplicateEdges: 0, areaDifference: 0n};

describe('GPUDelaunayTessellation WGSL executed on the CPU', () => {
  it('seeds a noncollinear simplex on 2000 collinear-plus-apex sites', () => {
    const pointCount = 2000;
    const coordinates = Float32Array.from(
      Array.from({length: pointCount}, (_, row) =>
        row === pointCount - 1 ? [pointCount / 2, 1] : [row, 0]
      ).flat()
    );
    const result = runDelaunayOnCPU(coordinates, 2 * pointCount);
    const expectedCount = 2 * pointCount - 2 - countHullBoundarySites(coordinates);
    expect(expectedCount).toBe(pointCount - 2);
    expect(result).toMatchObject({
      written: expectedCount,
      required: expectedCount,
      invalid: 0,
      converged: 1,
      iterationLimitReached: 0
    });
    expect(result.iterationCount).toBe(result.activeSteps);
    expect(result.iterationCount).toBeLessThan(100);
    expect(checkTriangulation(coordinates, toTriangleRows(result.indices))).toEqual(CLEAN);
    expectCanonicalOrder(result.indices);
  }, 1_200_000);

  it('finishes insertion and exact legalization in the bounded GPU continuation', () => {
    const coordinates = createUniformFixture(19, 40, 100, 100);
    // Stop the parallel schedule after the seed simplex, forcing both continuation kernels.
    const continued = runDelaunayOnCPU(coordinates, 80, 3, true);
    const parallel = runDelaunayOnCPU(coordinates, 80);
    expect(continued).toMatchObject({
      written: 2 * 40 - 2 - countHullBoundarySites(coordinates),
      invalid: 0,
      overflow: 0,
      converged: 1,
      iterationLimitReached: 0
    });
    expect(continued.iterationCount).toBeGreaterThan(continued.activeSteps);
    expect(continued.indices).toEqual(parallel.indices);
    expect(checkTriangulation(coordinates, toTriangleRows(continued.indices))).toEqual(CLEAN);
    expectCanonicalOrder(continued.indices);
  }, 600_000);

  it('reports iteration exhaustion separately from invalid input and output capacity', () => {
    const coordinates = Float32Array.of(0, 0, 1, 0, 0, 1, 0.25, 0.25);
    // Truncate the encoded continuation to force the same finalize path as budget exhaustion.
    expect(runDelaunayOnCPU(coordinates, 8, 1)).toMatchObject({
      written: 0,
      invalid: 1,
      converged: 0,
      iterationCount: 1,
      iterationLimitReached: 1
    });
    expect(runDelaunayOnCPU(coordinates, 1)).toMatchObject({
      written: 1,
      required: 3,
      overflow: 1,
      converged: 1,
      iterationLimitReached: 0
    });
    expect(runDelaunayOnCPU(Float32Array.of(0, 0, Number.NaN, 0), 4)).toMatchObject({
      written: 0,
      invalid: 1,
      converged: 0,
      iterationCount: 0,
      iterationLimitReached: 0
    });
    expect(runDelaunayOnCPU(new Float32Array(), 1)).toMatchObject({
      written: 0,
      invalid: 0,
      converged: 1,
      iterationCount: 0,
      iterationLimitReached: 0
    });
  });
  it('triangulates the square with its centre and drops the duplicate', () => {
    const result = runDelaunayOnCPU(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5, 0, 0), 8);
    expect(result).toMatchObject({written: 4, required: 4, overflow: 0, invalid: 0, duplicates: 1});
    expect(result.indices).toEqual([0, 1, 4, 0, 4, 3, 1, 2, 4, 2, 3, 4]);
  });

  it('handles collinear, cocircular, lattice and capacity cases', () => {
    expect(runDelaunayOnCPU(Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0), 4)).toMatchObject({
      written: 0,
      required: 0,
      invalid: 0
    });
    const square = Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1);
    const first = runDelaunayOnCPU(square, 4);
    expect(first).toMatchObject({written: 2, required: 2, invalid: 0});
    // Simulation of Simplicity on rows: row 0 has the largest lift perturbation, so it lies
    // outside the circle of 1, 2, 3 and the diagonal 1-3 avoids it.
    expect(first.indices).toEqual([0, 1, 3, 1, 2, 3]);
    expect(checkTriangulation(square, toTriangleRows(first.indices))).toEqual(CLEAN);

    const circle = Float32Array.from(
      [
        [5, 0],
        [4, 3],
        [3, 4],
        [0, 5],
        [-3, 4],
        [-4, 3],
        [-5, 0],
        [-4, -3],
        [-3, -4],
        [0, -5],
        [3, -4],
        [4, -3]
      ].flat()
    );
    const cocircular = runDelaunayOnCPU(circle, 24);
    expect(cocircular).toMatchObject({written: 10, required: 10, invalid: 0});
    expect(checkTriangulation(circle, toTriangleRows(cocircular.indices))).toEqual(CLEAN);

    const lattice = Float32Array.from(
      Array.from({length: 25}, (_, index) => [index % 5, Math.floor(index / 5)]).flat()
    );
    const latticeResult = runDelaunayOnCPU(lattice, 50);
    expect(latticeResult).toMatchObject({written: 32, required: 32, invalid: 0});
    expect(checkTriangulation(lattice, toTriangleRows(latticeResult.indices))).toEqual(CLEAN);
    expectCanonicalOrder(latticeResult.indices);

    const bounded = runDelaunayOnCPU(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5), 2);
    expect(bounded).toMatchObject({written: 2, required: 4, overflow: 1});
    expect(runDelaunayOnCPU(Float32Array.of(0, 0, 1, 0, Number.NaN, 0, 0, 1), 4)).toMatchObject({
      written: 0,
      invalid: 1
    });
  });

  it('matches the exact empty-circle oracle on random sites', () => {
    for (const seed of [11, 12, 13]) {
      const random = mulberry32(seed);
      const count = 40;
      const coordinates = new Float32Array(2 * count);
      for (let index = 0; index < 2 * count; index++) {
        coordinates[index] = Math.fround(random() * 100);
      }
      const result = runDelaunayOnCPU(coordinates, 2 * count);
      expect(result.invalid, `seed ${seed}`).toBe(0);
      expect(result.required, `seed ${seed}`).toBe(result.written);
      expect(
        checkTriangulation(coordinates, toTriangleRows(result.indices)),
        `seed ${seed}`
      ).toEqual(CLEAN);
      expectCanonicalOrder(result.indices);
    }
  });

  // scipy.spatial 1.18.1 digests (delaunay-reference.ts): uniform, 100:1 band and 1e-2 band. The
  // full hull survives (T = 2n - 2 - h) and the steps stay well under getDelaunayStepCount.
  for (const [name, fixture] of Object.entries(SCIPY_FIXTURES)) {
    it(`reproduces the scipy triangle set on ${name}`, () => {
      const coordinates = createUniformFixture(
        fixture.seed,
        fixture.count,
        fixture.width,
        fixture.height
      );
      const result = runDelaunayOnCPU(coordinates, 2 * fixture.count);
      expect(result).toMatchObject({
        written: 2 * fixture.count - 2 - fixture.hull,
        required: fixture.triangles,
        invalid: 0,
        duplicates: 0
      });
      const triangles = toTriangleRows(result.indices);
      expect(digestTriangleSet(triangles)).toBe(fixture.digest);
      const continued = runDelaunayOnCPU(coordinates, 2 * fixture.count, 3, true);
      expect(continued).toMatchObject({invalid: 0, converged: 1, required: fixture.triangles});
      expect(digestTriangleSet(toTriangleRows(continued.indices))).toBe(fixture.digest);
      expect(checkTriangulation(coordinates, triangles)).toEqual(CLEAN);
      console.log(
        `${name}: ${result.activeSteps} active steps of ${getDelaunayStepCount(fixture.count)}`
      );
      expect(result.activeSteps).toBeLessThan(getDelaunayStepCount(fixture.count) / 2);
    }, 600000);
  }
});
