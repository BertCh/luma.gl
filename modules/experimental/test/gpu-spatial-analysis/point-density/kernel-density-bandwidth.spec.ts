// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity
} from '../../../src/gpu-spatial-analysis/point-density';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {getBandwidthOracle} from './kernel-density-bandwidth-oracle';
import fixtureJson from '../../oracles/fixtures/geometry-kernel-density.json';
import {requireTestDevice} from '../../utils/require-test-device';

type Case = {
  offset: [number, number];
  weighted: boolean;
  points: number[];
  rules: Record<'scott' | 'silverman', {isotropic: number; effectiveSampleSize: number}>;
  cells: [number, number][];
  intensity: number[];
};
type Fixture = {
  scipy: string;
  width: number;
  height: number;
  size: number;
  weights: number[];
  cases: Case[];
};

const fixture = fixtureJson as unknown as Fixture;

/**
 * Bandwidth tolerance, relative 5e-5. Derivation: positions and weights are identical f32 values
 * for GPU and oracle, and `position - extentMin` is exact (both are multiples of the f32 spacing
 * of the offset, 0.0625 at 1e6 and 0.25 at 4e6, and the difference is below 2^24 spacings). The
 * moments are sums of 240 nonnegative terms through a hierarchical f32 reduction, each with
 * relative error at most (log2(240) + 1) * 2^-24 = 5e-7; the centered second moment has the same
 * relative error (the mean error enters squared, about 1e-14), and sqrt halves it. `pow` and
 * `sqrt` on a GPU are within a few ulp (about 2e-7). The sum stays below 3e-6, so 5e-5 leaves
 * more than 15x headroom and still fails for an uncentered, raw-coordinate moment: at the 1e6 offset
 * `E[x^2]` is about 1e12 with f32 rounding of about 6e4, against a variance of a few hundred.
 */
const BANDWIDTH_TOLERANCE = 5e-5;

/**
 * Density tolerance, relative to the largest sampled oracle value, 5e-4. Derivation: the GPU
 * bandwidth differs from the oracle's by up to 5e-5; the intensity depends on `h` with
 * `d ln(K_h) / d ln(h) = t - 2` per kernel term (`t = d^2 / h^2 < 25`), so a cell value moves by at
 * most about 23 * 5e-5 = 1.2e-3 of itself, and by far less of the peak where the tail cells are small; the
 * f32 sum of 240 kernel terms adds under 1e-6.
 */
const DENSITY_TOLERANCE = 5e-4;

for (const rule of ['scott', 'silverman'] as const) {
  it(`GPUKernelDensity ${rule} bandwidthRule matches scipy.stats.gaussian_kde (${fixture.scipy})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const testCase of fixture.cases) {
      const pointCount = testCase.points.length / 2;
      const cellCount = fixture.width * fixture.height;
      const extent = [
        testCase.offset[0],
        testCase.offset[1],
        testCase.offset[0] + fixture.size,
        testCase.offset[1] + fixture.size
      ] as [number, number, number, number];
      const buffers: Buffer[] = [];
      const track = (buffer: Buffer) => {
        buffers.push(buffer);
        return buffer;
      };
      const parameterBuffer = new GPUParameterBuffer(device, {
        id: 'kernel-density-bandwidth-parameters',
        format: 'float32',
        length: 9
      });
      const valuesOut = track(createOutputBuffer(device, cellCount));
      const bandwidthOut = track(createOutputBuffer(device, 1));
      const graph = new GPUCommandGraph(device, {id: 'kernel-density-bandwidth'});
      graph.add(
        new GPUKernelDensity({
          id: 'kde',
          positions: importGraphBuffer(
            graph,
            'positions',
            track(createInputBuffer(device, Float32Array.from(testCase.points))),
            'float32x2',
            pointCount
          ),
          weights: testCase.weighted
            ? importGraphBuffer(
                graph,
                'weights',
                track(createInputBuffer(device, Float32Array.from(fixture.weights))),
                'float32',
                pointCount
              )
            : undefined,
          parameters: parameterBuffer.importToGraph(graph),
          grid: makeExtentGrid(fixture.width, fixture.height, extent),
          indexGridSize: [10, 10],
          // Cover the whole cloud (clipped to 1..99 around the offset) with a margin.
          indexBounds: [extent[0] - 1, extent[1] - 1, extent[2] + 1, extent[3] + 1],
          kernel: 'gaussian',
          edgeCorrection: 'none',
          bandwidthRule: rule,
          output: {
            values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
            bandwidth: importGraphBuffer(graph, 'bandwidth-out', bandwidthOut, 'float32', 1)
          }
        })
      );
      const compiled = graph.compile();
      try {
        // The bandwidth in the parameters is ignored by the rule.
        parameterBuffer.write(
          getGPUKernelDensityParameterValues(
            {bandwidth: 1},
            makeExtentGrid(fixture.width, fixture.height, extent)
          )
        );
        submitGraph(device, compiled, undefined);
        const [bandwidth] = await readFloat32(bandwidthOut, 1);
        const label = `offset ${testCase.offset} weighted ${testCase.weighted}`;
        const expected = testCase.rules[rule].isotropic;
        expect(Math.abs(bandwidth - expected), `${label} bandwidth`).toBeLessThanOrEqual(
          BANDWIDTH_TOLERANCE * expected
        );
        // The TypeScript oracle agrees with scipy, so the fixture and the oracle stay in step.
        const oracle = getBandwidthOracle(
          testCase.points,
          testCase.weighted ? fixture.weights : undefined
        );
        expect(Math.abs(oracle.isotropic - expected)).toBeLessThanOrEqual(1e-12 * expected);
        const values = await readFloat32(valuesOut, cellCount);
        const scale = Math.max(...testCase.intensity.map(Math.abs));
        for (const [sample, [column, row]] of testCase.cells.entries()) {
          const actual = values[row * fixture.width + column];
          expect(
            Math.abs(actual - testCase.intensity[sample]),
            `${label} cell ${column},${row}`
          ).toBeLessThanOrEqual(DENSITY_TOLERANCE * scale);
        }
      } finally {
        compiled.destroy();
        parameterBuffer.destroy();
        for (const buffer of buffers) {
          buffer.destroy();
        }
      }
    }
  });
}
