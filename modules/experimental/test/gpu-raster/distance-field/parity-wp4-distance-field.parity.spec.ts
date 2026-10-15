// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDistanceField,
  getGPUDistanceFieldParameterValues
} from '../../../src/gpu-raster/distance-field';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {distanceCases, decodePackedRuns} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const NONE = 0xffffffff;
for (const scene of distanceCases) {
  it(`GPUDistanceField independent full EDT parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const length = scene.width * scene.height;
    const [mask, effectiveMask, squared] = await Promise.all([
      decodePackedRuns(scene.mask, length),
      decodePackedRuns(scene.effectiveMask, length),
      decodePackedRuns(scene.squared, length)
    ]);
    const fixture = new ParityRasterGraph(device, scene.name);
    const {graph} = fixture;
    const grid = makeGPURasterMetadata({
      width: scene.width,
      height: scene.height,
      cellSize: [scene.cellSize[0], scene.cellSize[1]],
      rowDirection: 'north'
    });
    const distances = fixture.output('distances', 'float32', length);
    const allocation = fixture.output('allocation', 'uint32', length);
    const nearest = fixture.output('nearest', 'uint32', length);
    const within = fixture.output('within', 'uint32', length);
    graph.add(
      new GPUDistanceField({
        grid,
        mode: 'exact',
        parameters: fixture.input(
          'parameters',
          'float32',
          getGPUDistanceFieldParameterValues({maximumDistance: scene.maximumDistance}, grid)
        ),
        seedMask: fixture.input('mask', 'uint32', mask),
        ...(scene.positions.length
          ? {
              seedPositions: fixture.input(
                'positions',
                'float32x2',
                new Float32Array(scene.positions)
              ),
              seedIds: fixture.input('seed-ids', 'uint32', new Uint32Array(scene.ids)),
              seedCount: fixture.input('seed-count', 'uint32', Uint32Array.of(scene.count))
            }
          : {}),
        output: {
          distances: distances.view,
          allocation: allocation.view,
          nearestCells: nearest.view,
          withinDistance: within.view
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actualDistances = await readFloat32(distances.buffer, length);
      const actualAllocation = await readUint32(allocation.buffer, length);
      const actualNearest = await readUint32(nearest.buffer, length);
      const expectedDistances = Float64Array.from(squared, value =>
        value === NONE || Math.sqrt(value) > scene.maximumDistance ? Infinity : Math.sqrt(value)
      );
      let minimum = Infinity;
      let maximum = 0;
      for (const value of expectedDistances)
        if (Number.isFinite(value)) {
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        }
      const absolute = 1e-5 * Math.max(1, maximum - Math.min(minimum, maximum));
      const relative = 1e-4;
      let maxAbs = 0;
      let maxRel = 0;
      let mismatches = 0;
      for (let index = 0; index < length; index++) {
        const expected = expectedDistances[index];
        const actual = actualDistances[index];
        if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
          if (actual !== expected) mismatches++;
        } else {
          const error = Math.abs(actual - expected);
          maxAbs = Math.max(maxAbs, error);
          maxRel = Math.max(maxRel, error / Math.max(1e-300, Math.abs(expected)));
          if (error > absolute + relative * Math.abs(expected)) mismatches++;
        }
      }
      console.log(
        'PARITY',
        JSON.stringify({
          class: 'GPUDistanceField',
          case: scene.name,
          reference: scene.reference,
          n: length,
          metric: 'all-distances-and-infinity-mask',
          maxAbs,
          maxRel,
          mismatches,
          tolerance: {profile: 'R', absolute, relative}
        })
      );
      expect(mismatches).toBe(0);
      const expectedWithin = Uint32Array.from(expectedDistances, value =>
        Number.isFinite(value) ? 1 : 0
      );
      const compare = (metric: string, actual: ArrayLike<number>, expected: ArrayLike<number>) =>
        compareRasterParity(
          'GPUDistanceField',
          scene.name,
          scene.reference,
          metric,
          actual,
          expected,
          'E'
        );
      compare('all-within-distance-flags', await readUint32(within.buffer, length), expectedWithin);
      compare(
        'all-allocation-and-smallest-label-ties',
        actualAllocation,
        Uint32Array.from(expectedWithin, (active, index) =>
          active ? (scene.allocation.length ? scene.allocation[index] : 0) : NONE
        )
      );
      const validNearest = Uint32Array.from(actualNearest, (seed, index) => {
        if (!expectedWithin[index]) return seed === NONE ? 1 : 0;
        if (
          seed >= length ||
          effectiveMask[seed] === 0 ||
          effectiveMask[seed] - 1 !== actualAllocation[index]
        )
          return 0;
        const horizontal = ((seed % scene.width) - (index % scene.width)) * scene.cellSize[0];
        const vertical =
          (Math.floor(seed / scene.width) - Math.floor(index / scene.width)) * scene.cellSize[1];
        // Integers under 2^53: exact complete tie-set membership, independent of SciPy's chosen tie.
        return horizontal ** 2 + vertical ** 2 === squared[index] ? 1 : 0;
      });
      compare(
        'every-nearest-cell-is-in-exact-optimal-tie-set',
        validNearest,
        new Uint32Array(length).fill(1)
      );
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
