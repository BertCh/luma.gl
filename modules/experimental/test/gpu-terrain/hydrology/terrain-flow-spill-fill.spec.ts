// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom} from '../terrain-oracle-utils';
import {createFillRunner} from './terrain-flow-spill-fill-gpu';
import {modelSpillFill} from './terrain-flow-spill-fill-model';
import {
  fillDepressionsPriorityFlood,
  makeSpillFillCases,
  makeSpiralCase
} from './terrain-flow-spill-fill-oracle';

/**
 * GPU specs of the spill graph depression fill (roadmap P3 #8). Expected values are bit-identical
 * to the literal Priority-Flood oracle, to the CPU model of the same algorithm, and to the previous
 * tiled relaxation run on the same device (the A/B in these specs).
 */

function expectSameBits(actual: Float32Array, expected: Float32Array, label: string) {
  expect(actual.length, label).toBe(expected.length);
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let cell = 0; cell < actual.length; cell++) {
    if (Number.isNaN(expected[cell])) {
      if (!Number.isNaN(actual[cell])) throw new Error(`${label}: cell ${cell} should be NaN`);
    } else if (actualBits[cell] !== expectedBits[cell]) {
      throw new Error(`${label}: cell ${cell} is ${actual[cell]}, expected ${expected[cell]}`);
    }
  }
}

it('the spill fill equals Priority-Flood, the CPU model and the relaxation on adversarial DEMs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of makeSpillFillCases()) {
    const {width, height, elevation} = testCase;
    const expected = fillDepressionsPriorityFlood(elevation, width, height);
    const spill = createFillRunner(device, 'spill', elevation, width, height);
    const relaxation = createFillRunner(device, 'relaxation', elevation, width, height);
    try {
      const spillResult = await spill.run();
      expect(spillResult.converged, testCase.name).toBe(1);
      expectSameBits(spillResult.filled, expected, `${testCase.name} (spill vs Priority-Flood)`);
      const model = modelSpillFill(elevation, width, height);
      expectSameBits(spillResult.filled, model.filled, `${testCase.name} (spill vs model)`);
      // The iteration count is the number of Boruvka rounds, which the model repeats exactly.
      expect(spillResult.iterations, testCase.name).toBe(model.boruvkaRounds);
      const relaxationResult = await relaxation.run();
      expect(relaxationResult.converged, testCase.name).toBe(1);
      expectSameBits(spillResult.filled, relaxationResult.filled, `${testCase.name} (A/B)`);
    } finally {
      spill.destroy();
      relaxation.destroy();
    }
  }
});

it('a 1024 x 1024 spiral corridor of width 1 fills exactly, where the relaxation needs sweeps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const corridorWidth of [1, 3]) {
    const testCase = makeSpiralCase(1024, corridorWidth);
    const expected = fillDepressionsPriorityFlood(testCase.elevation, 1024, 1024);
    const spill = createFillRunner(device, 'spill', testCase.elevation, 1024, 1024);
    try {
      const result = await spill.run();
      expect(result.converged, testCase.name).toBe(1);
      expectSameBits(result.filled, expected, testCase.name);
      // Logarithmic: a million cells need at most floor(log2 n) + 2 = 21 rounds.
      expect(result.iterations, testCase.name).toBeLessThanOrEqual(21);
    } finally {
      spill.destroy();
    }
  }
});

it('a lake over a few hundred tiles and random pits at 700 x 650 match Priority-Flood', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 700;
  const height = 650;
  const random = createSeededRandom(5);
  const bowl = Float32Array.from({length: width * height}, (_, cell) => {
    const radius = Math.hypot((cell % width) - 350, Math.floor(cell / width) - 325);
    return Math.min(300, Math.floor(radius)) + Math.floor(random() * 3);
  });
  const pits = Float32Array.from({length: width * height}, () => Math.floor(random() * 40) / 4);
  for (const [name, elevation] of [
    ['bowl', bowl],
    ['pits', pits]
  ] as const) {
    const expected = fillDepressionsPriorityFlood(elevation, width, height);
    const spill = createFillRunner(device, 'spill', elevation, width, height);
    try {
      const result = await spill.run();
      expect(result.converged, name).toBe(1);
      expectSameBits(result.filled, expected, name);
    } finally {
      spill.destroy();
    }
  }
});

it('fillEpsilon is a per-frame GPU switch: epsilon above 0 runs the relaxation, 0 the spill graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const testCase = makeSpiralCase(97, 1);
  const {width, height, elevation} = testCase;
  const spill = createFillRunner(device, 'spill', elevation, width, height);
  const relaxation = createFillRunner(device, 'relaxation', elevation, width, height);
  try {
    // One compiled graph, parameters changing between frames, no recompile.
    for (const epsilon of [0, 0.25, 0, 0.0625, Number.NaN, -3]) {
      const actual = await spill.run(epsilon);
      const expected = await relaxation.run(epsilon);
      expect(actual.converged, `epsilon ${epsilon}`).toBe(1);
      expect(expected.converged, `epsilon ${epsilon}`).toBe(1);
      expectSameBits(actual.filled, expected.filled, `epsilon ${epsilon}`);
    }
  } finally {
    spill.destroy();
    relaxation.destroy();
  }
});
