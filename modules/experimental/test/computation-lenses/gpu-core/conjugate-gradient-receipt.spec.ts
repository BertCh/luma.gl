// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {validateGPUComputationSidecars} from '@luma.gl/gpgpu';
import {expect, test} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import '../../../src/computation-lenses/scenarios/gpu-core/index';

test('CG receipt follows one unknown through the actual GPU updates', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const scenario = await getLensScenario('gpu-core.conjugate-gradient-program')!.load();
  const record = await scenario.run(device);
  expect(validateGPUComputationSidecars(record, record)).toMatchObject({valid: true});
  const snapshots = record.captures!.snapshots;
  const final = (name: string) => snapshots.filter(capture => capture.name === name).at(-1)!;
  const trajectory = (name: string) => snapshots.filter(capture => capture.name === name);
  const columns = Array.from(final('matrixColumns').values);
  const coefficients = Array.from(final('matrixValues').values);
  expect(columns).toEqual([118, 133, 134, 135, 150]);
  expect(coefficients).toEqual([-4, -4, 17, -4, -4]);
  expect(final('matrixValues').components).toBe(5);
  const solutions = trajectory('solution');
  const residuals = trajectory('residual');
  const alphas = trajectory('alpha');
  const searches = trajectory('searchBeforeUpdate');
  const residualNorms = record.metrics![0].series!.find(series => series.name === 'residual')!;
  expect(solutions).toHaveLength(residualNorms.values.length);
  expect(residuals).toHaveLength(solutions.length);
  expect(alphas).toHaveLength(solutions.length - 1);
  expect(searches).toHaveLength(alphas.length);
  expect(solutions[0].values[0]).toBe(0);
  expect(residuals[0].values[0]).toBe(final('elevation').values[134]);
  for (let iteration = 1; iteration < solutions.length; iteration++) {
    // CPU arithmetic is only the oracle for the captured GPU MADD.
    const expected =
      solutions[iteration - 1].values[0] +
      alphas[iteration - 1].values[0] * searches[iteration - 1].values[0];
    expect(Math.abs(solutions[iteration].values[0] - expected)).toBeLessThan(0.002);
    expect(solutions[iteration].iteration).toBe(iteration);
  }
  const smoothed = final('smoothed').values;
  expect(solutions.at(-1)!.values[0]).toBe(smoothed[134]);
  const expectedResidual =
    final('elevation').values[134] -
    columns.reduce((sum, column, index) => sum + coefficients[index] * smoothed[column], 0);
  expect(Math.abs(residuals.at(-1)!.values[0] - expectedResidual)).toBeLessThan(0.02);
  expect(snapshots.every(capture => capture.provenance === 'gpu-readback')).toBe(true);
  expect(record.captures!.replays).toHaveLength(0);
});
