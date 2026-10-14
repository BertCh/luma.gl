// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getMultipleTestingAdjustedPValues,
  GPUMultipleTestingAdjustment
} from '../../../src/gpu-spatial-analysis/multiple-testing';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {MULTIPLE_TESTING_FIXTURES} from './multiple-testing-fixtures';
import {requireTestDevice} from '../../utils/require-test-device';

const GARBAGE = 0x7f7f7f7f;
const METHODS: GPUMultipleTestingMethod[] = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
];

/** Runs the node once and reads back the adjusted p-values and the family size. */
async function runAdjustment(
  device: Device,
  pValues: ArrayLike<number>,
  method: GPUMultipleTestingMethod,
  minimumFamilySize: number
): Promise<{adjusted: number[]; familySize: number}> {
  const rows = pValues.length;
  const input = createInputBuffer(device, Float32Array.from(pValues));
  const adjustedBuffer = createOutputBuffer(device, rows);
  const testedBuffer = createOutputBuffer(device, 1);
  adjustedBuffer.write(new Uint32Array(rows).fill(GARBAGE));
  testedBuffer.write(new Uint32Array(1).fill(GARBAGE));
  const graph = new GPUCommandGraph(device, {id: 'multiple-testing-adjustment'});
  graph.add(
    new GPUMultipleTestingAdjustment({
      pValues: importGraphBuffer(graph, 'p-values', input, 'float32', rows),
      method,
      minimumFamilySize,
      output: {
        adjustedPValues: importGraphBuffer(graph, 'adjusted', adjustedBuffer, 'float32', rows),
        testedCount: importGraphBuffer(graph, 'tested-count', testedBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const adjusted = await readFloat32(adjustedBuffer, rows);
  const familySize = (await readUint32(testedBuffer, 1))[0];
  compiled.destroy();
  for (const buffer of [input, adjustedBuffer, testedBuffer]) {
    buffer.destroy();
  }
  return {adjusted, familySize};
}

/** f32 tolerance: relative 1e-6 plus absolute 1e-7. NaN must match NaN. */
function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string): void {
  expect(actual.length, label).toBe(expected.length);
  for (let row = 0; row < expected.length; row++) {
    if (Number.isNaN(expected[row])) {
      expect(actual[row], `${label} row ${row}`).toBeNaN();
    } else {
      expect(Math.abs(actual[row] - expected[row]), `${label} row ${row}`).toBeLessThanOrEqual(
        1e-7 + 1e-6 * Math.abs(expected[row])
      );
    }
  }
}

it('GPUMultipleTestingAdjustment matches the statsmodels oracle on every fixture and method', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of MULTIPLE_TESTING_FIXTURES) {
    const testedRows = fixture.pValues.filter(p => Number.isFinite(p)).length;
    const familySize = Math.max(testedRows, fixture.minimumFamilySize);
    for (const method of METHODS) {
      const label = `${fixture.name} ${method}`;
      const result = await runAdjustment(
        device,
        fixture.pValues,
        method,
        fixture.minimumFamilySize
      );
      expect(result.familySize, `${label} family size`).toBe(familySize);
      expectClose(
        result.adjusted,
        method === 'none' ? fixture.pValues : fixture.expected[method],
        label
      );
    }
  }
});

it('GPUMultipleTestingAdjustment gives tied p-values identical adjusted values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = MULTIPLE_TESTING_FIXTURES.find(entry => entry.name === 'ties')!;
  for (const method of METHODS) {
    const {adjusted} = await runAdjustment(device, fixture.pValues, method, 0);
    for (let a = 0; a < fixture.pValues.length; a++) {
      for (let b = 0; b < fixture.pValues.length; b++) {
        if (fixture.pValues[a] === fixture.pValues[b]) {
          expect(adjusted[a], `${method} ${a} ${b}`).toBe(adjusted[b]);
        }
      }
    }
  }
});

it('GPUMultipleTestingAdjustment handles an all-untested family and out-of-range p-values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const method of METHODS) {
    const untested = await runAdjustment(device, [Number.NaN, Number.NaN, Number.NaN], method, 0);
    expect(untested.familySize).toBe(0);
    expect(untested.adjusted.every(Number.isNaN), method).toBe(true);
    const mixed = [0.5, -0.1, 1.5, Number.POSITIVE_INFINITY, 0.25];
    const result = await runAdjustment(device, mixed, method, 0);
    expect(result.familySize).toBe(2);
    expectClose(
      result.adjusted,
      getMultipleTestingAdjustedPValues(mixed, method),
      `${method} mixed`
    );
  }
});

it('GPUMultipleTestingAdjustment matches the CPU reference on 70k randomized rows (multi-block scans)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let state = 987654321;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const rows = 70000;
  // A mixture of strong signals, uniform nulls, exact ties (discrete p-values) and missing rows.
  const pValues = Float32Array.from({length: rows}, (_, row) => {
    if (row % 17 === 0) {
      return Number.NaN;
    }
    if (row % 5 === 0) {
      return (Math.floor(random() * 100) + 1) / 100;
    }
    return random() < 0.1 ? random() ** 4 * 1e-3 : random();
  });
  for (const [method, minimumFamilySize] of [
    ['bonferroni', 0],
    ['holm', 0],
    ['benjamini-hochberg', 0],
    ['benjamini-yekutieli', 0],
    ['benjamini-hochberg', 80000]
  ] as const) {
    const {adjusted, familySize} = await runAdjustment(device, pValues, method, minimumFamilySize);
    const reference = getMultipleTestingAdjustedPValues(pValues, method, minimumFamilySize);
    const testedRows = pValues.filter(p => Number.isFinite(p)).length;
    expect(familySize).toBe(Math.max(testedRows, minimumFamilySize));
    expectClose(adjusted, reference, `${method} ${minimumFamilySize}`);
  }
});
