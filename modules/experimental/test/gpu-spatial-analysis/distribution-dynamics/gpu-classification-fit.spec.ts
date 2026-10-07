// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUClassificationFit,
  GPU_CLASSIFICATION_FIT_ADAM,
  GPU_CLASSIFICATION_FIT_ADCM,
  GPU_CLASSIFICATION_FIT_GADF,
  GPU_CLASSIFICATION_FIT_GVF,
  GPU_CLASSIFICATION_FIT_TSS,
  GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-classification-fit';
import {MAPCLASSIFY_FIT_CASES, MAPCLASSIFY_FIT_VALUES} from './mapclassify-fit-fixture';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';

const NO_CLASS = 0xffffffff;

/** Formula oracle (mapclassify 2.8 adcm/gadf/tss/gvf definitions, numpy median), float64. */
function computeFitOracle(
  values: number[],
  classes: number[],
  classCount: number,
  mask?: number[]
) {
  const median = (xs: number[]) => {
    const sorted = [...xs].sort((a, b) => a - b);
    const n = sorted.length;
    return n === 0 ? 0 : (sorted[(n - 1) >> 1] + sorted[n >> 1]) / 2;
  };
  const slots = [] as number[][];
  for (let slot = 0; slot <= classCount; slot++) slots.push([]);
  values.forEach((value, row) => {
    if (!Number.isFinite(value) || classes[row] >= classCount || (mask && !mask[row])) return;
    slots[classes[row]].push(value);
    slots[classCount].push(value);
  });
  const medians = slots.map(median);
  const absolute = slots.map((xs, s) => xs.reduce((a, x) => a + Math.abs(x - medians[s]), 0));
  const squared = slots.map(xs => {
    const mean = xs.length ? xs.reduce((a, x) => a + x, 0) / xs.length : 0;
    return xs.reduce((a, x) => a + (x - mean) ** 2, 0);
  });
  const adcm = absolute.slice(0, classCount).reduce((a, b) => a + b, 0);
  const sdcm = squared.slice(0, classCount).reduce((a, b) => a + b, 0);
  const adam = absolute[classCount];
  const tss = squared[classCount];
  return {
    counts: slots.map(xs => xs.length),
    medians,
    absolute,
    squared,
    gadf: adam > 0 ? 1 - adcm / adam : 1,
    adcm,
    adam,
    gvf: tss > 0 ? 1 - sdcm / tss : 1
  };
}

function expectClose(actual: number[], expected: number[], label: string, relative = 1e-4): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Math.abs(actual[index] - value) > 1e-5 + relative * Math.abs(value)) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${value}`);
    }
  });
}

async function runFit(
  rig: AnalysisRig,
  values: number[],
  classes: number[],
  classCount: number,
  mask?: number[]
) {
  const output = {
    counts: rig.output('uint32', classCount + 1),
    medians: rig.output('float32', classCount + 1),
    absoluteDeviations: rig.output('float32', classCount + 1),
    squaredDeviations: rig.output('float32', classCount + 1),
    summary: rig.output('float32', GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH)
  };
  rig.run(
    new GPUClassificationFit({
      values: rig.input(Float32Array.from(values), 'float32'),
      classes: rig.input(Uint32Array.from(classes), 'uint32'),
      classCount,
      mask: mask ? rig.input(Uint32Array.from(mask), 'uint32') : undefined,
      output
    })
  );
  return {
    counts: await rig.readUint(output.counts),
    medians: await rig.readFloat(output.medians),
    absolute: await rig.readFloat(output.absoluteDeviations),
    squared: await rig.readFloat(output.squaredDeviations),
    summary: await rig.readFloat(output.summary)
  };
}

it('GPUClassificationFit matches the formula oracle on a hand-checked fixture', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // class 0: [1, 2, 4, 10] median 3 (even count averages the middles); class 1: [-5, 7, 9] median 7.
  const values = [10, 1, 7, 2, -5, 4, 9];
  const classes = [0, 0, 1, 0, 1, 0, 1];
  const rig = new AnalysisRig(device);
  const actual = await runFit(rig, values, classes, 2);
  expect(actual.counts).toEqual([4, 3, 7]);
  expectClose(actual.medians.slice(0, 2), [3, 7], 'class medians');
  // ADCM = (2+1+1+7) + (12+0+2) = 25; overall median 4, ADAM = 6+3+3+2+9+0+5 = 28.
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_ADCM]], [25], 'adcm');
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_ADAM]], [28], 'adam');
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GADF]], [1 - 25 / 28], 'gadf');
  const oracle = computeFitOracle(values, classes, 2);
  expectClose(actual.medians, oracle.medians, 'medians');
  expectClose(actual.absolute, oracle.absolute, 'absolute');
  expectClose(actual.squared, oracle.squared, 'squared');
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GVF]], [oracle.gvf], 'gvf');
  rig.destroy();
});

it('GPUClassificationFit matches the oracle with ties, negatives, NaN, no-class rows, mask and empty class', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createSeededRandom(7);
  const classCount = 5;
  const rowCount = 900;
  const values: number[] = [];
  const classes: number[] = [];
  const mask: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    // Quantized values create many exact ties; class 3 is left empty.
    values.push(Math.round((random() - 0.4) * 40) / 4);
    let classId = Math.floor(random() * classCount);
    if (classId === 3) classId = 4;
    classes.push(classId);
    mask.push(random() < 0.9 ? 1 : 0);
  }
  values[5] = NaN;
  values[6] = Infinity;
  values[7] = -Infinity;
  classes[8] = NO_CLASS;
  values[9] = -0;
  values[10] = 0;
  classes[9] = classes[10] = 1;
  const rig = new AnalysisRig(device);
  const actual = await runFit(rig, values, classes, classCount, mask);
  const oracle = computeFitOracle(values, classes, classCount, mask);
  expect(actual.counts).toEqual(oracle.counts);
  expect(actual.counts[3]).toBe(0);
  expect(new Set(oracle.counts).size).toBeGreaterThan(3);
  expectClose(actual.medians, oracle.medians, 'medians', 1e-6);
  expectClose(actual.absolute, oracle.absolute, 'absolute');
  expectClose(actual.squared, oracle.squared, 'squared', 1e-3);
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GADF]], [oracle.gadf], 'gadf');
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GVF]], [oracle.gvf], 'gvf', 1e-3);
  expect(oracle.gadf).toBeGreaterThan(-1);
  // Deterministic: a second run is bit-identical.
  const again = await runFit(new AnalysisRig(device), values, classes, classCount, mask);
  expect(again.absolute).toEqual(actual.absolute);
  rig.destroy();
});

it('GPUClassificationFit defines a constant column as a perfect fit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rig = new AnalysisRig(device);
  const actual = await runFit(rig, [3, 3, 3, 3], [0, 1, 0, 1], 2);
  expect(actual.summary[GPU_CLASSIFICATION_FIT_GADF]).toBe(1);
  expect(actual.summary[GPU_CLASSIFICATION_FIT_GVF]).toBe(1);
  rig.destroy();
});

it('GPUClassificationFit reproduces pinned mapclassify 2.11.0 adcm, gadf and tss', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const [name, expected] of Object.entries(MAPCLASSIFY_FIT_CASES)) {
    const rig = new AnalysisRig(device);
    const actual = await runFit(rig, MAPCLASSIFY_FIT_VALUES, expected.classes, expected.k);
    expect(actual.counts.slice(0, expected.k), `${name} counts`).toEqual(expected.counts);
    expectClose(actual.medians.slice(0, expected.k), expected.medians, `${name} medians`, 1e-6);
    expectClose([actual.summary[GPU_CLASSIFICATION_FIT_ADCM]], [expected.adcm], `${name} adcm`);
    expectClose([actual.summary[GPU_CLASSIFICATION_FIT_ADAM]], [expected.adam], `${name} adam`);
    expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GADF]], [expected.gadf], `${name} gadf`);
    expectClose([actual.summary[GPU_CLASSIFICATION_FIT_TSS]], [expected.tss], `${name} tss`);
    expectClose(
      [actual.medians[expected.k]],
      [expected.overallMedian],
      `${name} overall median`,
      1e-6
    );
    rig.destroy();
  }
});

it('GPUClassificationFit handles an empty class and a constant column like mapclassify', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const rig = new AnalysisRig(device);
  // mapclassify UserDefined([1, 2, 10, 11], bins=[2, 5, 10, 11]): counts [2, 0, 1, 1], adcm 1.0.
  const actual = await runFit(rig, [1, 2, 10, 11], [0, 0, 2, 3], 4);
  expect(actual.counts).toEqual([2, 0, 1, 1, 4]);
  expect(actual.medians[1]).toBe(0);
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_ADCM]], [1], 'adcm');
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GADF]], [0.9444444444444444], 'gadf');
  rig.destroy();
});

it('GPUClassificationFit reduces a heavily skewed class split across many lanes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createSeededRandom(11);
  const classCount = 4;
  const rowCount = 20011;
  const values: number[] = [];
  const classes: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    values.push(Math.round(random() * 400) / 8 - 10);
    // 97% in class 0, a handful in class 2, none in class 1 or 3.
    classes.push(random() < 0.97 ? 0 : random() < 0.5 ? 2 : NO_CLASS);
  }
  const rig = new AnalysisRig(device);
  const actual = await runFit(rig, values, classes, classCount);
  const oracle = computeFitOracle(values, classes, classCount);
  expect(actual.counts).toEqual(oracle.counts);
  expect(actual.counts[0]).toBeGreaterThan(19000);
  expectClose(actual.medians, oracle.medians, 'medians', 1e-6);
  expectClose(actual.absolute, oracle.absolute, 'absolute', 1e-4);
  expectClose(actual.squared, oracle.squared, 'squared', 1e-3);
  expectClose([actual.summary[GPU_CLASSIFICATION_FIT_GADF]], [oracle.gadf], 'gadf', 1e-3);
  rig.destroy();
});
