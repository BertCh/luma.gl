// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Device} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {
  getGPUSpaceTimeParameterValues,
  GPUKnoxTest,
  GPUMantelTest,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {AnalysisRig, type CPUWeights} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeKnoxOracle, computeSequentialStop} from './space-time-oracle';
import {buildBand, createEvents, createStoppingEvents, uploadPairs} from './space-time-scenes';

type Scene = {csr: CPUWeights & {distances: number[]}; times: Float32Array};
type Run = {
  statistics: number[];
  summary: number[];
  permutationCount?: number;
  milliseconds: number;
};

/** Runs one Knox or Mantel test and reads everything back. `stopAfterExceedances` enables stopping. */
async function runTest(
  device: Device,
  kind: 'knox' | 'mantel',
  scene: Scene,
  settings: {
    seed: number;
    permutations: number;
    maximumPermutations: number;
    threshold?: number;
    stopAfterExceedances?: number;
  }
): Promise<Run> {
  const rig = new AnalysisRig(device);
  const {maximumPermutations, stopAfterExceedances} = settings;
  const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const permutationCount = stopAfterExceedances === undefined ? undefined : rig.output('uint32', 1);
  const parameters = rig.input(
    getGPUSpaceTimeParameterValues({
      seed: settings.seed,
      permutationCount: settings.permutations,
      timeThreshold: settings.threshold
    }),
    'uint32'
  );
  const common = {
    times: rig.input(scene.times, 'float32'),
    parameters,
    maximumPermutations,
    ...(stopAfterExceedances === undefined ? {} : {stopAfterExceedances})
  };
  const start = performance.now();
  let statistics: number[];
  if (kind === 'knox') {
    const output = rig.output('uint32', maximumPermutations + 1);
    rig.run(
      new GPUKnoxTest({
        ...common,
        pairs: rig.weights(scene.csr, 4),
        output: {statistics: output, summary, permutationCount}
      })
    );
    statistics = await rig.readUint(output);
  } else {
    const output = rig.output('float32', maximumPermutations + 1);
    rig.run(
      new GPUMantelTest({
        ...common,
        pairs: uploadPairs(rig, scene.csr),
        output: {statistics: output, summary, permutationCount}
      })
    );
    statistics = await rig.readFloat(output);
  }
  const summaryValues = await rig.readFloat(summary);
  const milliseconds = performance.now() - start;
  const used = permutationCount ? (await rig.readUint(permutationCount))[0] : undefined;
  rig.destroy();
  return {statistics, summary: summaryValues, permutationCount: used, milliseconds};
}

const index = GPU_SPACE_TIME_SUMMARY;

/**
 * Stopped runs equal the sequential rule applied to the full run: integer L, the same first L
 * statistics bit for bit, zero past L, and the Besag-Clifford p-value. The full run is the GPU's
 * own (the Knox one is also checked against the CPU swap-or-not oracle), so Mantel f32 near-ties
 * cannot flip a count between a CPU and a GPU reference. Three batches of 64 (maximumPermutations
 * 191) make a stop inside the first, a later and no batch.
 */
for (const kind of ['knox', 'mantel'] as const) {
  it(`${kind} stopAfterExceedances equals the sequential rule on the full run`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, times} = createStoppingEvents();
    const scene: Scene = {csr: buildBand(positions, kind === 'knox' ? 9 : 14), times};
    const settings = {seed: 7, permutations: 150, maximumPermutations: 191, threshold: 3};
    const full = await runTest(device, kind, scene, settings);
    if (kind === 'knox') {
      const oracle = computeKnoxOracle(scene.csr, times, 7, 150, 3);
      expect(full.statistics.slice(0, 151)).toEqual(oracle.statistics);
    }
    const stoppedLengths = new Set<number>();
    for (const stopAfterExceedances of [1, 2, 5, 20, 60, 191]) {
      const run = await runTest(device, kind, scene, {...settings, stopAfterExceedances});
      const expected = computeSequentialStop(full.statistics, 150, stopAfterExceedances);
      stoppedLengths.add(expected.permutationsUsed);
      expect(run.permutationCount, `h ${stopAfterExceedances}`).toBe(expected.permutationsUsed);
      expect(run.statistics.slice(0, expected.permutationsUsed + 1)).toEqual(
        full.statistics.slice(0, expected.permutationsUsed + 1)
      );
      expect(run.statistics.slice(expected.permutationsUsed + 1).every(value => value === 0)).toBe(
        true
      );
      expect(run.summary[index.greaterCount]).toBe(expected.greater);
      expect(run.summary[index.pseudoPGreater]).toBeCloseTo(expected.pValue, 6);
      expect(run.summary[index.observed]).toBe(full.summary[index.observed]);
    }
    // The chosen counts exercise distinct stopping indices, not one repeated value.
    expect(stoppedLengths.size).toBeGreaterThan(2);
  });

  // With h above the per-frame permutation count the rule never fires: the result is the full run
  // bit for bit (statistics and the whole summary), and the output reports P.
  it(`${kind} stopAfterExceedances above the permutation count equals the unstopped run`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions, times} = createEvents(200, 11);
    const scene: Scene = {csr: buildBand(positions, kind === 'knox' ? 8 : 12), times};
    const settings = {seed: 3, permutations: 70, maximumPermutations: 191, threshold: 4};
    const full = await runTest(device, kind, scene, settings);
    const stopped = await runTest(device, kind, scene, {...settings, stopAfterExceedances: 100});
    expect(stopped.permutationCount).toBe(70);
    expect(stopped.statistics).toEqual(full.statistics);
    expect(stopped.summary).toEqual(full.summary);
  });

  // Adversarial: two space-time clusters 1000 apart on a line, time proportional to position, and
  // spatial pairs only inside each cluster. The observed Knox count is every pair and the observed
  // Mantel r is 1 (time difference = distance / 2), the maxima; mixing the clusters or the order
  // lowers them, so no permutation reaches the observed value: the rule never stops.
  it(`${kind} stopAfterExceedances never stops when every permutation is below the observed`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = 40;
    const positions = new Float32Array(count * 2);
    const times = new Float32Array(count);
    for (let row = 0; row < count; row++) {
      const cluster = row < count / 2 ? 0 : 1;
      positions[row * 2] = cluster * 1000 + (row % 7);
      positions[row * 2 + 1] = 0;
      times[row] = positions[row * 2] * 0.5;
    }
    const scene: Scene = {csr: buildBand(positions, 20), times};
    const run = await runTest(device, kind, scene, {
      seed: 1,
      permutations: 99,
      maximumPermutations: 128,
      threshold: 5,
      stopAfterExceedances: 5
    });
    expect(run.permutationCount).toBe(99);
    expect(run.summary[index.greaterCount]).toBeLessThan(5);
    const expected = computeSequentialStop(run.statistics, 99, 5);
    expect(expected.stopped).toBe(false);
    expect(run.summary[index.pseudoPGreater]).toBeCloseTo(expected.pValue, 6);
  });

  // Adversarial: a threshold that makes every time pair close (Knox) or identical times (Mantel
  // correlation 0 for all): every permutation ties the observed, so the rule stops at L = h.
  it(`${kind} stopAfterExceedances stops at L = h when every permutation ties the observed`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {positions} = createEvents(120, 21);
    const times = new Float32Array(120).fill(2);
    const scene: Scene = {csr: buildBand(positions, 12), times};
    const run = await runTest(device, kind, scene, {
      seed: 5,
      permutations: 150,
      maximumPermutations: 191,
      threshold: 1,
      stopAfterExceedances: 10
    });
    expect(run.permutationCount).toBe(10);
    expect(run.summary[index.pseudoPGreater]).toBe(1);
    expect(run.statistics.slice(11).every(value => value === 0)).toBe(true);
  });

  // A/B: same run, full permutations against stopAfterExceedances = 10 on a clustered scene whose
  // p-value is small (the stopped run keeps going to P) and on a null scene where it stops early.
  // Logs both timings; asserts only that the results agree on the first L permutations.
  it(`${kind} A/B: stopAfterExceedances = 10 against the full run`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const events = createEvents(2000, 31);
    const nullTimes = Float32Array.from(
      events.times,
      (_, row) => events.times[(row * 7919) % 2000]
    );
    for (const [label, times] of [
      ['clustered', events.times],
      ['shuffled', nullTimes]
    ] as const) {
      const scene: Scene = {csr: buildBand(events.positions, kind === 'knox' ? 4 : 5), times};
      const settings = {seed: 9, permutations: 999, maximumPermutations: 999, threshold: 3};
      await runTest(device, kind, scene, {...settings, permutations: 31}); // warm-up
      const full = await runTest(device, kind, scene, settings);
      const stopped = await runTest(device, kind, scene, {...settings, stopAfterExceedances: 10});
      const used = stopped.permutationCount!;
      // eslint-disable-next-line no-console
      console.log(
        `${kind} ${label}: full ${full.milliseconds.toFixed(1)} ms, stopAfterExceedances 10: ` +
          `${stopped.milliseconds.toFixed(1)} ms, L ${used}`
      );
      expect(stopped.statistics.slice(0, used + 1)).toEqual(full.statistics.slice(0, used + 1));
    }
  });
}
