// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMapGraphParameterBuffer,
  importGraphBuffer,
  submitGraph
} from '../../../src/map-graphs/map-graph-utils';
import {
  GPUProfilePeaks,
  getGPUProfilePeaksParameterValues
} from '../../../src/map-graphs/terrain-features/gpu-profile-peaks';
import {
  createInputBuffer,
  createOutputBuffer,
  readCompactIds,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  concatenateProfiles,
  createCircularGaussianBumps,
  createGaussianBumps,
  findProfilePeaksOracle,
  type ProfilePeaksOracleResult
} from './profile-peaks-oracle';

type PeakOptions = {
  window: number;
  minProminence: number;
  minSide?: number;
  nms?: number;
  wrap?: boolean;
  nmsRounds?: number;
};

type PeakRun = {
  prominence: number[];
  refinedIndex: number[];
  refinedValue: number[];
  mask: number[];
  compactIds: number[];
  converged: number;
};

/** Compiles one graph and returns a function that encodes it again with a new minProminence. */
async function createPeakRunner(
  device: Device,
  values: Float32Array,
  validity: Uint32Array | undefined,
  offsets: Uint32Array,
  options: PeakOptions
) {
  const sampleCount = values.length;
  const graph = new GPUCommandGraph(device, {id: 'profile-peaks-test'});
  const valuesBuffer = createInputBuffer(device, values);
  const validityBuffer = validity ? createInputBuffer(device, validity) : undefined;
  const offsetsBuffer = createInputBuffer(device, offsets);
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'profile-peaks-settings',
    format: 'float32',
    length: 1,
    values: getGPUProfilePeaksParameterValues({minProminence: options.minProminence})
  });
  const prominenceBuffer = createOutputBuffer(device, sampleCount);
  const refinedIndexBuffer = createOutputBuffer(device, sampleCount);
  const refinedValueBuffer = createOutputBuffer(device, sampleCount);
  const maskBuffer = createOutputBuffer(device, sampleCount);
  const idsBuffer = createOutputBuffer(device, sampleCount);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const convergedBuffer = createOutputBuffer(device, 1);
  graph.add(
    new GPUProfilePeaks({
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', sampleCount),
      validity: validityBuffer
        ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', sampleCount)
        : undefined,
      offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', offsets.length),
      settings: parameterBuffer.importToGraph(graph),
      window: options.window,
      minSide: options.minSide,
      nms: options.nms,
      wrap: options.wrap,
      nmsRounds: options.nmsRounds,
      prominence: importGraphBuffer(graph, 'prominence', prominenceBuffer, 'float32', sampleCount),
      refinedIndex: importGraphBuffer(
        graph,
        'refined-index',
        refinedIndexBuffer,
        'float32',
        sampleCount
      ),
      refinedValue: importGraphBuffer(
        graph,
        'refined-value',
        refinedValueBuffer,
        'float32',
        sampleCount
      ),
      peakMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', sampleCount),
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', sampleCount),
        count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
      },
      converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  return {
    async run(minProminence: number = options.minProminence): Promise<PeakRun> {
      parameterBuffer.write(getGPUProfilePeaksParameterValues({minProminence}));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(overflowBuffer, 1)).toEqual([0]);
      return {
        prominence: await readFloat32(prominenceBuffer, sampleCount),
        refinedIndex: await readFloat32(refinedIndexBuffer, sampleCount),
        refinedValue: await readFloat32(refinedValueBuffer, sampleCount),
        mask: await readUint32(maskBuffer, sampleCount),
        compactIds: await readCompactIds(idsBuffer, countBuffer),
        converged: (await readUint32(convergedBuffer, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of [
        valuesBuffer,
        validityBuffer,
        offsetsBuffer,
        prominenceBuffer,
        refinedIndexBuffer,
        refinedValueBuffer,
        maskBuffer,
        idsBuffer,
        countBuffer,
        overflowBuffer,
        convergedBuffer
      ]) {
        buffer?.destroy();
      }
    }
  };
}

async function runOnce(
  device: Device,
  values: Float32Array,
  validity: Uint32Array | undefined,
  offsets: Uint32Array,
  options: PeakOptions
): Promise<PeakRun> {
  const runner = await createPeakRunner(device, values, validity, offsets, options);
  const run = await runner.run();
  runner.destroy();
  return run;
}

function getRows(oracle: ProfilePeaksOracleResult): number[] {
  return oracle.peaks.map(peak => peak.row);
}

/** Asserts the GPU run equals the oracle: exact mask, ids and prominence, tolerant refinement. */
function expectMatchesOracle(
  run: PeakRun,
  oracle: ProfilePeaksOracleResult,
  offsets: Uint32Array,
  wrap: boolean = false,
  tolerance: number = 2e-3
): void {
  expect(run.converged).toBe(1);
  expect(run.mask).toEqual(Array.from(oracle.mask));
  expect(run.compactIds).toEqual(getRows(oracle));
  // Prominence is a single correctly rounded f32 subtraction: bit exact, NaN where not a peak.
  expect(run.prominence).toEqual(Array.from(oracle.prominence));
  const profileOfRow = (row: number) =>
    offsets.findIndex((end, index) => index > 0 && row < end) - 1;
  for (let row = 0; row < run.mask.length; row++) {
    if (!oracle.mask[row]) {
      expect(run.refinedIndex[row]).toBeNaN();
      expect(run.refinedValue[row]).toBeNaN();
      continue;
    }
    const profile = profileOfRow(row);
    const count = offsets[profile + 1] - offsets[profile];
    let indexError = Math.abs(run.refinedIndex[row] - oracle.refinedIndex[row]);
    if (wrap) {
      indexError = Math.min(indexError, count - indexError);
    }
    expect(indexError).toBeLessThan(tolerance);
    const valueScale = Math.max(1, Math.abs(oracle.refinedValue[row]));
    expect(Math.abs(run.refinedValue[row] - oracle.refinedValue[row])).toBeLessThan(
      tolerance * valueScale
    );
  }
}

async function expectOracleRun(
  device: Device,
  values: Float32Array,
  validity: Uint32Array | undefined,
  offsets: Uint32Array,
  options: PeakOptions
): Promise<{run: PeakRun; oracle: ProfilePeaksOracleResult}> {
  const oracle = findProfilePeaksOracle(values, validity, offsets, options);
  const run = await runOnce(device, values, validity, offsets, options);
  expectMatchesOracle(run, oracle, offsets, options.wrap ?? false);
  return {run, oracle};
}

function createRandomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

it('GPUProfilePeaks finds Gaussian summits at sub-sample accuracy', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createGaussianBumps(200, [
    [50.3, 10, 6],
    [120.7, 6, 6]
  ]);
  const offsets = Uint32Array.from([0, 200]);
  const {run, oracle} = await expectOracleRun(device, values, undefined, offsets, {
    window: 30,
    minProminence: 2
  });
  expect(oracle.peaks).toHaveLength(2);
  expect(run.compactIds).toHaveLength(2);
  expect(run.refinedIndex[run.compactIds[0]]).toBeCloseTo(50.3, 1);
  expect(run.refinedIndex[run.compactIds[1]]).toBeCloseTo(120.7, 1);
  expect(run.refinedValue[run.compactIds[0]]).toBeCloseTo(10, 0);
  expect(run.prominence[run.compactIds[0]]).toBeGreaterThan(9);
  device.destroy();
});

it('GPUProfilePeaks rejects peaks below the prominence threshold', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createGaussianBumps(100, [[50, 1, 4]]);
  const offsets = Uint32Array.from([0, 100]);
  const high = await runOnce(device, values, undefined, offsets, {window: 20, minProminence: 2});
  expect(high.compactIds).toEqual([]);
  expect(high.converged).toBe(1);
  const low = await expectOracleRun(device, values, undefined, offsets, {
    window: 20,
    minProminence: 0.5
  });
  expect(low.run.compactIds).toEqual([50]);
  device.destroy();
});

it('GPUProfilePeaks is invariant to a constant offset', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const offsets = Uint32Array.from([0, 120]);
  const base = createGaussianBumps(120, [[60.2, 7, 5]]);
  const shifted = base.map(value => value + 1024);
  const options = {window: 25, minProminence: 1};
  const first = await expectOracleRun(device, base, undefined, offsets, options);
  const second = await expectOracleRun(device, shifted, undefined, offsets, options);
  expect(first.run.compactIds).toEqual([60]);
  expect(second.run.compactIds).toEqual(first.run.compactIds);
  // Exactly representable shift: prominence unchanged within f32 rounding of the shifted values.
  expect(Math.abs(second.run.prominence[60] - first.run.prominence[60])).toBeLessThan(1e-3);
  device.destroy();
});

it('GPUProfilePeaks takes the first sample of a plateau', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = new Float32Array(60);
  for (let index = 25; index <= 30; index++) {
    values[index] = 5;
  }
  const offsets = Uint32Array.from([0, 60]);
  const {run, oracle} = await expectOracleRun(device, values, undefined, offsets, {
    window: 15,
    minProminence: 1
  });
  expect(oracle.peaks).toHaveLength(1);
  expect(run.compactIds).toEqual([25]);
  expect(run.refinedIndex[25]).toBeGreaterThanOrEqual(24.5);
  expect(run.refinedIndex[25]).toBeLessThanOrEqual(26);
  // Flat input has no peaks.
  const flat = await runOnce(device, new Float32Array(50), undefined, Uint32Array.from([0, 50]), {
    window: 10,
    minProminence: 0.1
  });
  expect(flat.compactIds).toEqual([]);
  expect(flat.converged).toBe(1);
  device.destroy();
});

it('GPUProfilePeaks never lets a gap value bleed: sentinel with validity 0 and NaN gaps', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const clean = createGaussianBumps(160, [
    [40, 10, 4],
    [100, 8, 4]
  ]);
  const offsets = Uint32Array.from([0, 160]);
  const options = {window: 20, minProminence: 1};
  // A huge sentinel inside the window (and as a would-be neighbour) with validity zero.
  const sentinel = Float32Array.from(clean);
  const validity = new Uint32Array(160).fill(1);
  for (const index of [30, 31, 47, 48, 49, 108]) {
    sentinel[index] = 1e30;
    validity[index] = 0;
  }
  // The same gaps expressed as NaN with all-valid validity must give identical results.
  const nan = Float32Array.from(sentinel);
  for (let index = 0; index < 160; index++) {
    if (!validity[index]) {
      nan[index] = Number.NaN;
    }
  }
  const viaSentinel = await expectOracleRun(device, sentinel, validity, offsets, options);
  const viaNaN = await expectOracleRun(device, nan, undefined, offsets, options);
  expect(viaSentinel.run.mask).toEqual(viaNaN.run.mask);
  expect(viaSentinel.run.prominence).toEqual(viaNaN.run.prominence);
  // Non-trivial: the first summit's right side is cut at 47 (7 valid samples) but still passes
  // minSide 4; no output carries the sentinel.
  expect(viaSentinel.oracle.peaks.length).toBeGreaterThan(0);
  for (const value of [...viaSentinel.run.prominence, ...viaSentinel.run.refinedValue]) {
    if (!Number.isNaN(value)) {
      expect(value).toBeLessThan(1e3);
    }
  }
  // A gap directly beside a candidate disables the parabola but not the candidate.
  const adjacent = Float32Array.from(clean);
  const adjacentValidity = new Uint32Array(160).fill(1);
  adjacent[41] = -32768;
  adjacentValidity[41] = 0;
  // With the default minSide the candidate is rejected (its right side has no valid sample).
  const rejected = await expectOracleRun(device, adjacent, adjacentValidity, offsets, options);
  expect(rejected.run.mask[40]).toBe(0);
  const beside = await expectOracleRun(device, adjacent, adjacentValidity, offsets, {
    window: 20,
    minSide: 0,
    // An unwalked side leaves lo = v, so prominence is 0: only a zero threshold admits it.
    minProminence: 0
  });
  expect(beside.run.mask[40]).toBe(1);
  expect(beside.run.refinedIndex[40]).toBe(40);
  device.destroy();
});

it('GPUProfilePeaks rejects peaks cut by a profile end or a gap (minSide)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const edge = createGaussianBumps(100, [[2, 10, 3]]);
  const offsets = Uint32Array.from([0, 100]);
  const atEdge = await expectOracleRun(device, edge, undefined, offsets, {
    window: 20,
    minProminence: 1
  });
  expect(atEdge.run.compactIds).toEqual([]);
  // With minSide 1 the same summit is accepted: the rejection really is minSide.
  const relaxed = await expectOracleRun(device, edge, undefined, offsets, {
    window: 20,
    minProminence: 1,
    minSide: 1
  });
  expect(relaxed.run.compactIds).toEqual([2]);
  const gapped = createGaussianBumps(100, [[50, 10, 4]]);
  for (let index = 53; index < 100; index++) {
    gapped[index] = Number.NaN;
  }
  const cut = await expectOracleRun(device, gapped, undefined, offsets, {
    window: 20,
    minProminence: 1
  });
  expect(cut.run.compactIds).toEqual([]);
  // Profiles do not see each other: a summit at the start of the second profile is cut at its
  // own start even though the first profile's samples sit right before it.
  const {values, offsets: twoOffsets} = concatenateProfiles([
    createGaussianBumps(50, [[25, 5, 4]]),
    createGaussianBumps(50, [[1, 9, 3]])
  ]);
  const twoProfiles = await expectOracleRun(device, values, undefined, twoOffsets, {
    window: 20,
    minProminence: 1
  });
  expect(twoProfiles.run.compactIds).toEqual([25]);
  device.destroy();
});

it('GPUProfilePeaks resolves suppression chains exactly like greedy (keeps A and C, suppresses B)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A > B > C, A-B and B-C within nms (8 <= 10), A-C beyond it (16 > 10).
  const values = createGaussianBumps(80, [
    [30, 10, 2],
    [38, 8, 2],
    [46, 6, 2]
  ]);
  const offsets = Uint32Array.from([0, 80]);
  const options = {window: 12, minProminence: 1, nms: 10};
  const {run, oracle} = await expectOracleRun(device, values, undefined, offsets, options);
  expect(oracle.peaks.map(peak => peak.local)).toEqual([30, 46]);
  expect(run.compactIds).toEqual([30, 46]);
  // Without suppression all three are reported.
  const unsuppressed = await expectOracleRun(device, values, undefined, offsets, {
    ...options,
    nms: 3
  });
  expect(unsuppressed.run.compactIds).toEqual([30, 38, 46]);
  device.destroy();
});

it('GPUProfilePeaks reports converged 0 and nothing undecided when nmsRounds is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createGaussianBumps(80, [
    [30, 10, 2],
    [38, 8, 2],
    [46, 6, 2]
  ]);
  const offsets = Uint32Array.from([0, 80]);
  const options = {window: 12, minProminence: 1, nms: 10};
  const none = await runOnce(device, values, undefined, offsets, {...options, nmsRounds: 0});
  expect(none.converged).toBe(0);
  expect(none.compactIds).toEqual([]);
  expect(none.mask.every(word => word === 0)).toBe(true);
  expect(none.prominence.every(value => Number.isNaN(value))).toBe(true);
  // One round may or may not finish (in-place updates), but whatever is reported is a subset of
  // the exact answer and `converged` tells which case it is.
  const one = await runOnce(device, values, undefined, offsets, {...options, nmsRounds: 1});
  const exact = [30, 46];
  for (const row of one.compactIds) {
    expect(exact).toContain(row);
  }
  if (one.converged === 1) {
    expect(one.compactIds).toEqual(exact);
  }
  const full = await runOnce(device, values, undefined, offsets, options);
  expect(full.converged).toBe(1);
  expect(full.compactIds).toEqual(exact);
  device.destroy();
});

it('GPUProfilePeaks handles many CSR profiles including empty, single and two-sample ones', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const gappy = createGaussianBumps(70, [
    [20, 6, 3],
    [50, 9, 3]
  ]);
  gappy[35] = Number.NaN;
  const {values, offsets} = concatenateProfiles([
    [],
    createGaussianBumps(80, [[40.4, 10, 5]]),
    [],
    [5],
    [1, 2],
    gappy,
    createGaussianBumps(5, [[2, 3, 1]]),
    [],
    createGaussianBumps(60, [
      [15, 4, 3],
      [45, 7, 3]
    ]),
    []
  ]);
  expect(Array.from(offsets).slice(0, 5)).toEqual([0, 0, 80, 80, 81]);
  const options = {window: 15, minProminence: 1};
  const {run, oracle} = await expectOracleRun(device, values, undefined, offsets, options);
  expect(oracle.peaks.length).toBeGreaterThanOrEqual(4);
  expect(new Set(oracle.peaks.map(peak => peak.profile)).size).toBeGreaterThanOrEqual(3);
  expect(run.compactIds.length).toBe(oracle.peaks.length);
  device.destroy();
});

it('GPUProfilePeaks wrap mode finds a peak straddling index 0', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const count = 120;
  const values = createCircularGaussianBumps(count, [
    [119.6, 10, 5],
    [60.3, 6, 5]
  ]);
  const offsets = Uint32Array.from([0, count]);
  const options = {window: 30, minProminence: 2};
  const open = await runOnce(device, values, undefined, offsets, options);
  const seamIsReported = (rows: number[]) => rows.some(row => row <= 1 || row >= count - 2);
  expect(seamIsReported(open.compactIds)).toBe(false);
  const circular = await expectOracleRun(device, values, undefined, offsets, {
    ...options,
    wrap: true
  });
  expect(circular.run.compactIds).toHaveLength(2);
  const seamRow = circular.run.compactIds.find(row => row <= 1 || row >= count - 2) as number;
  expect(seamRow).toBeDefined();
  const refined = circular.run.refinedIndex[seamRow];
  const circularError = Math.min(Math.abs(refined - 119.6), count - Math.abs(refined - 119.6));
  expect(circularError).toBeLessThan(0.1);
  expect(refined).toBeGreaterThanOrEqual(0);
  expect(refined).toBeLessThan(count);
  // Circular suppression: two summits across the seam within nms collapse to the taller one.
  const twins = createCircularGaussianBumps(count, [
    [114, 10, 3],
    [6, 8, 3]
  ]);
  const collapsed = await expectOracleRun(device, twins, undefined, offsets, {
    window: 20,
    minProminence: 1,
    nms: 15,
    wrap: true
  });
  expect(collapsed.run.compactIds).toHaveLength(1);
  const separate = await expectOracleRun(device, twins, undefined, offsets, {
    window: 20,
    minProminence: 1,
    nms: 5,
    wrap: true
  });
  expect(separate.run.compactIds).toHaveLength(2);
  device.destroy();
});

it('GPUProfilePeaks applies a new minProminence without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createGaussianBumps(200, [
    [30, 10, 4],
    [90, 5, 4],
    [150, 2.5, 4]
  ]);
  const offsets = Uint32Array.from([0, 200]);
  const options = {window: 25, minProminence: 1};
  const runner = await createPeakRunner(device, values, undefined, offsets, options);
  const counts: number[] = [];
  for (const minProminence of [1, 4, 8, 20, 1]) {
    const run = await runner.run(minProminence);
    const oracle = findProfilePeaksOracle(values, undefined, offsets, {...options, minProminence});
    expectMatchesOracle(run, oracle, offsets);
    counts.push(run.compactIds.length);
  }
  expect(counts).toEqual([3, 2, 1, 0, 3]);
  runner.destroy();
  device.destroy();
});

for (const wrap of [false, true]) {
  it(`GPUProfilePeaks matches the oracle on random gappy profiles (wrap ${wrap})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    let totalPeaks = 0;
    for (const seed of [11, 12, 13, 14]) {
      const random = createRandomGenerator(seed);
      const profiles: number[][] = [];
      const validityRows: number[][] = [];
      for (let profile = 0; profile < 6; profile++) {
        const count = [0, 1, 2, 40, 150, 300][profile] + Math.floor(random() * 3);
        const phases = [random() * 6, random() * 6, random() * 6];
        const row: number[] = [];
        const rowValidity: number[] = [];
        for (let index = 0; index < count; index++) {
          const smooth =
            4 * Math.sin(index / 7 + phases[0]) +
            3 * Math.sin(index / 3.1 + phases[1]) +
            2 * Math.sin(index / 13 + phases[2]);
          row.push(smooth + random());
          rowValidity.push(random() < 0.03 ? 0 : 1);
        }
        profiles.push(row);
        validityRows.push(rowValidity);
      }
      const {values, offsets} = concatenateProfiles(profiles);
      const validity = Uint32Array.from(validityRows.flat());
      for (let index = 0; index < values.length; index++) {
        if (!validity[index]) {
          values[index] = index % 2 ? 1e30 : -32768;
        }
      }
      const {oracle} = await expectOracleRun(device, values, validity, offsets, {
        window: 12,
        minProminence: 0.75,
        minSide: 3,
        nms: 4,
        wrap
      });
      totalPeaks += oracle.peaks.length;
    }
    expect(totalPeaks).toBeGreaterThan(20);
    device.destroy();
  });
}
