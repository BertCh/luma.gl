// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {GPURasterProfile} from '../../../src/gpu-raster/raster-sampling';
import {
  getSegmentedScanBlockCount,
  getSegmentedScanTreeHeight
} from '../../../src/gpu-raster/raster-sampling/gpu-raster-profile-scan';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../raster-algebra/raster-algebra-test-utils';
import {packProfileCase, profileRasterOnCPU} from './raster-sampling-oracle';
import {
  combineDistanceStates,
  combineProfileStates,
  emulateSegmentedScan,
  getIdentityDistanceState,
  getIdentityProfileState,
  getProfileItemState
} from './raster-profile-scan-emulation';

const f = Math.fround;
const UNIT_ROUNDOFF = 2 ** -24;

/** Relative error bound of a non-negative sum formed by a tree of `height` float32 additions plus one rounded term. */
function getTreeRelativeBound(height: number): number {
  const factor = (height + 1) * UNIT_ROUNDOFF;
  return factor / (1 - factor);
}

type Kernel = {id: string; source: string};

function getProfileKernels(
  options: {vertexCount?: number; capacity?: number; allOutputs?: boolean} = {}
): {ids: string[]; kernels: Kernel[]} {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `profile-wgsl-${serial++}`, format, length);
  const capacity = options.capacity ?? 64;
  const lineCount = 3;
  const output: Record<string, unknown> = {
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
  if (options.allOutputs !== false) {
    Object.assign(output, {
      samplePositions: view('float32x2', capacity),
      sampleDistances: view('float32', capacity),
      sampleValues: view('float32', capacity),
      sampleLineIds: view('uint32', capacity),
      sampleCumulativeGain: view('float32', capacity),
      sampleCumulativeLoss: view('float32', capacity),
      lineSampleOffsets: view('uint32', lineCount + 1),
      lineLengths: view('float32', lineCount),
      lineGains: view('float32', lineCount),
      lineLosses: view('float32', lineCount),
      lineMinimums: view('float32', lineCount),
      lineMaximums: view('float32', lineCount)
    });
  }
  const nodes = new GPURasterProfile({
    id: 'p',
    grid: makeGPURasterMetadata({width: 8, height: 8}),
    values: view('float32', 64),
    positions: view('float32x2', options.vertexCount ?? 9),
    lineOffsets: view('uint32', lineCount + 1),
    sampleCapacity: capacity,
    parameters: view('float32', 12),
    output: output as ConstructorParameters<typeof GPURasterProfile>[0]['output']
  }).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  device.destroy();
  return {ids: nodes.map(node => node.id), kernels};
}

it('GPURasterProfile builds segmented scans with the expected node order', () => {
  const {ids} = getProfileKernels();
  expect(ids.filter(id => !id.startsWith('p-scan-'))).toEqual([
    'p-truncated-fill',
    'p-measure',
    'p-measure-carries',
    'p-measure-distances',
    'p-sample-counts',
    'p-locate',
    'p-sample',
    'p-cumulative-blocks',
    'p-cumulative-carries',
    'p-cumulative',
    'p-summaries',
    'p-finish-lines',
    'p-finish-counts'
  ]);
  // The count scan sits between the sample counts and the locate kernel.
  expect(ids.indexOf('p-sample-counts')).toBeLessThan(
    ids.findIndex(id => id.startsWith('p-scan-'))
  );
  expect(ids.findIndex(id => id.startsWith('p-scan-'))).toBeLessThan(ids.indexOf('p-locate'));
});

it('GPURasterProfile kernels have no per-line serial loop over samples or vertices', () => {
  for (const options of [{}, {allOutputs: false}, {vertexCount: 100000, capacity: 1000000}]) {
    const {kernels} = getProfileKernels(options);
    const own = kernels.filter(({id}) => id.startsWith('p-'));
    expect(own.length).toBeGreaterThan(options.allOutputs === false ? 6 : 8);
    for (const {id, source} of own) {
      expect(() => new WgslReflect(source), id).not.toThrow();
      // Storage buffers: the core limit is 8 per stage.
      expect(
        new Set(source.match(/@binding\((\d+)\)/g)).size,
        `${id} bindings`
      ).toBeLessThanOrEqual(8);
      expect(source, `${id} serial sample loop`).not.toMatch(/for \(var (sample|vertex)\b/);
      if (id !== 'p-summaries') {
        expect(source, `${id} per-line sample range`).not.toMatch(/endSample/);
      }
      // Atomics only in the per-line count kernel (one per line), never per sample.
      if (id !== 'p-sample-counts') {
        expect(source, `${id} atomics`).not.toContain('atomicMax');
      }
    }
    // The three kernels of each scan use workgroup scans with barriers and a 256-lane group.
    for (const id of [
      'p-measure',
      'p-measure-carries',
      'p-measure-distances',
      'p-cumulative-blocks',
      'p-cumulative-carries',
      'p-cumulative'
    ].filter(id => !(options.allOutputs === false && id.startsWith('p-cumulative')))) {
      const kernel = own.find(candidate => candidate.id === id)!;
      expect(kernel.source, id).toContain('workgroupBarrier()');
      expect(kernel.source, id).toContain('@workgroup_size(256)');
      expect(kernel.source, id).toContain('var<workgroup> scanScratch: array<ScanState, 256>');
    }
    // Loops over blocks exist only in the one-workgroup carry kernels.
    for (const {id, source} of own) {
      const hasBlockLoop = /for \(var chunkBlock/.test(source);
      expect(hasBlockLoop, id).toBe(id.endsWith('-carries'));
    }
  }
});

it('GPURasterProfile scan state fits workgroup memory and block counts follow capacity', () => {
  // ScanState has 8 four-byte fields: 256 * 32 B = 8192 B of the 16384 B core limit.
  expect(256 * 8 * 4).toBeLessThanOrEqual(16384);
  expect(getSegmentedScanBlockCount(0)).toBe(1);
  expect(getSegmentedScanBlockCount(256)).toBe(1);
  expect(getSegmentedScanBlockCount(257)).toBe(2);
  expect(getSegmentedScanTreeHeight(2e5)).toBe(23);
  expect(getSegmentedScanTreeHeight(256 * 256)).toBe(17);
  expect(getSegmentedScanTreeHeight(256 * 256 + 1)).toBe(19);
  const {kernels} = getProfileKernels({vertexCount: 100000, capacity: 1000000});
  const carries = kernels.find(({id}) => id === 'p-cumulative-carries')!;
  expect(carries.source).toContain(`const BLOCK_COUNT: u32 = ${Math.ceil(1000000 / 256)}u;`);
});

/** Serial float32 walk of the previous kernel: the oracle for the scan emulation. */
function serialProfile(values: Float32Array, starts: readonly number[]) {
  const cumulativeGain = new Float32Array(values.length);
  const cumulativeLoss = new Float32Array(values.length);
  const lineGains: number[] = [];
  const lineLosses: number[] = [];
  const lineMinimums: number[] = [];
  const lineMaximums: number[] = [];
  for (let line = 0; line + 1 < starts.length; line++) {
    let gain = 0;
    let loss = 0;
    let minimum = NaN;
    let maximum = NaN;
    let previous = 0;
    let hasPrevious = false;
    for (let sample = starts[line]; sample < starts[line + 1]; sample++) {
      const value = values[sample];
      if (Number.isFinite(value)) {
        if (hasPrevious) {
          const delta = f(value - previous);
          if (delta > 0) gain = f(gain + delta);
          else if (delta < 0) loss = f(loss - delta);
          minimum = value < minimum ? value : minimum;
          maximum = value > maximum ? value : maximum;
        } else {
          minimum = maximum = value;
        }
        previous = value;
        hasPrevious = true;
      }
      cumulativeGain[sample] = gain;
      cumulativeLoss[sample] = loss;
    }
    lineGains.push(gain);
    lineLosses.push(loss);
    lineMinimums.push(minimum);
    lineMaximums.push(maximum);
  }
  return {cumulativeGain, cumulativeLoss, lineGains, lineLosses, lineMinimums, lineMaximums};
}

/** Exact float64 prefix sums of the consecutive-finite deltas of the float32 values. */
function exactProfile(values: Float32Array, starts: readonly number[]) {
  const cumulativeGain = new Float64Array(values.length);
  const cumulativeLoss = new Float64Array(values.length);
  for (let line = 0; line + 1 < starts.length; line++) {
    let gain = 0;
    let loss = 0;
    let previous = 0;
    let hasPrevious = false;
    for (let sample = starts[line]; sample < starts[line + 1]; sample++) {
      const value = values[sample];
      if (Number.isFinite(value)) {
        if (hasPrevious) {
          const delta = value - previous;
          if (delta > 0) gain += delta;
          else if (delta < 0) loss -= delta;
        }
        previous = value;
        hasPrevious = true;
      }
      cumulativeGain[sample] = gain;
      cumulativeLoss[sample] = loss;
    }
  }
  return {cumulativeGain, cumulativeLoss};
}

function emulateProfile(values: Float32Array, starts: readonly number[]) {
  const startSet = new Set(starts.slice(0, -1));
  const items = Array.from(values, (value, sample) =>
    getProfileItemState(value, startSet.has(sample))
  );
  return emulateSegmentedScan(items, getIdentityProfileState, combineProfileStates);
}

function createWalk(count: number, seed: number, step: number, nanRate: number): Float32Array {
  const random = createRandom(seed);
  const values = new Float32Array(count);
  let level = 100;
  for (let sample = 0; sample < count; sample++) {
    level += (random() - 0.5) * step;
    values[sample] = random() < nanRate ? NaN : level;
  }
  return values;
}

// Line starts straddle block boundaries (256) and include an empty line (repeated offset).
const LONG_STARTS = [0, 99999, 100003, 100003, 199000, 200000] as const;

it('profile scan emulation matches the serial walk on a long transect within the tree bound', () => {
  const values = createWalk(200000, 5, 3, 0.03);
  const serial = serialProfile(values, LONG_STARTS);
  const exact = exactProfile(values, LONG_STARTS);
  const {inclusive, blockCount} = emulateProfile(values, LONG_STARTS);
  expect(blockCount).toBe(782);
  const height = getSegmentedScanTreeHeight(values.length);
  // Gain and loss add at most two terms per combine level (carry plus junction delta, then the
  // right-hand total), so the addition tree is at most twice the combine tree.
  const treeBound = getTreeRelativeBound(2 * height);
  const serialBound = getTreeRelativeBound(values.length);
  let maximumTreeError = 0;
  let maximumSerialError = 0;
  for (let sample = 0; sample < values.length; sample++) {
    for (const [name, emulated, serialValue, exactValue] of [
      ['gain', inclusive[sample].gain, serial.cumulativeGain[sample], exact.cumulativeGain[sample]],
      ['loss', inclusive[sample].loss, serial.cumulativeLoss[sample], exact.cumulativeLoss[sample]]
    ] as const) {
      const treeError = Math.abs(emulated - exactValue);
      const serialError = Math.abs(serialValue - exactValue);
      expect(treeError, `${name}[${sample}]`).toBeLessThanOrEqual(treeBound * exactValue);
      expect(serialError, `serial ${name}[${sample}]`).toBeLessThanOrEqual(
        serialBound * exactValue
      );
      if (exactValue > 0) {
        maximumTreeError = Math.max(maximumTreeError, treeError / exactValue);
        maximumSerialError = Math.max(maximumSerialError, serialError / exactValue);
      }
    }
  }
  // The tree order is not worse than the serial order's own bound and far inside its derivation.
  expect(maximumTreeError).toBeLessThanOrEqual(treeBound);
  expect(treeBound).toBeLessThan(serialBound / 1000);
  // Line summaries come from the last sample of each line: totals and exact extrema.
  for (let line = 0; line + 1 < LONG_STARTS.length; line++) {
    if (LONG_STARTS[line + 1] === LONG_STARTS[line]) {
      continue;
    }
    const last = inclusive[LONG_STARTS[line + 1] - 1];
    const gainExact = exact.cumulativeGain[LONG_STARTS[line + 1] - 1];
    expect(Math.abs(last.gain - gainExact)).toBeLessThanOrEqual(treeBound * gainExact);
    expect(last.hasValue ? last.minimum : NaN).toBe(serial.lineMinimums[line]);
    expect(last.hasValue ? last.maximum : NaN).toBe(serial.lineMaximums[line]);
  }
});

it('profile scan emulation is bit-exact on integer data and keeps the missing-value semantics', () => {
  const random = createRandom(11);
  const count = 70000;
  const values = new Float32Array(count);
  for (let sample = 0; sample < count; sample++) {
    const roll = random();
    // Long NaN runs, isolated NaN, and Infinity are all skipped; sums stay exact below 2^24.
    values[sample] =
      roll < 0.05
        ? NaN
        : roll < 0.06
          ? Infinity
          : roll < 0.07
            ? -Infinity
            : Math.floor(random() * 10);
  }
  for (let sample = 20000; sample < 20700; sample++) values[sample] = NaN;
  for (const starts of [
    [0, 33333, 33333, 33334, 33600, 70000],
    [0, 256, 512, 513, 70000],
    [0, 70000]
  ]) {
    const serial = serialProfile(values, starts);
    const {inclusive} = emulateProfile(values, starts);
    for (let sample = 0; sample < count; sample++) {
      expect(inclusive[sample].gain, `gain[${sample}]`).toBe(serial.cumulativeGain[sample]);
      expect(inclusive[sample].loss, `loss[${sample}]`).toBe(serial.cumulativeLoss[sample]);
    }
    for (let line = 0; line + 1 < starts.length; line++) {
      if (starts[line + 1] === starts[line]) continue;
      const last = inclusive[starts[line + 1] - 1];
      expect(last.hasValue ? last.minimum : NaN).toBe(serial.lineMinimums[line]);
      expect(last.hasValue ? last.maximum : NaN).toBe(serial.lineMaximums[line]);
    }
  }
  // A line whose samples are all non-finite reports NaN extrema and zero gain and loss.
  const allMissing = new Float32Array(1000).fill(NaN);
  const {inclusive} = emulateProfile(allMissing, [0, 1000]);
  expect(inclusive[999].hasValue).toBe(0);
  expect(inclusive[999].gain).toBe(0);
  expect(inclusive[999].loss).toBe(0);
});

it('vertex distance scan emulation matches the serial walk within the tree bound', () => {
  // One long polyline of 200000 vertices plus short lines, with an empty line in between.
  const random = createRandom(17);
  const vertexCount = 200000 + 10;
  const positions = new Float32Array(vertexCount * 2);
  let x = 0;
  let y = 0;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    x = f(x + (random() - 0.3) * 4);
    y = f(y + (random() - 0.5) * 4);
    positions[2 * vertex] = x;
    positions[2 * vertex + 1] = y;
  }
  const offsets = Uint32Array.from([0, 200000, 200000, 200004, vertexCount]);
  const raster = {width: 8, height: 8, values: new Float32Array(64)};
  // A huge spacing keeps the oracle's sample list short; only vertex distances matter here.
  const oracle = profileRasterOnCPU(
    raster,
    packProfileCase({width: 8, height: 8, extent: [0, 0, 8, 8], spacing: 1e9}),
    positions,
    offsets
  );
  const startSet = new Set([0, 200000, 200004]);
  const items = Array.from({length: vertexCount}, (_, vertex) => {
    if (startSet.has(vertex)) {
      return {flag: 1, sum: 0};
    }
    const deltaX = f(positions[2 * vertex] - positions[2 * vertex - 2]);
    const deltaY = f(positions[2 * vertex + 1] - positions[2 * vertex - 1]);
    return {flag: 0, sum: f(Math.sqrt(f(f(deltaX * deltaX) + f(deltaY * deltaY))))};
  });
  const {inclusive} = emulateSegmentedScan(items, getIdentityDistanceState, combineDistanceStates);
  const bound = getTreeRelativeBound(getSegmentedScanTreeHeight(vertexCount));
  const serialBound = getTreeRelativeBound(vertexCount);
  let exactSum = 0;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    exactSum = startSet.has(vertex) ? 0 : exactSum + items[vertex].sum;
    expect(Math.abs(inclusive[vertex].sum - exactSum), `distance[${vertex}]`).toBeLessThanOrEqual(
      bound * exactSum
    );
    expect(
      Math.abs(oracle.vertexDistances[vertex] - exactSum),
      `serial distance[${vertex}]`
    ).toBeLessThanOrEqual(serialBound * exactSum);
  }
  // Line lengths agree with the oracle to the serial rounding bound.
  expect(Math.abs(inclusive[199999].sum - oracle.lineLengths[0])).toBeLessThanOrEqual(
    (bound + serialBound) * oracle.lineLengths[0]
  );
});
