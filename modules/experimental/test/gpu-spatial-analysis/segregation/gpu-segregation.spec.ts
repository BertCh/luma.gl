// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSegregationLayout,
  GPUSegregation
} from '../../../src/gpu-spatial-analysis/segregation/index';
import {
  createSeededRandom,
  expectClose,
  GraphRig
} from '../areal-interpolation/areal-interpolation-harness';
import {
  computeSegregationOracle,
  computeTextbookDissimilarity,
  createGridBandWeights
} from './segregation-oracle';

const COLUMNS = 9;
const ROWS = 7;
const UNITS = COLUMNS * ROWS;
const GROUPS = 3;

/** Clustered populations: group 0 on the left, group 1 on the right, group 2 mixed. */
function createCounts(seed: number): Float32Array {
  const random = createSeededRandom(seed);
  const counts = new Float32Array(UNITS * GROUPS);
  for (let unit = 0; unit < UNITS; unit++) {
    const column = unit % COLUMNS;
    const lean = column / (COLUMNS - 1);
    counts[unit * GROUPS] = Math.floor(10 + 80 * (1 - lean) * random());
    counts[unit * GROUPS + 1] = Math.floor(10 + 80 * lean * random());
    counts[unit * GROUPS + 2] = Math.floor(5 + 30 * random());
  }
  // An empty unit exercises zero totals.
  counts.fill(0, 4 * GROUPS, 5 * GROUPS);
  return counts;
}

async function runSegregation(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  options: {radii: (number | null)[]; selfWeight?: number; atkinsonB?: number; seed: number}
) {
  const counts = createCounts(options.seed);
  const layout = getGPUSegregationLayout(GROUPS);
  const scaleCount = options.radii.length;
  const rig = new GraphRig(device);
  const oracleWeights = options.radii.map(radius =>
    radius === null ? null : createGridBandWeights(COLUMNS, ROWS, radius)
  );
  const scales = oracleWeights.map(csr => {
    if (!csr) return null;
    const capacity = csr.neighbors.length;
    return {
      offsets: rig.input(Uint32Array.from(csr.offsets), 'uint32'),
      neighbors: rig.input(Uint32Array.from(csr.neighbors), 'uint32', capacity),
      weights: rig.input(Float32Array.from(csr.weights), 'float32', capacity)
    };
  });
  const indices = rig.output('float32', scaleCount * layout.stride);
  const environment = rig.output('float32', scaleCount * UNITS * GROUPS);
  const entropy = rig.output('float32', scaleCount * UNITS);
  const dissimilarity = rig.output('float32', scaleCount * UNITS * GROUPS);
  const theil = rig.output('float32', scaleCount * UNITS);
  rig.run(
    new GPUSegregation({
      unitCount: UNITS,
      groupCount: GROUPS,
      groupCounts: rig.input(counts, 'float32'),
      scales,
      selfWeight: options.selfWeight,
      atkinsonB: options.atkinsonB,
      indices: indices.view,
      local: {
        environment: environment.view,
        entropy: entropy.view,
        dissimilarity: dissimilarity.view,
        theil: theil.view
      }
    })
  );
  const actual = {
    indices: await indices.readFloat32(),
    environment: await environment.readFloat32(),
    entropy: await entropy.readFloat32(),
    dissimilarity: await dissimilarity.readFloat32(),
    theil: await theil.readFloat32()
  };
  rig.destroy();
  const expected = oracleWeights.map(weights =>
    computeSegregationOracle({
      unitCount: UNITS,
      groupCount: GROUPS,
      counts,
      weights,
      selfWeight: options.selfWeight,
      atkinsonB: options.atkinsonB
    })
  );
  return {actual, expected, counts, layout};
}

it('GPUSegregation matches the oracle for the aspatial indices and a multiscale profile', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {actual, expected, layout} = await runSegregation(device, {
    radii: [null, 1.5, 3, 6],
    seed: 5
  });
  const stride = layout.stride;
  for (let scale = 0; scale < expected.length; scale++) {
    expectClose(
      actual.indices.slice(scale * stride, (scale + 1) * stride),
      expected[scale].indices,
      `scale ${scale} indices`,
      2e-4,
      1e-5
    );
    expectClose(
      actual.environment.slice(scale * UNITS * GROUPS, (scale + 1) * UNITS * GROUPS),
      expected[scale].environment,
      `scale ${scale} environment`
    );
    expectClose(
      actual.entropy.slice(scale * UNITS, (scale + 1) * UNITS),
      expected[scale].entropy,
      `scale ${scale} local entropy`,
      1e-4,
      1e-5
    );
    expectClose(
      actual.theil.slice(scale * UNITS, (scale + 1) * UNITS),
      expected[scale].theil,
      `scale ${scale} local theil`,
      1e-3,
      1e-6
    );
    expectClose(
      actual.dissimilarity.slice(scale * UNITS * GROUPS, (scale + 1) * UNITS * GROUPS),
      expected[scale].dissimilarity,
      `scale ${scale} local dissimilarity`,
      1e-3,
      1e-6
    );
  }
  // Nonzero guards: a failed WGSL compile gives silent zeros.
  expect(actual.indices[layout.entropy]).toBeGreaterThan(0.01);
  expect(actual.indices[layout.multiGroupDissimilarity]).toBeGreaterThan(0.01);
  expect(actual.indices[layout.dissimilarity]).toBeGreaterThan(0.01);
  expect(actual.indices[layout.isolation]).toBeGreaterThan(0.01);
  expect(actual.indices[layout.atkinson]).toBeGreaterThan(0.001);
  // Spatial smoothing blends neighbors, so segregation falls as the bandwidth grows.
  const multiD = (scale: number) => actual.indices[scale * stride + layout.multiGroupDissimilarity];
  expect(multiD(1)).toBeLessThan(multiD(0));
  expect(multiD(3)).toBeLessThan(multiD(1));
});

it('GPUSegregation aspatial D equals the textbook dissimilarity and exposures sum to one', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {actual, counts, layout} = await runSegregation(device, {radii: [null], seed: 8});
  for (let group = 0; group < GROUPS; group++) {
    expect(actual.indices[layout.dissimilarity + group]).toBeCloseTo(
      computeTextbookDissimilarity(UNITS, GROUPS, counts, group),
      4
    );
    let exposure = 0;
    for (let other = 0; other < GROUPS; other++) {
      exposure += actual.indices[layout.interaction + group * GROUPS + other];
    }
    expect(exposure).toBeCloseTo(1, 4);
    expect(actual.indices[layout.isolation + group]).toBeCloseTo(
      actual.indices[layout.interaction + group * GROUPS + group],
      6
    );
  }
});

it('GPUSegregation local indices add up to the global ones', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {actual, layout} = await runSegregation(device, {
    radii: [2.5],
    selfWeight: 0.5,
    atkinsonB: 0.3,
    seed: 11
  });
  let theil = 0;
  for (let unit = 0; unit < UNITS; unit++) theil += actual.theil[unit];
  expect(theil).toBeCloseTo(actual.indices[layout.entropy], 3);
  for (let group = 0; group < GROUPS; group++) {
    let dissimilarity = 0;
    for (let unit = 0; unit < UNITS; unit++)
      dissimilarity += actual.dissimilarity[unit * GROUPS + group];
    expect(dissimilarity).toBeCloseTo(actual.indices[layout.dissimilarity + group], 3);
  }
  // Environment compositions of populated units sum to one.
  for (let unit = 0; unit < UNITS; unit++) {
    let sum = 0;
    for (let group = 0; group < GROUPS; group++) sum += actual.environment[unit * GROUPS + group];
    expect(sum).toBeCloseTo(1, 4);
  }
});

it('GPUSegregation reports zero segregation for identical unit compositions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const counts = new Float32Array(UNITS * 2);
  for (let unit = 0; unit < UNITS; unit++) {
    const size = 10 + (unit % 5);
    counts[unit * 2] = size * 3;
    counts[unit * 2 + 1] = size * 7;
  }
  const layout = getGPUSegregationLayout(2);
  const rig = new GraphRig(device);
  const indices = rig.output('float32', layout.stride);
  rig.run(
    new GPUSegregation({
      unitCount: UNITS,
      groupCount: 2,
      groupCounts: rig.input(counts, 'float32'),
      indices: indices.view
    })
  );
  const result = await indices.readFloat32();
  rig.destroy();
  expect(result[layout.diversity]).toBeGreaterThan(0.5);
  expect(result[layout.entropy]).toBeCloseTo(0, 4);
  expect(result[layout.multiGroupDissimilarity]).toBeCloseTo(0, 4);
  expect(result[layout.dissimilarity]).toBeCloseTo(0, 4);
  expect(result[layout.atkinson]).toBeCloseTo(0, 3);
  // Isolation of a group equals its population share when unit compositions are identical.
  expect(result[layout.isolation]).toBeCloseTo(0.3, 3);
  expect(result[layout.isolation + 1]).toBeCloseTo(0.7, 3);
});
