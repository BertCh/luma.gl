// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  type GPUGlobalSpatialStatisticsProps
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  GLOBAL_STATISTICS_REFERENCE as REFERENCE,
  JOIN_COUNT_REFERENCE
} from './global-spatial-statistics-reference';
import {
  computeGlobalSpatialStatisticsOracle,
  getPValue,
  createGridWeights,
  createSeededRandom,
  createWeights,
  getDenseMatrix,
  getGearyMoments,
  getMoranMoments,
  getPermutationMoments,
  type CPUSpatialWeights
} from './global-spatial-statistics-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGlobalSpatialStatisticsProps> = {}
): GPUGlobalSpatialStatisticsProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    weights: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 40),
      weights: view('float32', 40)
    },
    values: view('float32', 10),
    statistics: ['moran'],
    output: {summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)},
    ...overrides
  };
}

/** Mirrors the WGSL centered-form randomization variance of x'Wx (zero diagonal). */
function getCenteredQuadraticFormVariance(matrix: number[][], values: number[]): number {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const z = values.map(value => value - mean);
  const power = (k: number) => z.reduce((total, value) => total + value ** k, 0);
  let s0 = 0;
  let s1 = 0;
  const degrees = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      s0 += matrix[i][j];
      s1 += 0.5 * (matrix[i][j] + matrix[j][i]) ** 2;
      degrees[i] += matrix[i][j];
      degrees[j] += matrix[i][j];
    }
  }
  const s2 = degrees.reduce((total, d) => total + d * d, 0);
  const meanDegree = degrees.reduce((a, b) => a + b, 0) / n;
  const degreeDeviation = degrees.reduce((total, d) => total + (d - meanDegree) ** 2, 0);
  const kurtosis = (n * power(4)) / power(2) ** 2;
  const moranVariance = getMoranMoments(n, s0, s1, s2, kurtosis)[2];
  const scale = (s0 * power(2)) / n;
  return (
    (mean * mean * power(2) * degreeDeviation) / (n - 1) -
    (2 * mean * power(3) * degreeDeviation) / ((n - 1) * (n - 2)) +
    moranVariance * scale * scale
  );
}

function createRandomWeights(n: number, seed: number, density: number): CPUSpatialWeights {
  const random = createSeededRandom(seed);
  const lists = Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j): [number, number] => [j, random()]).filter(
      ([j]) => j !== i && random() < density
    )
  );
  return createWeights(lists);
}

function quadraticForm(matrix: number[][], left: number[], right: number[]): number {
  let total = 0;
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) {
      total += matrix[i][j] * left[i] * right[j];
    }
  }
  return total;
}

function expectClose(actual: number, expected: number, label: string): void {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    1e-9 * Math.max(1, Math.abs(expected))
  );
}

it('analytic randomization moments equal exact permutation moments on tiny asymmetric weights', () => {
  for (const seed of [1, 2, 3]) {
    const n = 7;
    const weights = createRandomWeights(n, seed, 0.5);
    const matrix = getDenseMatrix(weights);
    const random = createSeededRandom(seed + 100);
    const values = Array.from({length: n}, () => Math.round(random() * 9) + 1);
    const secondValues = Array.from({length: n}, () => random() * 4 - 2);
    const oracle = computeGlobalSpatialStatisticsOracle({
      weights,
      values: Float32Array.from(values),
      secondValues: Float32Array.from(secondValues),
      // The exact permutation moments below are over all n rows, islands included.
      islands: 'keep'
    });
    const label = `seed ${seed}`;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const moran = (permuted: number[]) => {
      const z = permuted.map(value => value - mean);
      return ((n / oracle.s0) * quadraticForm(matrix, z, z)) / z.reduce((t, v) => t + v * v, 0);
    };
    const [moranMean, moranVariance] = getPermutationMoments(values, moran);
    expectClose(oracle.moran.expected, moranMean, `${label} EI`);
    expectClose(oracle.moran.varianceRandomization, moranVariance, `${label} VI_rand`);

    const geary = (permuted: number[]) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          sum += matrix[i][j] * (permuted[i] - permuted[j]) ** 2;
        }
      }
      const squares = permuted.reduce((t, v) => t + (v - mean) ** 2, 0);
      return ((n - 1) * sum) / (2 * oracle.s0 * squares);
    };
    const [gearyMean, gearyVariance] = getPermutationMoments(values, geary);
    expectClose(gearyMean, 1, `${label} EC`);
    expectClose(oracle.geary.varianceRandomization, gearyVariance, `${label} VC_rand`);

    const sum = values.reduce((a, b) => a + b, 0);
    const sumSquares = values.reduce((a, b) => a + b * b, 0);
    const generalG = (permuted: number[]) =>
      quadraticForm(matrix, permuted, permuted) / (sum * sum - sumSquares);
    const [gMean, gVariance] = getPermutationMoments(values, generalG);
    expectClose(oracle.getisOrdG.expected, gMean, `${label} EG`);
    expectClose(oracle.getisOrdG.varianceRandomization, gVariance, `${label} VG (esda)`);
    expectClose(
      getCenteredQuadraticFormVariance(matrix, values) / (sum * sum - sumSquares) ** 2,
      gVariance,
      `${label} VG (centered)`
    );

    const meanY = secondValues.reduce((a, b) => a + b, 0) / n;
    const zx = values.map(value => value - mean);
    const zxSquares = zx.reduce((t, v) => t + v * v, 0);
    const bivariate = (permuted: number[]) => {
      const zy = permuted.map(value => value - meanY);
      return (
        ((n / oracle.s0) * quadraticForm(matrix, zx, zy)) /
        Math.sqrt(zxSquares * zy.reduce((t, v) => t + v * v, 0))
      );
    };
    const [bivariateMean, bivariateVariance] = getPermutationMoments(secondValues, bivariate);
    expectClose(bivariateMean, 0, `${label} E BV`);
    expectClose(oracle.bivariateMoran.varianceRandomization, bivariateVariance, `${label} V BV`);
  }
});

it('Cliff-Ord join-count moments equal exact permutation moments and the centered forms', () => {
  for (const seed of [4, 5]) {
    const n = 8;
    const weights = createRandomWeights(n, seed, 0.45);
    const matrix = getDenseMatrix(weights);
    const binary = matrix.map(row => row.map(w => (w > 0 ? 1 : 0)));
    const values = [1, 0, 1, 1, 0, 0, 1, 0];
    const oracle = computeGlobalSpatialStatisticsOracle({
      weights,
      values: Float32Array.from(values),
      islands: 'keep'
    });
    const blackBlack = (permuted: number[]) => 0.5 * quadraticForm(binary, permuted, permuted);
    const blackWhite = (permuted: number[]) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          sum += binary[i][j] * (permuted[i] - permuted[j]) ** 2;
        }
      }
      return 0.5 * sum;
    };
    const [bbMean, bbVariance] = getPermutationMoments(values, blackBlack);
    const [bwMean, bwVariance] = getPermutationMoments(values, blackWhite);
    const join = oracle.joinCount;
    expectClose(join.expectedBlackBlack, bbMean, `seed ${seed} E BB`);
    expectClose(join.varianceBlackBlack, bbVariance, `seed ${seed} V BB`);
    expectClose(join.expectedBlackWhite, bwMean, `seed ${seed} E BW`);
    expectClose(join.varianceBlackWhite, bwVariance, `seed ${seed} V BW`);
    // The GPU evaluates BB through the quadratic form and BW through Geary's moments.
    expectClose(
      0.25 * getCenteredQuadraticFormVariance(binary, values),
      bbVariance,
      `seed ${seed} V BB centered`
    );
    const p = 4 / n;
    const sumSquares = n * p * (1 - p);
    const kurtosis = (n * (n * p * (1 - p) * (p ** 3 + (1 - p) ** 3))) / sumSquares ** 2;
    let s0 = 0;
    let s1 = 0;
    const degrees = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        s0 += binary[i][j];
        s1 += 0.5 * (binary[i][j] + binary[j][i]) ** 2;
        degrees[i] += binary[i][j];
        degrees[j] += binary[i][j];
      }
    }
    const s2 = degrees.reduce((t, d) => t + d * d, 0);
    const scale = (s0 * sumSquares) / (n - 1);
    expectClose(
      getGearyMoments(n, s0, s1, s2, kurtosis)[1] * scale * scale,
      bwVariance,
      `seed ${seed} V BW Geary`
    );
  }
});

it('the oracle gives I = -1 and BB = 0 on a rook checkerboard', () => {
  const weights = createGridWeights(6, 6);
  const values = Float32Array.from(
    {length: 36},
    (_, index) => (Math.floor(index / 6) + (index % 6)) % 2
  );
  const oracle = computeGlobalSpatialStatisticsOracle({weights, values});
  expectClose(oracle.moran.statistic, -1, 'I');
  expect(oracle.joinCount.blackBlack).toBe(0);
  expect(oracle.joinCount.whiteWhite).toBe(0);
  expect(oracle.joinCount.blackWhite).toBe(oracle.joinCount.joins);
});

it('GPUGlobalSpatialStatistics validates its props and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'global-statistics-validation'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `bad-${serial++}`, format, length);
  expect(() => new GPUGlobalSpatialStatistics(createProps(graph))).not.toThrow();
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {values: view('float32', 9)}))
  ).toThrow(/values needs one row/);
  expect(() => new GPUGlobalSpatialStatistics(createProps(graph, {statistics: []}))).toThrow(
    /at least one/
  );
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {statistics: ['bivariateMoran']}))
  ).toThrow(/secondValues/);
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {statistics: ['nope' as 'moran']}))
  ).toThrow(/unknown/);
  expect(
    () =>
      new GPUGlobalSpatialStatistics(createProps(graph, {output: {summary: view('float32', 8)}}))
  ).toThrow(/output.summary/);
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {mask: view('uint32', 3)}))
  ).toThrow(/mask/);
  const values = view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length);
  expect(
    () =>
      new GPUGlobalSpatialStatistics(
        createProps(graph, {
          weights: {
            offsets: view('uint32', values.length + 1),
            neighbors: view('uint32', 4),
            weights: view('float32', 4)
          },
          values,
          output: {summary: values}
        })
      )
  ).toThrow(/must not share/);
  const contributor = new GPUGlobalSpatialStatistics({...createProps(graph), id: 'g'});
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.slice(0, 3)).toEqual(['g-counters-clear', 'g-include', 'g-islands']);
  expect(ids.at(-1)).toBe('g-finalize');
  expect(new Set(ids).size).toBe(ids.length);
  // The row-scale kernel exists only for row-standardized weights.
  expect(ids).not.toContain('g-row-scale');
  const standardized = new GPUGlobalSpatialStatistics({
    ...createProps(graph),
    id: 'h',
    weights: {
      ...createProps(graph).weights,
      information: {rowCount: 10, transform: 'row', symmetric: 'unknown'}
    }
  });
  expect(standardized.getCommandNodes(graph).map(node => node.id)).toContain('h-row-scale');
});

it('GPUGlobalSpatialStatistics resolves islands and alternative and rejects permutation tails', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'global-statistics-options'});
  expect(new GPUGlobalSpatialStatistics(createProps(graph)).islands).toBe('exclude');
  expect(new GPUGlobalSpatialStatistics(createProps(graph)).alternative).toBe('two-sided');
  expect(
    new GPUGlobalSpatialStatistics(createProps(graph, {islands: 'keep', alternative: 'lesser'}))
  ).toMatchObject({islands: 'keep', alternative: 'lesser'});
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {islands: 'drop' as 'keep'}))
  ).toThrow(/islands/);
  for (const alternative of ['directed', 'folded'] as const) {
    expect(() => new GPUGlobalSpatialStatistics(createProps(graph, {alternative}))).toThrow(
      /alternative must be/
    );
  }
  expect(
    new GPUGlobalSpatialStatistics(
      createProps(graph, {alternative: 'greater'})
    ).getInferenceMetadata()
  ).toMatchObject({alternative: 'greater', permutationCount: 0, multipleTesting: 'none'});
  // Counts and degrees are f32 columns, exact only below 2^24 rows.
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `limit-${serial++}`, format, length);
  expect(
    () =>
      new GPUGlobalSpatialStatistics(
        createProps(graph, {
          weights: {
            offsets: view('uint32', 2 ** 24 + 1),
            neighbors: view('uint32', 4),
            weights: view('float32', 4)
          },
          values: view('float32', 2 ** 24)
        })
      )
  ).toThrow(/2\^24 rows/);
});

it('the oracle reproduces esda on the island, subset and re-standardized fixtures', () => {
  const lists = REFERENCE.rows.map(row =>
    row.neighbors.map((neighbor, slot): [number, number] => [neighbor, row.weights[slot]])
  );
  const weights = createWeights(lists);
  const binary = createWeights(
    lists.map(list => list.map(([neighbor]): [number, number] => [neighbor, 1]))
  );
  const values = Float32Array.from(REFERENCE.values);
  const mask = Uint32Array.from(REFERENCE.values, (_, row) =>
    (REFERENCE.masked as readonly number[]).includes(row) ? 0 : 1
  );
  const cases = [
    ['keep', {weights, values, islands: 'keep'}],
    ['excludeSubset', {weights, values, islands: 'exclude'}],
    // Re-standardizing the raw weights over the unmasked neighbors equals standardizing first.
    ['standardizedMasked', {weights, values, mask, islands: 'keep', standardized: true}],
    ['binaryKeep', {weights: binary, values, islands: 'keep'}],
    ['binaryExclude', {weights: binary, values, islands: 'exclude'}]
  ] as const;
  for (const [name, input] of cases) {
    const reference = REFERENCE[name];
    const oracle = computeGlobalSpatialStatisticsOracle({...input});
    expect(oracle.count, name).toBe(reference.count);
    for (const key of ['s0', 's1', 's2', 'mean', 'variance'] as const) {
      expectClose(oracle[key], reference[key], `${name} ${key}`);
    }
    for (const statistic of ['moran', 'geary', 'getisOrdG'] as const) {
      const expected = (reference as unknown as Record<string, Record<string, number>>)[statistic];
      if (!expected) {
        continue;
      }
      for (const field of [
        'statistic',
        'expected',
        'varianceNormality',
        'zNormality',
        'varianceRandomization',
        'zRandomization'
      ]) {
        if (field in expected) {
          // The oracle p-values use the Numerical Recipes erfc (fractional error 1.2e-7).
          expectClose(
            oracle[statistic][field as 'statistic'],
            expected[field],
            `${name} ${statistic}.${field}`
          );
        }
      }
      for (const [suffix, alternative] of [
        ['', 'two-sided'],
        ['Greater', 'greater'],
        ['Lesser', 'lesser']
      ] as const) {
        const expectedP = expected[`pRandomization${suffix}`];
        const actualP = getPValue(expected['zRandomization'], alternative);
        expect(Math.abs(actualP - expectedP), `${name} ${statistic} p ${alternative}`).toBeLessThan(
          1e-6 + 1e-6 * expectedP
        );
      }
    }
  }
  // Islands: two island rows, excluded from n under 'exclude' and kept under 'keep'.
  expect(computeGlobalSpatialStatisticsOracle({weights, values}).islandCount).toBe(2);
  expect(computeGlobalSpatialStatisticsOracle({weights, values}).count).toBe(38);
  expect(computeGlobalSpatialStatisticsOracle({weights, values, islands: 'keep'}).count).toBe(40);
});

it('the oracle makes General G NaN for negative data and everything NaN for truncated weights', () => {
  const weights = createGridWeights(6, 6);
  const values = Float32Array.from({length: 36}, (_, index) => 1 + (index % 7));
  const clean = computeGlobalSpatialStatisticsOracle({weights, values});
  expect(Number.isFinite(clean.getisOrdG.statistic)).toBe(true);
  const negative = Float32Array.from(values);
  negative[4] = -2;
  const withNegative = computeGlobalSpatialStatisticsOracle({weights, values: negative});
  for (const value of Object.values(withNegative.getisOrdG)) {
    expect(value).toBeNaN();
  }
  expect(Number.isFinite(withNegative.moran.statistic)).toBe(true);
  const truncated = computeGlobalSpatialStatisticsOracle({weights, values, truncated: true});
  expect(truncated.count).toBe(36);
  expect(truncated.s0).toBeNaN();
  expect(truncated.moran.statistic).toBeNaN();
  expect(truncated.joinCount.blackBlack).toBeNaN();
});

it('islands and join-count kernels add to the global counters once per workgroup, from lane 0', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPUGlobalSpatialStatistics(
    createProps(graph, {statistics: ['moran', 'joinCount']})
  ).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  device.destroy();
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  const islands = sources.find(({id}) => id.endsWith('-islands'))!;
  const joinCounts = sources.find(({source}) => source.includes('groupBlackBlack'))!;
  expect(joinCounts).toBeDefined();
  for (const [kernel, expected] of [
    [islands, 2],
    [joinCounts, 3]
  ] as const) {
    const body = kernel.source.slice(kernel.source.indexOf('fn main('));
    const adds = body.match(/atomicAdd\(&counters/g) ?? [];
    expect(adds, kernel.id).toHaveLength(expected);
    // Every global add sits after the barrier, inside the lane 0 block.
    const barrier = body.indexOf('workgroupBarrier()');
    const laneZero = body.indexOf('if (localInvocationIndex == 0u)');
    expect(barrier, kernel.id).toBeGreaterThan(0);
    expect(laneZero, kernel.id).toBeGreaterThan(barrier);
    expect(body.indexOf('atomicAdd(&counters'), kernel.id).toBeGreaterThan(laneZero);
    // Barrier at function scope, not under the row guard.
    expect(body, kernel.id).toMatch(/\n {2}workgroupBarrier\(\);/);
    expect(body, kernel.id).toContain('if (index < ROWS) {');
    expect(getWorkgroupMemoryBytes(kernel.source), kernel.id).toBe(expected * 4);
  }
});

it('GPUGlobalSpatialStatistics takes uint32 counts through one conversion node (A-5)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'global-statistics-counts'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `counts-${serial++}`, format, length);
  const floatIds = new GPUGlobalSpatialStatistics({...createProps(graph), id: 'f'})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(floatIds.some(id => id.endsWith('-to-float32'))).toBe(false);
  const countIds = new GPUGlobalSpatialStatistics({
    ...createProps(graph),
    id: 'c',
    statistics: ['moran', 'bivariateMoran'],
    values: view('uint32', 10),
    secondValues: view('uint32', 10)
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(countIds.slice(0, 2)).toEqual(['c-values-to-float32', 'c-second-values-to-float32']);
  expect(
    () => new GPUGlobalSpatialStatistics(createProps(graph, {values: view('uint32', 9)}))
  ).toThrow(/values/);
});

it('the join-count oracle reproduces esda counts and the exhaustive randomization moments', () => {
  // Oracle: test/oracles/statistics-global-autocorrelation.py (esda Join_Counts; moments by
  // enumerating every placement of the black rows).
  for (const scene of JOIN_COUNT_REFERENCE) {
    const weights = createWeights(
      scene.rows.map(row =>
        row.neighbors.map((neighbor, slot): [number, number] => [neighbor, row.weights[slot]])
      )
    );
    const oracle = computeGlobalSpatialStatisticsOracle({
      weights,
      values: Float32Array.from(scene.values),
      islands: 'keep'
    });
    expect(oracle.blackCount, scene.name).toBe(scene.blackCount);
    const joinCount = oracle.joinCount as unknown as Record<string, number>;
    for (const key of ['blackBlack', 'blackWhite', 'whiteWhite', 'joins'] as const) {
      expect(joinCount[key], `${scene.name} ${key}`).toBe(scene[key]);
    }
    for (const key of [
      'expectedBlackBlack',
      'varianceBlackBlack',
      'zBlackBlack',
      'expectedBlackWhite',
      'varianceBlackWhite',
      'zBlackWhite'
    ] as const) {
      expectClose(joinCount[key], scene[key], `${scene.name} ${key}`);
    }
    // The oracle p-values use the Numerical Recipes erfc (fractional error 1.2e-7).
    for (const key of ['pBlackBlack', 'pBlackWhite'] as const) {
      expect(Math.abs(joinCount[key] - scene[key]), `${scene.name} ${key}`).toBeLessThanOrEqual(
        1.2e-7 * scene[key]
      );
    }
  }
});
