// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU specs of the golden-section bandwidth search and the Gaussian cutoff of
 * `GPUGeographicallyWeightedRegression`, against mgwr 2.2.1 (`Sel_BW(...).search('golden_section')`
 * and the exact Gaussian `GWR` fits of `test/oracles/statistics-gwr.py`) on the planar 144-row scene.
 *
 * Tolerances, derived:
 * - Adaptive bandwidths are integers and the float32 search takes mgwr's path (the float32 mirror in
 *   `gwr-golden-section-reference.ts` reproduces every pinned adaptive search), so the selected `k`
 *   must equal mgwr's exactly.
 * - Fixed bandwidths: the kernel stops when the two interior AICc values differ by at most
 *   `max(1e-6, 4e-6 |AICc|)` (about 1e-3 at AICc 254) and each float32 AICc carries about the same
 *   noise. A bandwidth within `delta` of the AICc minimum is indistinguishable, so the selected
 *   value may differ from mgwr's by `sqrt(2 delta / f'')`, with `f''` the second derivative of the
 *   dense mgwr curve at its minimum (about 95) and `delta` four stop tolerances, plus mgwr's own
 *   rounding of its answer to two decimals (0.005). The AICc at the selection may exceed mgwr's by
 *   `delta` as well.
 * - Gaussian cutoff `c`: the relative effect on the local sums is at most the omitted second-moment
 *   share `(1 + c^2 / 2) exp(-c^2 / 2)` of a locally uniform density; the node spec pins the float64
 *   coefficient error of this scene at 20 times that share, which bounds the cutoff error here, and
 *   the float32 fit error of the exact Gaussian (5e-3 relative for the arrays, as the other mgwr
 *   comparisons of the scene) adds to it.
 */

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {createNonstationarityScene} from './geographically-weighted-regression-scenes';
import {
  MGWR_BANDWIDTH_SEARCH_REFERENCE,
  MGWR_GOLDEN_SECTION_REFERENCE,
  MGWR_INFERENCE_REFERENCE,
  type MgwrGoldenSectionReference
} from './gwr-mgwr-inference-reference';

const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
const STATE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE;
const TABLE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE;
const STRIDE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE;
const LADDER_CAPACITY = 8;

type Scene = {
  positions: Float32Array;
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
  mask?: Uint32Array;
};

type RunOptions = {
  bandwidthSearch?: 'ladder' | 'golden-section';
  maximumSearchIterations?: number;
  maximumNeighborCount?: number;
  gaussianCutoff?: number;
  indexGridSize?: readonly [number, number] | false;
  /** Use the contributor's automatic grid index instead of the scan (the default of these specs). */
  automaticGrid?: boolean;
  refinement?: boolean | number;
};

type Result = {
  coefficients: number[];
  candidateTable: number[];
  selectedBandwidth: number[];
  summary: number[];
  searchState: number[];
  tableRowCount: number;
};

/** Builds, runs and reads back one fit; the golden-section search state is always attached. */
async function runFit(
  device: Device,
  scene: Scene,
  settings: GPUGeographicallyWeightedRegressionSettings,
  options: RunOptions
): Promise<Result> {
  const rows = scene.response.length;
  const {predictorCount} = scene;
  const p = predictorCount + 1;
  const isGoldenSection = options.bandwidthSearch === 'golden-section';
  const maximumSearchIterations = options.maximumSearchIterations ?? 30;
  const refinementPassCount =
    options.refinement === false
      ? 0
      : typeof options.refinement === 'number'
        ? options.refinement
        : 2;
  const tableRowCount = isGoldenSection
    ? maximumSearchIterations + 1
    : (1 + refinementPassCount) * LADDER_CAPACITY;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'gwr-golden-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
  });
  const outputs = {
    coefficients: track(createOutputBuffer(device, rows * p)),
    candidateTable: track(createOutputBuffer(device, tableRowCount * STRIDE)),
    selectedBandwidth: track(createOutputBuffer(device, 2)),
    summary: track(
      createOutputBuffer(device, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
    ),
    searchState: track(
      createOutputBuffer(device, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH)
    )
  };
  const graph = new GPUCommandGraph(device, {id: 'gwr-golden-graph'});
  const float = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', length);
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, scene.positions)),
        'float32x2',
        rows
      ),
      predictors: importGraphBuffer(
        graph,
        'predictors',
        track(createInputBuffer(device, scene.predictors)),
        'float32',
        rows * predictorCount
      ),
      predictorCount,
      response: importGraphBuffer(
        graph,
        'response',
        track(createInputBuffer(device, scene.response)),
        'float32',
        rows
      ),
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            rows
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      maximumBandwidthCount: LADDER_CAPACITY,
      maximumNeighborCount: options.maximumNeighborCount ?? 128,
      bandwidthSearch: options.bandwidthSearch,
      maximumSearchIterations: options.maximumSearchIterations,
      gaussianCutoff: options.gaussianCutoff,
      refinement: options.refinement,
      indexGridSize: options.automaticGrid ? undefined : (options.indexGridSize ?? false),
      output: {
        coefficients: float('coefficients', rows * p),
        candidateTable: float('candidateTable', tableRowCount * STRIDE),
        selectedBandwidth: float('selectedBandwidth', 2),
        summary: float('summary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH),
        ...(isGoldenSection
          ? {
              searchState: float(
                'searchState',
                GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH
              )
            }
          : {})
      }
    })
  );
  const compiled = graph.compile();
  try {
    parameterBuffer.write(
      getGPUGeographicallyWeightedRegressionParameterValues(settings, LADDER_CAPACITY)
    );
    submitGraph(device, compiled, undefined);
    return {
      coefficients: await readFloat32(outputs.coefficients, rows * p),
      candidateTable: await readFloat32(outputs.candidateTable, tableRowCount * STRIDE),
      selectedBandwidth: await readFloat32(outputs.selectedBandwidth, 2),
      summary: await readFloat32(
        outputs.summary,
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
      ),
      searchState: isGoldenSection
        ? await readFloat32(
            outputs.searchState,
            GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH
          )
        : [],
      tableRowCount
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

const planar = createNonstationarityScene(12, 77);

/** Finite candidate-table rows as `[row, value, aicc]`, in table order. */
function getEvaluations(result: Result): [number, number, number][] {
  const evaluations: [number, number, number][] = [];
  for (let row = 0; row < result.tableRowCount; row++) {
    const value = result.candidateTable[row * STRIDE + TABLE.VALUE];
    if (Number.isFinite(value)) {
      evaluations.push([row, value, result.candidateTable[row * STRIDE + TABLE.AICC]]);
    }
  }
  return evaluations;
}

function getSettings(reference: MgwrGoldenSectionReference, bounds: number[]) {
  return {
    kernel: reference.kernel,
    bandwidthMode: reference.fixed ? 'fixed' : 'adaptive',
    bandwidths: bounds
  } as GPUGeographicallyWeightedRegressionSettings;
}

it('golden section selects the adaptive k of mgwr exactly and follows its path', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of MGWR_GOLDEN_SECTION_REFERENCE.filter(entry => !entry.fixed)) {
    const label = reference.label;
    const result = await runFit(
      device,
      planar,
      getSettings(reference, [reference.lower, reference.upper]),
      {bandwidthSearch: 'golden-section'}
    );
    expect(result.selectedBandwidth[1], label).toBe(reference.bandwidth);
    // Every evaluated k is one mgwr evaluated, with its AICc (float32 and the 1.00001 neighbour
    // margin against mgwr's 1.0000001 move the AICc by well under 5e-4 relative).
    const evaluations = getEvaluations(result);
    expect(evaluations.length, label).toBeGreaterThanOrEqual(reference.evaluated.length - 1);
    for (const [row, value, aicc] of evaluations) {
      const mgwr = reference.evaluated.find(([k]) => k === value);
      expect(mgwr, `${label} row ${row} evaluates k = ${value}`).toBeDefined();
      expect(Math.abs(aicc - mgwr![1]), `${label} AICc at k = ${value}`).toBeLessThanOrEqual(
        5e-4 * Math.abs(mgwr![1])
      );
    }
    expect(Math.abs(result.summary[SUMMARY.AICC] - reference.aicc)).toBeLessThanOrEqual(
      5e-4 * Math.abs(reference.aicc)
    );
    // The float32 floor can only stop the search earlier than mgwr's 1e-6.
    expect(result.searchState[STATE.ITERATION_COUNT], label).toBeLessThanOrEqual(
      reference.iterationCount
    );
    expect(result.searchState[STATE.ACTIVE], label).toBe(0);
    expect(result.searchState[STATE.CONVERGED], label).toBe(1);
    // The table row of the selection holds the selected value.
    const selectedRow = result.selectedBandwidth[0] * STRIDE;
    expect(result.candidateTable[selectedRow + TABLE.VALUE], label).toBe(reference.bandwidth);
    // Iterations after convergence evaluate nothing: their rows stay NaN.
    const used = result.searchState[STATE.EVALUATION_COUNT];
    for (let row = used; row < result.tableRowCount; row++) {
      expect(result.candidateTable[row * STRIDE + TABLE.AICC], `${label} row ${row}`).toBeNaN();
    }
  }
  device.destroy();
});

it('golden section computes the mgwr default bounds on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Adaptive: [40 + 2 (predictors + 1), n capped at the neighbour list] = [46, 128].
  const adaptiveReference = MGWR_GOLDEN_SECTION_REFERENCE.find(
    entry => entry.label === 'adaptive bisquare default lower, upper 128'
  )!;
  const adaptive = await runFit(device, planar, getSettings(adaptiveReference, [0, 0]), {
    bandwidthSearch: 'golden-section',
    maximumNeighborCount: 128
  });
  expect(adaptive.selectedBandwidth[1]).toBe(adaptiveReference.bandwidth);
  // A positive first bound overrides only the lower default: the 6-to-128 search.
  const wideReference = MGWR_GOLDEN_SECTION_REFERENCE.find(
    entry => entry.label === 'adaptive bisquare 6 to 128'
  )!;
  const wide = await runFit(device, planar, getSettings(wideReference, [6, 0]), {
    bandwidthSearch: 'golden-section',
    maximumNeighborCount: 128
  });
  expect(wide.selectedBandwidth[1]).toBe(wideReference.bandwidth);
  // Fixed: [min_dist / 2, 2 max_dist] from the data, exact in float32 up to the sqrt of the sums.
  for (const label of ['fixed bisquare default bounds', 'fixed gaussian default bounds']) {
    const reference = MGWR_GOLDEN_SECTION_REFERENCE.find(entry => entry.label === label)!;
    const fixed = await runFit(device, planar, getSettings(reference, [0, 0]), {
      bandwidthSearch: 'golden-section'
    });
    // The first search iteration moves the bracket, so the bounds are read from the first two
    // table rows: they are the interior points of [lower, upper] with mgwr's delta.
    const first = fixed.candidateTable[TABLE.VALUE];
    const second = fixed.candidateTable[STRIDE + TABLE.VALUE];
    const delta = 0.38197;
    expect(first, `${label} b`).toBeCloseTo(
      reference.lower + delta * (reference.upper - reference.lower),
      3
    );
    expect(second, `${label} d`).toBeCloseTo(
      reference.upper - delta * (reference.upper - reference.lower),
      3
    );
  }
  device.destroy();
});

/** Second derivative of the dense mgwr fixed bisquare AICc curve at its minimum. */
function getFixedCurvature(): number {
  const {distances, aicc} = MGWR_BANDWIDTH_SEARCH_REFERENCE.fixed;
  const index = aicc.indexOf(Math.min(...aicc));
  const step = distances[1] - distances[0];
  return (aicc[index - 1] - 2 * aicc[index] + aicc[index + 1]) / (step * step);
}

/**
 * AICc slack of the float32 search (four stop tolerances: the stop test, the float32 noise of each
 * of the two scores, and a margin) and the bandwidth distance it allows around the mgwr optimum.
 */
function getFixedSearchTolerance(aicc: number): {delta: number; bandwidth: number} {
  const delta = 4 * Math.max(1e-6, 4e-6 * Math.abs(aicc));
  return {delta, bandwidth: 0.005 + Math.sqrt((2 * delta) / getFixedCurvature())};
}

it('golden section selects the fixed bandwidth of mgwr within the derived tolerance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of MGWR_GOLDEN_SECTION_REFERENCE.filter(entry => entry.fixed)) {
    const label = reference.label;
    const result = await runFit(
      device,
      planar,
      getSettings(reference, [reference.lower, reference.upper]),
      {bandwidthSearch: 'golden-section'}
    );
    const tolerance = getFixedSearchTolerance(reference.aicc);
    expect(
      Math.abs(result.selectedBandwidth[1] - reference.unroundedBandwidth),
      `${label}: ${result.selectedBandwidth[1]} vs ${reference.unroundedBandwidth}`
    ).toBeLessThanOrEqual(tolerance.bandwidth);
    expect(result.summary[SUMMARY.AICC] - reference.aicc, label).toBeLessThanOrEqual(
      tolerance.delta
    );
    // mgwr needs 20 to 25 iterations at 1e-6; the float32 floor stops earlier, inside the default 30.
    expect(result.searchState[STATE.ITERATION_COUNT], label).toBeLessThanOrEqual(
      reference.iterationCount
    );
    expect(result.searchState[STATE.CONVERGED], label).toBe(1);
  }
  device.destroy();
});

it('golden section agrees with the float64 oracle on the evaluated bandwidths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [10, 120]
  };
  const gpu = await runFit(device, planar, settings, {bandwidthSearch: 'golden-section'});
  const oracle = computeGeographicallyWeightedRegressionOnCPU({
    ...planar,
    settings,
    bandwidthSearch: 'golden-section',
    maximumNeighborCount: 128
  });
  expect(gpu.selectedBandwidth[1]).toBe(oracle.selectedValue);
  // The float32 AICc can flip the stop test of the last comparison (a difference within a few
  // 1e-4 of the tolerance), which moves the iteration count by at most one.
  expect(
    Math.abs(gpu.searchState[STATE.ITERATION_COUNT] - (oracle.searchIterationCount ?? 0))
  ).toBeLessThanOrEqual(1);
  let sharedRows = 0;
  for (let row = 0; row < oracle.candidates.length; row++) {
    const want = oracle.candidates[row];
    const got = gpu.candidateTable[row * STRIDE + TABLE.VALUE];
    if (Number.isNaN(want.value) || Number.isNaN(got)) {
      continue;
    }
    sharedRows++;
    expect(got, `row ${row}`).toBe(want.value);
    expect(
      Math.abs(gpu.candidateTable[row * STRIDE + TABLE.AICC] - want.aicc),
      `AICc row ${row}`
    ).toBeLessThanOrEqual(5e-4 * Math.abs(want.aicc));
  }
  expect(sharedRows).toBeGreaterThanOrEqual(5);
  // The fit at the selection matches the oracle's local coefficients.
  for (let index = 0; index < oracle.coefficients.length; index++) {
    expect(
      Math.abs(gpu.coefficients[index] - oracle.coefficients[index]),
      `coefficient ${index}`
    ).toBeLessThanOrEqual(5e-3 * Math.max(0.1, Math.abs(oracle.coefficients[index])));
  }
  device.destroy();
});

it('golden section handles a single value, the iteration cap, masks and repeated runs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A single value is a fixed bandwidth: no search, row 0, mgwr fit at 2.2.
  const single = await runFit(
    device,
    planar,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [2.2]},
    {bandwidthSearch: 'golden-section'}
  );
  expect(single.selectedBandwidth[0]).toBe(0);
  expect(single.selectedBandwidth[1]).toBeCloseTo(2.2, 6);
  expect(single.searchState[STATE.ACTIVE]).toBe(0);
  expect(single.searchState[STATE.ITERATION_COUNT]).toBe(0);
  expect(single.searchState[STATE.OPTIMUM_ROW]).toBe(0);
  expect(
    Math.abs(single.summary[SUMMARY.AICC] - MGWR_INFERENCE_REFERENCE.fixedBisquare22.aicc)
  ).toBeLessThanOrEqual(2e-3 * Math.abs(MGWR_INFERENCE_REFERENCE.fixedBisquare22.aicc));
  expect(getEvaluations(single).length).toBe(1);

  // An iteration cap stops the search unconverged after exactly that many comparisons.
  const adaptiveSettings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [46, 128]
  };
  const capped = await runFit(device, planar, adaptiveSettings, {
    bandwidthSearch: 'golden-section',
    maximumSearchIterations: 3
  });
  expect(capped.searchState[STATE.ITERATION_COUNT]).toBe(3);
  expect(capped.searchState[STATE.CONVERGED]).toBe(0);
  expect(capped.searchState[STATE.ACTIVE]).toBe(0);
  // Iterations 1 to 3 evaluate 2 + 1 + 1 candidates: the comparison of the last schedules none.
  expect(getEvaluations(capped).length).toBe(4);
  const oracleCapped = computeGeographicallyWeightedRegressionOnCPU({
    ...planar,
    settings: adaptiveSettings,
    bandwidthSearch: 'golden-section',
    maximumSearchIterations: 3
  });
  expect(capped.selectedBandwidth[1]).toBe(oracleCapped.selectedValue);

  // Masked rows do not enter the default bounds or the search.
  const mask = new Uint32Array(planar.response.length).fill(1);
  for (let row = 0; row < mask.length; row += 7) {
    mask[row] = 0;
  }
  const masked = await runFit(
    device,
    {...planar, mask},
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [0, 0]},
    {bandwidthSearch: 'golden-section'}
  );
  const maskedOracle = computeGeographicallyWeightedRegressionOnCPU({
    ...planar,
    mask,
    settings: {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [0, 0]},
    bandwidthSearch: 'golden-section',
    maximumNeighborCount: 128
  });
  expect(masked.selectedBandwidth[1]).toBe(maskedOracle.selectedValue);

  // The same graph run twice gives bit-identical state, table and coefficients.
  const again = await runFit(device, planar, adaptiveSettings, {
    bandwidthSearch: 'golden-section'
  });
  const first = await runFit(device, planar, adaptiveSettings, {bandwidthSearch: 'golden-section'});
  expect(Array.from(again.candidateTable)).toEqual(Array.from(first.candidateTable));
  expect(Array.from(again.searchState)).toEqual(Array.from(first.searchState));
  expect(Array.from(again.coefficients)).toEqual(Array.from(first.coefficients));
  device.destroy();
});

function expectCoefficientsClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(
      Math.abs(actual[index] - expected[index]) / Math.max(0.1, Math.abs(expected[index])),
      `${label}[${index}] ${actual[index]} vs ${expected[index]}`
    ).toBeLessThanOrEqual(tolerance);
  }
}

it('the Gaussian cutoff fits within its truncation bound of the exact Gaussian fit of mgwr', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixedSettings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'gaussian',
    bandwidthMode: 'fixed',
    bandwidths: [1.5]
  };
  const adaptiveSettings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'gaussian',
    bandwidthMode: 'adaptive',
    bandwidths: [20]
  };
  const floatTolerance = 5e-3;
  const exactFixed = MGWR_INFERENCE_REFERENCE.fixedGaussian15Full;
  const exactAdaptive = MGWR_INFERENCE_REFERENCE.adaptiveGaussian20Full;
  for (const cutoff of [3, 4, 5, 6]) {
    const share = (1 + (cutoff * cutoff) / 2) * Math.exp(-(cutoff * cutoff) / 2);
    const tolerance = floatTolerance + 20 * share;
    const fixed = await runFit(device, planar, fixedSettings, {
      gaussianCutoff: cutoff,
      refinement: false
    });
    expectCoefficientsClose(
      fixed.coefficients,
      exactFixed.coefficients,
      tolerance,
      `fixed c=${cutoff}`
    );
    expect(
      Math.abs(fixed.summary[SUMMARY.AICC] - exactFixed.aicc),
      `fixed c=${cutoff} AICc`
    ).toBeLessThanOrEqual(tolerance * Math.abs(exactFixed.aicc));
    const adaptive = await runFit(device, planar, adaptiveSettings, {
      gaussianCutoff: cutoff,
      refinement: false
    });
    expectCoefficientsClose(
      adaptive.coefficients,
      exactAdaptive.coefficients,
      tolerance,
      `adaptive c=${cutoff}`
    );
    // The cutoff fit is the truncated fit of the float64 oracle at the float32 tolerance.
    const oracle = computeGeographicallyWeightedRegressionOnCPU({
      ...planar,
      settings: fixedSettings,
      maximumBandwidthCount: 1,
      gaussianCutoff: cutoff
    });
    expectCoefficientsClose(
      fixed.coefficients,
      oracle.coefficients,
      floatTolerance,
      `oracle c=${cutoff}`
    );
  }
  // Without a cutoff the exact Gaussian fit is unchanged.
  const exact = await runFit(device, planar, fixedSettings, {refinement: false});
  expectCoefficientsClose(exact.coefficients, exactFixed.coefficients, floatTolerance, 'exact');
  device.destroy();
});

it('the Gaussian cutoff gives the same fit through the grid index as through the scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const settings of [
    {kernel: 'gaussian', bandwidthMode: 'fixed', bandwidths: [1.2]},
    {kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [25]}
  ] as GPUGeographicallyWeightedRegressionSettings[]) {
    const scan = await runFit(device, planar, settings, {
      gaussianCutoff: 4,
      indexGridSize: false,
      refinement: false
    });
    const grid = await runFit(device, planar, settings, {
      gaussianCutoff: 4,
      indexGridSize: [6, 6],
      refinement: false
    });
    // Cell order against row order changes only the float32 summation order (about 1e-5).
    expectCoefficientsClose(
      grid.coefficients,
      scan.coefficients,
      1e-4,
      `${settings.bandwidthMode} grid`
    );
    expect(Math.abs(grid.summary[SUMMARY.AICC] - scan.summary[SUMMARY.AICC])).toBeLessThanOrEqual(
      1e-4 * Math.abs(scan.summary[SUMMARY.AICC])
    );
  }
  // The grid golden-section search takes the same adaptive path as the scan search.
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [46, 128]
  };
  const scanSearch = await runFit(device, planar, settings, {bandwidthSearch: 'golden-section'});
  const gridSearch = await runFit(device, planar, settings, {
    bandwidthSearch: 'golden-section',
    indexGridSize: [6, 6]
  });
  expect(gridSearch.selectedBandwidth[1]).toBe(scanSearch.selectedBandwidth[1]);
  // Fixed bounds on a grid: the farthest distance is an upper bound (a corner of the extent), so
  // the upper bracket end is at least the scan's exact one.
  const fixedSettings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    bandwidths: [0, 0]
  };
  const fixedScan = await runFit(device, planar, fixedSettings, {
    bandwidthSearch: 'golden-section'
  });
  const fixedGrid = await runFit(device, planar, fixedSettings, {
    bandwidthSearch: 'golden-section',
    indexGridSize: [6, 6]
  });
  expect(fixedGrid.candidateTable[TABLE.VALUE]).toBeGreaterThanOrEqual(
    fixedScan.candidateTable[TABLE.VALUE]
  );
  const fixedReference = MGWR_GOLDEN_SECTION_REFERENCE.find(
    entry => entry.label === 'fixed bisquare default bounds'
  )!;
  expect(
    Math.abs(fixedGrid.selectedBandwidth[1] - fixedReference.unroundedBandwidth)
  ).toBeLessThanOrEqual(getFixedSearchTolerance(fixedReference.aicc).bandwidth);
  device.destroy();
});

function createLargeScene(side: number): Scene {
  let state = 20260614;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const rows = side * side;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    const x = ((row % side) + 0.2 + 0.6 * random()) * (100 / side);
    const y = (Math.floor(row / side) + 0.2 + 0.6 * random()) * (100 / side);
    positions[2 * row] = x;
    positions[2 * row + 1] = y;
    predictors[2 * row] = random() * 4 - 2;
    predictors[2 * row + 1] = random() * 4 - 2;
    response[row] =
      1 + 0.02 * x * predictors[2 * row] + 0.01 * y * predictors[2 * row + 1] + 0.2 * random();
  }
  return {positions, predictors, predictorCount: 2, response};
}

/**
 * A/B, same run, same adapter: ladder search with its default refinement against the golden-section
 * search. Logs both wall times (submit to read-back) and the AICc each reaches; the golden section
 * must reach an AICc within the ladder's plus the derived tolerance, and the log shows the cost.
 */
async function compareSearches(
  device: Device,
  label: string,
  scene: Scene,
  ladder: GPUGeographicallyWeightedRegressionSettings,
  bounds: GPUGeographicallyWeightedRegressionSettings,
  options: RunOptions
): Promise<void> {
  const time = async (run: () => Promise<Result>): Promise<[Result, number]> => {
    await run();
    const start = performance.now();
    const result = await run();
    return [result, performance.now() - start];
  };
  const [ladderResult, ladderMilliseconds] = await time(() =>
    runFit(device, scene, ladder, {...options, bandwidthSearch: 'ladder'})
  );
  const [goldenResult, goldenMilliseconds] = await time(() =>
    runFit(device, scene, bounds, {...options, bandwidthSearch: 'golden-section'})
  );
  const evaluated = getEvaluations(goldenResult).length;
  // eslint-disable-next-line no-console
  console.log(
    `GWR search A/B ${label}: ladder + refinement ${ladderMilliseconds.toFixed(1)} ms ` +
      `(k or h = ${ladderResult.selectedBandwidth[1]}, AICc ${ladderResult.summary[SUMMARY.AICC]}), ` +
      `golden section ${goldenMilliseconds.toFixed(1)} ms (${evaluated} evaluations, ` +
      `${goldenResult.searchState[STATE.ITERATION_COUNT]} iterations, ` +
      `k or h = ${goldenResult.selectedBandwidth[1]}, AICc ${goldenResult.summary[SUMMARY.AICC]}), ` +
      `speedup x${(ladderMilliseconds / goldenMilliseconds).toFixed(2)}`
  );
  expect(Number.isFinite(goldenResult.summary[SUMMARY.AICC]), label).toBe(true);
  expect(Number.isFinite(ladderResult.summary[SUMMARY.AICC]), label).toBe(true);
  // A search over a different path may land a little off the ladder's optimum, never far from it.
  expect(goldenResult.summary[SUMMARY.AICC], label).toBeLessThanOrEqual(
    ladderResult.summary[SUMMARY.AICC] + 0.01 * Math.abs(ladderResult.summary[SUMMARY.AICC])
  );
}

it('A/B: ladder with refinement against golden section on the 144-row scene', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  await compareSearches(
    device,
    '144 rows, adaptive bisquare',
    planar,
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [10, 20, 40, 80, 120]},
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [6, 128]},
    {}
  );
  await compareSearches(
    device,
    '144 rows, fixed bisquare',
    planar,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [1.5, 2, 3, 4, 6]},
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [0, 0]},
    {}
  );
  device.destroy();
});

it('A/B: ladder with refinement against golden section on a 50,176-row bisquare grid scene', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createLargeScene(224);
  // About 60 rows per bandwidth disc at the middle of the ladder: h = sqrt(60 * 1e4 / (n pi)) = 1.95.
  await compareSearches(
    device,
    '50,176 rows, fixed bisquare, grid index',
    scene,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [1, 1.5, 2, 2.5, 3.5]},
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [0.5, 6]},
    {automaticGrid: true}
  );
  await compareSearches(
    device,
    '50,176 rows, adaptive bisquare, grid index',
    scene,
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [30, 50, 80, 100, 128]},
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [0, 0]},
    {automaticGrid: true}
  );
  device.destroy();
}, 600000);
