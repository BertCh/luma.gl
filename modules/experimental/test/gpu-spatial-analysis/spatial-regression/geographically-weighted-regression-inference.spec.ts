// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU specs of the GWR correctness round: local standard errors, t values and significance
 * (mgwr `bse`, `tvalues`, `adj_alpha`, `critical_tval`, `filter_tvals`), singular-row handling,
 * bandwidth refinement against mgwr's golden-section search, great-circle distances against mgwr
 * `spherical=True`, the model result, and the 11-predictor limit.
 *
 * Expected values come from mgwr 2.2.1 through `gwr-mgwr-inference-reference.ts` (generator
 * `test/oracles/statistics-gwr.py`) and from the float64 oracle for definitions mgwr does not have (singular-row
 * handling, refinement ladders).
 */

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {getGPUGeographicallyWeightedRegressionModelResult} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  type GPUGeographicallyWeightedRegressionSettings
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {
  createGeographicScene,
  createNonstationarityScene,
  createSingularRowScene
} from './geographically-weighted-regression-scenes';
import {
  MGWR_BANDWIDTH_SEARCH_REFERENCE,
  MGWR_INFERENCE_REFERENCE,
  MGWR_SPHERICAL_REFERENCE
} from './gwr-mgwr-inference-reference';
import {requireTestDevice} from '../../utils/require-test-device';

const STATUS = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS;
const SUMMARY = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
const TABLE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE;
const TABLE_STRIDE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE;

type Scene = {
  positions: Float32Array;
  predictors: Float32Array;
  predictorCount: number;
  response: Float32Array;
  mask?: Uint32Array;
};

type RunOptions = {
  ladderCapacity: number;
  refinement?: boolean | number;
  spatialContext?: GPUSpatialContext;
  maximumNeighborCount?: number;
};

type Result = {
  coefficients: number[];
  standardErrors: number[];
  tValues: number[];
  significant: number[];
  localStatus: number[];
  fitted: number[];
  residuals: number[];
  hatDiagonal: number[];
  bandwidthScores: number[];
  singularRowCounts: number[];
  candidateTable: number[];
  selectedBandwidth: number[];
  summary: number[];
};

/** Builds, runs and reads back one fit with every output attached. */
async function runFit(
  device: Device,
  scene: Scene,
  settings: GPUGeographicallyWeightedRegressionSettings,
  options: RunOptions
): Promise<Result> {
  const rows = scene.response.length;
  const {predictorCount} = scene;
  const p = predictorCount + 1;
  const {ladderCapacity} = options;
  const passCount =
    1 +
    (ladderCapacity < 2 || options.refinement === false
      ? 0
      : typeof options.refinement === 'number'
        ? options.refinement
        : 2);
  const tableLength = passCount * ladderCapacity * TABLE_STRIDE;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'gwr-inference-parameters',
    format: 'float32',
    length: getGPUGeographicallyWeightedRegressionParameterLength(ladderCapacity)
  });
  const outputs = {
    coefficients: track(createOutputBuffer(device, rows * p)),
    standardErrors: track(createOutputBuffer(device, rows * p)),
    tValues: track(createOutputBuffer(device, rows * p)),
    significant: track(createOutputBuffer(device, rows * p)),
    localStatus: track(createOutputBuffer(device, rows)),
    fitted: track(createOutputBuffer(device, rows)),
    residuals: track(createOutputBuffer(device, rows)),
    hatDiagonal: track(createOutputBuffer(device, rows)),
    bandwidthScores: track(createOutputBuffer(device, ladderCapacity)),
    singularRowCounts: track(createOutputBuffer(device, ladderCapacity)),
    candidateTable: track(createOutputBuffer(device, tableLength)),
    selectedBandwidth: track(createOutputBuffer(device, 2)),
    summary: track(
      createOutputBuffer(device, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
    )
  };
  const graph = new GPUCommandGraph(device, {id: 'gwr-inference-graph'});
  const float = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', length);
  const unsigned = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'uint32', length);
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
      maximumBandwidthCount: ladderCapacity,
      maximumNeighborCount: options.maximumNeighborCount ?? 64,
      refinement: options.refinement,
      spatialContext: options.spatialContext,
      indexGridSize: false,
      output: {
        coefficients: float('coefficients', rows * p),
        standardErrors: float('standardErrors', rows * p),
        tValues: float('tValues', rows * p),
        significant: unsigned('significant', rows * p),
        localStatus: unsigned('localStatus', rows),
        fitted: float('fitted', rows),
        residuals: float('residuals', rows),
        hatDiagonal: float('hatDiagonal', rows),
        bandwidthScores: float('bandwidthScores', ladderCapacity),
        bandwidthSingularRowCounts: unsigned('singularRowCounts', ladderCapacity),
        candidateTable: float('candidateTable', tableLength),
        selectedBandwidth: float('selectedBandwidth', 2),
        summary: float('summary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
      }
    })
  );
  const compiled = graph.compile();
  try {
    parameterBuffer.write(
      getGPUGeographicallyWeightedRegressionParameterValues(settings, ladderCapacity)
    );
    submitGraph(device, compiled, undefined);
    return {
      coefficients: await readFloat32(outputs.coefficients, rows * p),
      standardErrors: await readFloat32(outputs.standardErrors, rows * p),
      tValues: await readFloat32(outputs.tValues, rows * p),
      significant: await readUint32(outputs.significant, rows * p),
      localStatus: await readUint32(outputs.localStatus, rows),
      fitted: await readFloat32(outputs.fitted, rows),
      residuals: await readFloat32(outputs.residuals, rows),
      hatDiagonal: await readFloat32(outputs.hatDiagonal, rows),
      bandwidthScores: await readFloat32(outputs.bandwidthScores, ladderCapacity),
      singularRowCounts: await readUint32(outputs.singularRowCounts, ladderCapacity),
      candidateTable: await readFloat32(outputs.candidateTable, tableLength),
      selectedBandwidth: await readFloat32(outputs.selectedBandwidth, 2),
      summary: await readFloat32(
        outputs.summary,
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
      )
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/** NaN pattern must match; finite values agree within `tolerance` of `max(floor, |expected|)`. */
function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  floor: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const want = expected[index];
    const got = actual[index];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label}[${index}] should be NaN, got ${got}`).toBe(true);
    } else {
      expect(
        Math.abs(got - want) / Math.max(floor, Math.abs(want)),
        `${label}[${index}] ${got} vs ${want}`
      ).toBeLessThanOrEqual(tolerance);
    }
  }
}

type MgwrFit = (typeof MGWR_INFERENCE_REFERENCE)['fixedBisquare22'];

/**
 * Compares a GPU fit with an mgwr fit. float32 versus float64 and, for adaptive bandwidths,
 * mgwr's 1.0000001 versus the contributor's 1.00001 margin on the k-th neighbour distance bound
 * the agreement: `tolerance` applies to scalars, `arrayTolerance` to the local arrays.
 */
function expectMatchesMgwr(
  result: Result,
  reference: MgwrFit,
  label: string,
  tolerance: number,
  arrayTolerance: number
): void {
  expectClose(result.coefficients, reference.coefficients, arrayTolerance, 0.1, `${label} beta`);
  expectClose(
    result.standardErrors,
    reference.standardErrors,
    arrayTolerance,
    0.05,
    `${label} bse`
  );
  expectClose(result.tValues, reference.tValues, arrayTolerance, 1, `${label} t`);
  // The mask may differ only where |t| sits within the tolerance of the critical value.
  for (let index = 0; index < reference.significant.length; index++) {
    if (result.significant[index] !== reference.significant[index]) {
      expect(
        Math.abs(Math.abs(reference.tValues[index]) - reference.criticalT),
        `${label} mask[${index}] t = ${reference.tValues[index]}`
      ).toBeLessThan(arrayTolerance * 10);
    }
  }
  expectClose(
    [
      result.summary[SUMMARY.TRACE_OF_HAT],
      result.summary[SUMMARY.SIGMA_SQUARED],
      result.summary[SUMMARY.AICC],
      result.summary[SUMMARY.LOG_LIKELIHOOD],
      result.summary[SUMMARY.R_SQUARED],
      result.summary[SUMMARY.ADJUSTED_ALPHA],
      result.summary[SUMMARY.CRITICAL_T],
      result.summary[SUMMARY.OBSERVATION_COUNT],
      result.summary[SUMMARY.SINGULAR_ROW_COUNT]
    ],
    [
      reference.enp,
      reference.sigma2,
      reference.aicc,
      reference.llf,
      reference.r2,
      reference.adjAlpha05,
      reference.criticalT,
      reference.n,
      0
    ],
    tolerance,
    1,
    `${label} summary`
  );
}

const planar = createNonstationarityScene(12, 77);

it('GPUGeographicallyWeightedRegression reports mgwr local standard errors, t values and significance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // float32 against float64 on a near-saturated local model (ENP 59 of 144): 2e-3 on scalars.
  const fixed = await runFit(
    device,
    planar,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [2.2]},
    {ladderCapacity: 4}
  );
  expectMatchesMgwr(fixed, MGWR_INFERENCE_REFERENCE.fixedBisquare22, 'fixed 2.2', 2e-3, 5e-3);
  expect(fixed.localStatus.every(status => status === STATUS.OK)).toBe(true);
  // A single value is an unselected, unrefined bandwidth.
  expect(fixed.selectedBandwidth[0]).toBe(0);
  expect(fixed.selectedBandwidth[1]).toBeCloseTo(2.2, 5);

  const adaptive = await runFit(
    device,
    planar,
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [20]},
    {ladderCapacity: 4}
  );
  expectMatchesMgwr(
    adaptive,
    MGWR_INFERENCE_REFERENCE.adaptiveBisquare20,
    'adaptive 20',
    2e-3,
    1e-2
  );

  // The significance level is a per-frame setting: 0.1 is mgwr adj_alpha[0].
  const loose = await runFit(
    device,
    planar,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [2.2], significanceLevel: 0.1},
    {ladderCapacity: 4}
  );
  expectClose(
    [loose.summary[SUMMARY.ADJUSTED_ALPHA]],
    [MGWR_INFERENCE_REFERENCE.adjustedAlpha10],
    2e-3,
    1e-9,
    'adjusted alpha at 0.1'
  );
  expect(loose.summary[SUMMARY.CRITICAL_T]).toBeLessThan(fixed.summary[SUMMARY.CRITICAL_T]);
  device.destroy();
});

it('GPUGeographicallyWeightedRegression matches the float64 oracle for inference on every row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'gaussian',
    bandwidthMode: 'fixed',
    bandwidths: [1.2, 1.8, 2.6]
  };
  const result = await runFit(device, planar, settings, {ladderCapacity: 4, refinement: false});
  const oracle = computeGeographicallyWeightedRegressionOnCPU({
    ...planar,
    settings,
    maximumBandwidthCount: 4,
    maximumNeighborCount: 64
  });
  expect(result.selectedBandwidth[0]).toBe(oracle.selectedIndex);
  expectClose(result.coefficients, oracle.coefficients, 5e-3, 0.1, 'beta');
  expectClose(result.standardErrors, oracle.standardErrors, 5e-3, 0.05, 'bse');
  expectClose(result.tValues, oracle.tValues, 5e-3, 1, 't');
  expectClose(
    [
      result.summary[SUMMARY.SIGMA_SQUARED],
      result.summary[SUMMARY.ADJUSTED_ALPHA],
      result.summary[SUMMARY.CRITICAL_T],
      result.summary[SUMMARY.LOG_LIKELIHOOD]
    ],
    [oracle.sigmaSquared, oracle.adjustedAlpha, oracle.criticalT, oracle.logLikelihood],
    2e-3,
    1,
    'summary'
  );
  device.destroy();
});

it('GPUGeographicallyWeightedRegression turns singular locations into NaN rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createSingularRowScene(12, 5, 4, 3.5, 30);
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    bandwidths: [4.5, 5, 6, 7, 8]
  };
  const options = {ladderCapacity: 8, refinement: false} as const;
  const oracleOf = (bandwidths: number[]) =>
    computeGeographicallyWeightedRegressionOnCPU({
      ...scene,
      settings: {...settings, bandwidths},
      maximumBandwidthCount: 8,
      maximumNeighborCount: 64
    });

  // Every candidate keeps its AICc over its non-singular rows and reports its singular rows.
  const wide = await runFit(device, scene, settings, options);
  const wideOracle = oracleOf([4.5, 5, 6, 7, 8]);
  expect(Array.from(wide.singularRowCounts.slice(0, 5))).toEqual([32, 13, 2, 0, 0]);
  expect(Array.from(wide.singularRowCounts.slice(5))).toEqual([0, 0, 0]);
  expectClose(wide.bandwidthScores.slice(0, 5), wideOracle.bandwidthScores, 2e-3, 1, 'scores');
  // The most singular candidate has the smallest AICc but loses to the first one with no
  // singular rows (index 3 and 4 tie on zero; 8 has the smaller AICc).
  expect(wide.bandwidthScores[0]).toBeLessThan(wide.bandwidthScores[4]);
  expect(wide.selectedBandwidth[0]).toBe(4);
  expect(wide.summary[SUMMARY.SINGULAR_ROW_COUNT]).toBe(0);
  expect(wide.localStatus.every(status => status === STATUS.OK)).toBe(true);

  // All candidates singular somewhere: the fewest singular rows wins, not the lowest AICc.
  const narrowSettings = {...settings, bandwidths: [4.5, 5, 6]};
  const narrow = await runFit(device, scene, narrowSettings, options);
  const narrowOracle = oracleOf([4.5, 5, 6]);
  expect(narrow.selectedBandwidth[0]).toBe(2);
  expect(narrow.summary[SUMMARY.SINGULAR_ROW_COUNT]).toBe(2);
  expect(narrow.summary[SUMMARY.OBSERVATION_COUNT]).toBe(142);
  expect(narrow.summary[SUMMARY.INCLUDED_ROW_COUNT]).toBe(144);
  expect(narrow.localStatus).toEqual(narrowOracle.localStatus);
  const singularRows = narrow.localStatus.flatMap((status, row) =>
    status === STATUS.SINGULAR ? [row] : []
  );
  expect(singularRows).toHaveLength(2);
  for (const row of singularRows) {
    for (const values of [narrow.coefficients, narrow.standardErrors, narrow.tValues]) {
      expect(values.slice(row * 3, row * 3 + 3).every(Number.isNaN)).toBe(true);
    }
    expect(Number.isNaN(narrow.fitted[row])).toBe(true);
    expect(Number.isNaN(narrow.residuals[row])).toBe(true);
    expect(Number.isNaN(narrow.hatDiagonal[row])).toBe(true);
    expect(Array.from(narrow.significant.slice(row * 3, row * 3 + 3))).toEqual([0, 0, 0]);
  }
  expectClose(narrow.coefficients, narrowOracle.coefficients, 5e-3, 0.1, 'beta');
  expectClose(narrow.standardErrors, narrowOracle.standardErrors, 5e-3, 0.05, 'bse');
  expectClose(
    [
      narrow.summary[SUMMARY.AICC],
      narrow.summary[SUMMARY.R_SQUARED],
      narrow.summary[SUMMARY.TRACE_OF_HAT]
    ],
    [narrowOracle.aicc, narrowOracle.rSquared, narrowOracle.traceOfHat],
    2e-3,
    1,
    'summary'
  );

  // Invalid bandwidths stay NaN candidates instead of singular ones.
  const invalid = await runFit(
    device,
    scene,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [-1, 0, 8]},
    options
  );
  expect(Number.isNaN(invalid.bandwidthScores[0])).toBe(true);
  expect(Number.isNaN(invalid.bandwidthScores[1])).toBe(true);
  expect(invalid.selectedBandwidth[0]).toBe(2);
  device.destroy();
});

it('GPUGeographicallyWeightedRegression refinement reaches the mgwr optimum', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const adaptive = MGWR_BANDWIDTH_SEARCH_REFERENCE.adaptive;
  const curve = (k: number) => adaptive.aicc[adaptive.ks.indexOf(k)];
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [10, 20, 40, 80, 130]
  };
  // No refinement: the best of the caller ladder is k = 20.
  const coarse = await runFit(device, planar, settings, {
    ladderCapacity: 32,
    refinement: false,
    maximumNeighborCount: 128
  });
  expect(coarse.selectedBandwidth[1]).toBe(20);
  expect(coarse.candidateTable.length).toBe(32 * TABLE_STRIDE);

  // One pass over a 32-slot ladder tries every integer in (10, 40) and finds mgwr's optimum.
  const refined = await runFit(device, planar, settings, {
    ladderCapacity: 32,
    refinement: 1,
    maximumNeighborCount: 128
  });
  expect(refined.selectedBandwidth[1]).toBe(adaptive.argminK);
  expect(refined.selectedBandwidth[0]).toBeGreaterThanOrEqual(32);
  expect(refined.candidateTable.length).toBe(2 * 32 * TABLE_STRIDE);
  const selectedRow = refined.selectedBandwidth[0] * TABLE_STRIDE;
  expect(refined.candidateTable[selectedRow + TABLE.VALUE]).toBe(adaptive.argminK);
  expectClose(
    [refined.summary[SUMMARY.AICC], refined.candidateTable[selectedRow + TABLE.AICC]],
    [curve(adaptive.argminK), curve(adaptive.argminK)],
    1e-3,
    1,
    'AICc at the optimum'
  );
  expect(refined.summary[SUMMARY.AICC]).toBeLessThan(coarse.summary[SUMMARY.AICC]);
  // mgwr with its default bounds [46, 144] stops at its lower bound; with the bounds widened its
  // golden section agrees with the exhaustive optimum, as does the refinement.
  expect(adaptive.goldenBounds6to144).toBe(refined.selectedBandwidth[1]);
  // Every refined candidate carries the mgwr AICc of its k.
  for (let slot = 32; slot < 64; slot++) {
    const row = slot * TABLE_STRIDE;
    const value = refined.candidateTable[row + TABLE.VALUE];
    if (Number.isFinite(value) && value >= 10) {
      expectClose(
        [refined.candidateTable[row + TABLE.AICC]],
        [curve(value)],
        2e-3,
        1,
        `AICc at k = ${value}`
      );
    }
  }

  // A narrow ladder samples the bracket; further passes zoom in on the same optimum, as the
  // float64 oracle does.
  const narrow = await runFit(device, planar, settings, {
    ladderCapacity: 8,
    refinement: 3,
    maximumNeighborCount: 128
  });
  expect(narrow.selectedBandwidth[1]).toBe(adaptive.argminK);
  const oracle = computeGeographicallyWeightedRegressionOnCPU({
    ...planar,
    settings,
    maximumBandwidthCount: 8,
    refinementPassCount: 3,
    maximumNeighborCount: 128
  });
  expect(oracle.selectedValue).toBe(adaptive.argminK);
  expect(narrow.selectedBandwidth[0]).toBe(oracle.selectedIndex);

  // Fixed distances: the zoom lands next to mgwr's optimum with an AICc at least as good.
  const fixedSettings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'fixed',
    bandwidths: [1.5, 2, 3, 4, 6]
  };
  const fixedCoarse = await runFit(device, planar, fixedSettings, {
    ladderCapacity: 8,
    refinement: false
  });
  const fixedRefined = await runFit(device, planar, fixedSettings, {ladderCapacity: 8});
  const fixed = MGWR_BANDWIDTH_SEARCH_REFERENCE.fixed;
  expect(fixedCoarse.selectedBandwidth[1]).toBeCloseTo(3, 5);
  expect(fixedRefined.summary[SUMMARY.AICC]).toBeLessThan(fixedCoarse.summary[SUMMARY.AICC]);
  expect(Math.abs(fixedRefined.selectedBandwidth[1] - fixed.goldenBounds1to6)).toBeLessThan(0.05);
  expect(fixedRefined.summary[SUMMARY.AICC]).toBeLessThanOrEqual(fixed.goldenAicc + 0.05);
  device.destroy();
});

it('GPUGeographicallyWeightedRegression great-circle distances match mgwr spherical', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const geographic = createGeographicScene(planar);
  // mgwr's haversine radius is 6371.0 km, so metres and a 6,371,000 m sphere give the same fit.
  const spatialContext: GPUSpatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    sphereRadius: 6371000
  };
  const fixed = await runFit(
    device,
    geographic,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [25000]},
    {ladderCapacity: 4, spatialContext}
  );
  expectMatchesMgwr(
    fixed,
    MGWR_SPHERICAL_REFERENCE.fixedBisquare25km,
    'spherical fixed',
    2e-3,
    5e-3
  );
  const adaptive = await runFit(
    device,
    geographic,
    {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [20]},
    {ladderCapacity: 4, spatialContext}
  );
  expectMatchesMgwr(
    adaptive,
    MGWR_SPHERICAL_REFERENCE.adaptiveBisquare20,
    'spherical adaptive',
    2e-3,
    1e-2
  );

  // The default Earth radius is the IUGG mean radius, 6,371,008.8 m.
  const earth = await runFit(
    device,
    geographic,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [25000]},
    {
      ladderCapacity: 4,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      }
    }
  );
  expectClose(
    [earth.summary[SUMMARY.AICC]],
    [fixed.summary[SUMMARY.AICC]],
    5e-3,
    1,
    'default radius AICc'
  );
  device.destroy();
});

it('GPUGeographicallyWeightedRegression model result follows mgwr aicc, aic, llf, R2, sigma2 and ENP', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runFit(
    device,
    planar,
    {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [2.2]},
    {ladderCapacity: 4}
  );
  const model = getGPUGeographicallyWeightedRegressionModelResult({
    summary: result.summary,
    selectedBandwidth: result.selectedBandwidth
  });
  const reference = MGWR_INFERENCE_REFERENCE.fixedBisquare22;
  expect(model.kind).toBe('geographically-weighted-regression');
  expect(model.status.ok).toBe(true);
  expectClose(
    [
      model.parameterCount,
      model.sigmaSquared,
      model.logLikelihood!,
      model.akaikeInformationCriterion!,
      model.schwarzCriterion!,
      model.correctedAkaikeInformationCriterion!,
      model.rSquared,
      model.adjustedAlpha,
      model.criticalT
    ],
    [
      reference.enp,
      reference.sigma2,
      reference.llf,
      reference.aic,
      reference.bic,
      reference.aicc,
      reference.r2,
      reference.adjAlpha05,
      reference.criticalT
    ],
    2e-3,
    1,
    'model result'
  );
  expect(model.rowCount).toBe(reference.n);
  expect(model.singularRowCount).toBe(0);
  device.destroy();
});

it('GPUGeographicallyWeightedRegression fits eleven predictors (twelve design columns)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let state = 424242;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const side = 20;
  const rows = side * side;
  const predictorCount = 11;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows * predictorCount);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    positions[2 * row] = (row % side) + random() * 0.5;
    positions[2 * row + 1] = Math.floor(row / side) + random() * 0.5;
    let value = 1;
    for (let column = 0; column < predictorCount; column++) {
      const x = random() * 4 - 2;
      predictors[row * predictorCount + column] = x;
      value += (0.2 + 0.1 * column) * x;
    }
    response[row] = value + 0.3 * (random() - 0.5);
  }
  const scene = {positions, predictors, predictorCount, response};
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'gaussian',
    bandwidthMode: 'fixed',
    bandwidths: [6, 9]
  };
  const result = await runFit(device, scene, settings, {ladderCapacity: 4, refinement: false});
  const oracle = computeGeographicallyWeightedRegressionOnCPU({
    ...scene,
    settings,
    maximumBandwidthCount: 4,
    maximumNeighborCount: 64
  });
  expect(oracle.localStatus.every(status => status === STATUS.OK)).toBe(true);
  expect(result.localStatus).toEqual(oracle.localStatus);
  expect(result.selectedBandwidth[0]).toBe(oracle.selectedIndex);
  expectClose(result.coefficients, oracle.coefficients, 1e-2, 0.2, 'beta');
  expectClose(result.standardErrors, oracle.standardErrors, 2e-2, 0.02, 'bse');
  device.destroy();
});

it('GPUGeographicallyWeightedRegression refinement is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const settings: GPUGeographicallyWeightedRegressionSettings = {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidths: [15, 30, 60]
  };
  const first = await runFit(device, planar, settings, {ladderCapacity: 8});
  const second = await runFit(device, planar, settings, {ladderCapacity: 8});
  for (const key of Object.keys(first) as (keyof Result)[]) {
    const left = new Float32Array(first[key] as number[]);
    const right = new Float32Array(second[key] as number[]);
    expect(new Uint32Array(left.buffer), key).toEqual(new Uint32Array(right.buffer));
  }
  device.destroy();
});
