// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGeographicallyWeightedRegression,
  GPUGeographicallyWeightedRegressionNonstationarityTest,
  GPUOrdinaryLeastSquares,
  GPUSpatialErrorGM,
  GPUSpatialErrorMaximumLikelihood,
  GPUSpatialLagImpacts,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialRegressionDiagnostics,
  GPUSpatialTwoStageLeastSquares,
  GPUSpatialWeightsTraces,
  getGPUGeographicallyWeightedRegressionParameterLength,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER,
  GPU_GWR_NONSTATIONARITY_MAXIMUM_PREDICTOR_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-regression';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

type Bufferish = {byteLength?: number; length?: number; rowByteLength?: number};

function getResourceBytes(buffer: Bufferish): number {
  return buffer.byteLength ?? (buffer.length ?? 0) * (buffer.rowByteLength ?? 4);
}

/** Largest bound or transient resource of the nodes, in bytes. */
function getLargestResourceBytes(nodes: readonly unknown[]): number {
  let largest = 0;
  for (const node of nodes as {resources?: {buffer: Bufferish}[]}[]) {
    for (const resource of node.resources ?? []) {
      largest = Math.max(largest, getResourceBytes(resource.buffer));
    }
  }
  return largest;
}

function expectKernelsWithinCoreLimits(nodes: readonly unknown[]): void {
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
}

const ROWS = 300;
const POWER = GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER;

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  type Format = 'uint32' | 'float32' | 'float32x2' | 'sint32';
  const view = <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `spatial-regression-${serial++}`, format, length);
  const weights = (rows: number = ROWS) => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows * 8),
    weights: view('float32', rows * 8)
  });
  return {device, graph, view, weights};
}

function expectNodesWithinCoreLimits(
  create: (context: ReturnType<typeof createContext>) => readonly unknown[]
): void {
  const context = createContext();
  const nodes = create(context);
  expectKernelsWithinCoreLimits(nodes);
  context.device.destroy();
}

it('GPUOrdinaryLeastSquares at maximal predictors with every output fits the core limits', () => {
  const k = 15;
  expectNodesWithinCoreLimits(({graph, view}) =>
    new GPUOrdinaryLeastSquares({
      predictors: view('float32', ROWS * k),
      response: view('float32', ROWS),
      mask: view('uint32', ROWS),
      parameters: view('float32', GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH),
      predictorCount: k,
      tileRowCount: 64,
      output: {
        coefficients: view('float32', k + 1),
        standardErrors: view('float32', k + 1),
        tStatistics: view('float32', k + 1),
        summary: view('float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
        status: {code: view('uint32', 1)},
        residuals: view('float32', ROWS),
        fitted: view('float32', ROWS)
      }
    }).getCommandNodes(graph)
  );
});

it('GPUSpatialRegressionDiagnostics at maximal predictors with every option fits the core limits', () => {
  for (const restandardizeRows of [false, true]) {
    const k = 15;
    expectNodesWithinCoreLimits(({graph, view, weights}) =>
      new GPUSpatialRegressionDiagnostics({
        weights: weights(),
        predictors: view('float32', ROWS * k),
        response: view('float32', ROWS),
        residuals: view('float32', ROWS),
        mask: view('uint32', ROWS),
        restandardizeRows,
        predictorCount: k,
        tileRowCount: 64,
        output: {
          tests: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH),
          summary: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH),
          status: {code: view('uint32', 1)}
        }
      }).getCommandNodes(graph)
    );
  }
});

for (const instrumentOrder of [1, 2] as const) {
  it(`GPUSpatialTwoStageLeastSquares order ${instrumentOrder} at maximal predictors fits the core limits`, () => {
    const k =
      instrumentOrder === 1 ? GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT : 5;
    expectNodesWithinCoreLimits(({graph, view, weights}) =>
      new GPUSpatialTwoStageLeastSquares({
        weights: weights(),
        predictors: view('float32', ROWS * k),
        response: view('float32', ROWS),
        predictorCount: k,
        tileRowCount: 64,
        instrumentOrder,
        output: {
          table: view('float32', (k + 3) * 4),
          summary: view('float32', GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH),
          status: {code: view('uint32', 1)},
          residuals: view('float32', ROWS)
        }
      }).getCommandNodes(graph)
    );
  });
}

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const) {
  for (const traceMode of ['exact', 'stochastic'] as const) {
    it(`GPUSpatialErrorGM ${estimator}/${traceMode} at maximal predictors fits the core limits`, () => {
      const k = GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT;
      expectNodesWithinCoreLimits(({graph, view, weights}) =>
        new GPUSpatialErrorGM({
          weights: weights(),
          predictors: view('float32', ROWS * k),
          response: view('float32', ROWS),
          predictorCount: k,
          estimator,
          traceMode,
          traceProbeCount: 64,
          traceParameters: view('uint32', 1),
          tileRowCount: 64,
          output: {
            table: view('float32', (k + 3) * 4),
            summary: view('float32', GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH),
            status: {code: view('uint32', 1)},
            residuals: view('float32', ROWS)
          }
        }).getCommandNodes(graph)
      );
    });
  }
}

for (const traceMode of ['exact', 'stochastic'] as const) {
  it(`GPUSpatialLagMaximumLikelihood ${traceMode} at maximal predictors and power fits the core limits`, () => {
    const k = GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT;
    expectNodesWithinCoreLimits(({graph, view, weights}) =>
      new GPUSpatialLagMaximumLikelihood({
        weights: weights(),
        predictors: view('float32', ROWS * k),
        response: view('float32', ROWS),
        predictorCount: k,
        maximumPower: POWER,
        traceMode,
        traceProbeCount: 64,
        traceParameters: view('uint32', 1),
        tileRowCount: 64,
        output: {
          table: view('float32', (k + 3) * 4),
          summary: view('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
          status: {code: view('uint32', 1), converged: view('uint32', 1)},
          residuals: view('float32', ROWS),
          powerTraces: view('float32', POWER),
          powerSums: view('float32', POWER),
          infinityNorm: view('float32', 1)
        }
      }).getCommandNodes(graph)
    );
  });

  it(`GPUSpatialErrorMaximumLikelihood ${traceMode} at maximal predictors and power fits the core limits`, () => {
    const k = GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT;
    expectNodesWithinCoreLimits(({graph, view, weights}) =>
      new GPUSpatialErrorMaximumLikelihood({
        weights: weights(),
        predictors: view('float32', ROWS * k),
        response: view('float32', ROWS),
        predictorCount: k,
        maximumPower: POWER,
        traceMode,
        traceProbeCount: 64,
        traceParameters: view('uint32', 1),
        tileRowCount: 64,
        output: {
          table: view('float32', (k + 3) * 4),
          summary: view('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
          status: {code: view('uint32', 1), converged: view('uint32', 1)},
          residuals: view('float32', ROWS)
        }
      }).getCommandNodes(graph)
    );
  });
}

it('GPUSpatialLagImpacts at maximal power and predictors fits the core limits', () => {
  const k = 14;
  expectNodesWithinCoreLimits(({graph, view}) =>
    new GPUSpatialLagImpacts({
      coefficients: view('float32', (k + 2) * 4),
      coefficientStride: 4,
      rhoIndex: k + 1,
      firstPredictorIndex: 0,
      predictorCount: k,
      rowCount: ROWS,
      maximumPower: POWER,
      traces: {
        powerTraces: view('float32', POWER),
        powerSums: view('float32', POWER),
        infinityNorm: view('float32', 1)
      },
      status: view('uint32', 1),
      rhoBound: 0.99,
      output: {impacts: view('float32', k * 3)}
    }).getCommandNodes(graph)
  );
});

for (const mode of ['exact', 'stochastic'] as const) {
  for (const includeGramTraces of [false, true]) {
    it(`GPUSpatialWeightsTraces ${mode} gram ${includeGramTraces} at maximal power fits the core limits`, () => {
      expectNodesWithinCoreLimits(({graph, view, weights}) =>
        new GPUSpatialWeightsTraces({
          weights: weights(),
          maximumPower: POWER,
          mode,
          probeCount: 64,
          parameters: view('uint32', 1),
          includeGramTraces,
          tileRowCount: 64,
          probeBatchSize: 16,
          output: {
            powerTraces: view('float32', POWER),
            powerSums: view('float32', POWER),
            gramTraces: includeGramTraces ? view('float32', 4) : undefined,
            infinityNorm: view('float32', 1),
            status: {approximation: view('uint32', 1)}
          }
        }).getCommandNodes(graph)
      );
    });
  }
}

function createGWR(
  context: ReturnType<typeof createContext>,
  rows: number,
  options: {
    greatCircle?: boolean;
    ladderLength?: number;
    refinement?: boolean | number;
    indexGridSize?: readonly [number, number] | false;
  } = {}
) {
  const {view} = context;
  const k = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT;
  const ladder =
    options.ladderLength ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH;
  const refinement =
    options.refinement ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT;
  const passes = typeof refinement === 'number' ? refinement : refinement ? 1 : 0;
  const coefficientCount = k + 1;
  return new GPUGeographicallyWeightedRegression({
    positions: view('float32x2', rows),
    predictors: view('float32', rows * k),
    predictorCount: k,
    response: view('float32', rows),
    mask: view('uint32', rows),
    parameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength(ladder)),
    maximumBandwidthCount: ladder,
    maximumNeighborCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
    refinement,
    indexGridSize: options.indexGridSize,
    spatialContext: options.greatCircle
      ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      : undefined,
    output: {
      coefficients: view('float32', rows * coefficientCount),
      standardErrors: view('float32', rows * coefficientCount),
      tValues: view('float32', rows * coefficientCount),
      significant: view('uint32', rows * coefficientCount),
      localR2: view('float32', rows),
      fitted: view('float32', rows),
      residuals: view('float32', rows),
      hatDiagonal: view('float32', rows),
      localStatus: view('uint32', rows),
      localConditionNumber: view('float32', rows),
      bandwidthScores: view('float32', ladder),
      bandwidthSingularRowCounts: view('uint32', ladder),
      candidateTable: view(
        'float32',
        (1 + passes) * ladder * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE
      ),
      selectedBandwidth: view('float32', 2),
      summary: view('float32', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
    }
  });
}

for (const greatCircle of [false, true]) {
  it(`GPUGeographicallyWeightedRegression ${greatCircle ? 'great-circle' : 'planar'} at maximal options fits the core limits`, () => {
    expectNodesWithinCoreLimits(context =>
      createGWR(context, ROWS, {greatCircle}).getCommandNodes(context.graph)
    );
  });
}

it('GPUGeographicallyWeightedRegression without index grid or refinement fits the core limits', () => {
  expectNodesWithinCoreLimits(context =>
    createGWR(context, ROWS, {refinement: false, indexGridSize: false}).getCommandNodes(
      context.graph
    )
  );
});

// The constructor bounds rows * maximumBandwidthCount * 8 bytes by the 128 MiB binding size.
for (const [rows, ladderLength] of [
  [GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT, 16],
  [(1 << 24) / GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH, 32]
]) {
  it(`GPUGeographicallyWeightedRegression at ${rows} rows and ${ladderLength} bandwidths binds within maxStorageBufferBindingSize`, () => {
    const context = createContext();
    const nodes = createGWR(context, rows, {ladderLength}).getCommandNodes(context.graph);
    expect(getLargestResourceBytes(nodes)).toBeGreaterThanOrEqual(rows * 12 * 4);
    expect(getLargestResourceBytes(nodes)).toBeLessThanOrEqual(
      context.device.limits.maxStorageBufferBindingSize
    );
    context.device.destroy();
  });
}

it('GPUGeographicallyWeightedRegressionNonstationarityTest at maximal options fits the core limits', () => {
  const k = GPU_GWR_NONSTATIONARITY_MAXIMUM_PREDICTOR_COUNT;
  {
    expectNodesWithinCoreLimits(({graph, view}) =>
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        positions: view('float32x2', ROWS),
        predictors: view('float32', ROWS * k),
        predictorCount: k,
        response: view('float32', ROWS),
        mask: view('uint32', ROWS),
        bandwidthParameters: view(
          'float32',
          getGPUGeographicallyWeightedRegressionParameterLength()
        ),
        selectedBandwidth: view('float32', 2),
        coefficients: view('float32', ROWS * (k + 1)),
        parameters: view('uint32', 4),
        maximumPermutations: 999,
        maximumBandwidthCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
        maximumNeighborCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
        tileRowCount: 64,
        output: {
          table: view('float32', (k + 1) * 8),
          summary: view('float32', 16),
          standardDeviations: view('float32', 1000 * (k + 1))
        }
      }).getCommandNodes(graph)
    );
  }
});
