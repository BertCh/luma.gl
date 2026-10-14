// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import type {GPUSpatialErrorMaximumLikelihoodProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagImpacts} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import type {GPUSpatialLagMaximumLikelihoodProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {
  GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-parameters';
import {createSeriesProbeNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
const ROWS = 10;

function createFixtures() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const createProps = (predictorCount = 2) => ({
    weights: {
      offsets: view('uint32', ROWS + 1),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    },
    predictors: view('float32', ROWS * predictorCount),
    response: view('float32', ROWS),
    predictorCount,
    output: {
      table: view('float32', (predictorCount + 2) * 4),
      summary: view('float32', 22),
      status: {code: view('uint32', 1)}
    }
  });
  return {graph, view, createProps};
}

it('GPUSpatialLagMaximumLikelihood validates its inputs and declares its nodes', () => {
  const {graph, view, createProps} = createFixtures();
  const create = (overrides: Partial<GPUSpatialLagMaximumLikelihoodProps> = {}) =>
    new GPUSpatialLagMaximumLikelihood({...createProps(), ...overrides});
  const expectThrows = (overrides: Partial<GPUSpatialLagMaximumLikelihoodProps>, message: RegExp) =>
    expect(() => create(overrides)).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 15}, /predictorCount/);
  expectThrows({maximumPower: 1}, /maximumPower/);
  expectThrows({maximumPower: 257}, /maximumPower/);
  expectThrows({maximumPower: 2.5}, /maximumPower/);
  expectThrows({traceMode: 'dense' as 'exact'}, /traceMode/);
  expectThrows({traceProbeCount: 0}, /traceProbeCount/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows({response: view('float32', ROWS - 1)}, /response length/);
  expectThrows({predictors: view('float32', ROWS * 2 - 1)}, /predictors length/);
  const output = createProps().output;
  expectThrows({output: {...output, table: view('float32', 15)}}, /output.table/);
  expectThrows({output: {...output, summary: view('float32', 21)}}, /output.summary/);
  expectThrows(
    {output: {...output, status: {...output.status, converged: view('uint32', 0)}}},
    /output\.status\.converged/
  );
  expectThrows({output: {...output, residuals: view('float32', ROWS - 1)}}, /output.residuals/);
  expectThrows({output: {...output, powerTraces: view('float32', 128)}}, /provided together/);
  expectThrows(
    {
      output: {
        ...output,
        powerTraces: view('float32', 3),
        powerSums: view('float32', 128),
        infinityNorm: view('float32', 1)
      }
    },
    /powerTraces/
  );
  const lag = create({id: 'lag', maximumPower: 4});
  expect(lag.traceMode).toBe('exact');
  expect(lag.maximumPower).toBe(4);
  const names = lag.getCommandNodes(graph).map(node => node.id.replace('lag-', ''));
  expect(names.filter(name => !name.startsWith('ols-') && !name.startsWith('traces-'))).toEqual([
    'lagged-response-tiles',
    'means-tiles',
    'means-merge',
    'moments-tiles',
    'rho-search',
    'solve-coefficients',
    'residuals',
    'inverse-series-1',
    'inverse-series-2',
    'inverse-series-3',
    'weighted-prediction',
    'final-moments-tiles',
    'final-merge',
    'tr3-batch-information',
    'tr3-probe-first-0-1',
    'tr3-probe-step-0-2',
    'tr3-probe-step-0-3',
    'tr3-probe-final-0-4',
    'tr3-probe-merge-0',
    'variance',
    'finish'
  ]);
  expect(names.filter(name => name.startsWith('ols-response-')).length).toBeGreaterThan(0);
  expect(names.filter(name => name.startsWith('ols-lagged-')).length).toBeGreaterThan(0);
  expect(names.some(name => name.startsWith('traces-'))).toBe(true);
});

it('GPUSpatialLagMaximumLikelihood accepts its predictor cap and no kernel exceeds 8 bindings', () => {
  const {graph, createProps} = createFixtures();
  const cap = GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT;
  expect(cap).toBe(14);
  // Declaring the nodes at the cap throws if any kernel binds more than the device's 8 buffers.
  expect(() =>
    new GPUSpatialLagMaximumLikelihood({
      ...createProps(cap),
      maximumPower: 3,
      output: {
        ...createProps(cap).output,
        status: {
          ...createProps(cap).output.status,
          converged: createTransientView(graph, `converged-${serial++}`, 'uint32', 1)
        },
        residuals: createTransientView(graph, `residuals-${serial++}`, 'float32', ROWS)
      }
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('GPUSpatialLagMaximumLikelihood feeds GPUSpatialLagImpacts through its table and traces', () => {
  const {graph, view, createProps} = createFixtures();
  const k = 2;
  const props = createProps(k);
  const powerTraces = view('float32', 16);
  const powerSums = view('float32', 16);
  const infinityNorm = view('float32', 1);
  const model = new GPUSpatialLagMaximumLikelihood({
    ...props,
    maximumPower: 16,
    output: {...props.output, powerTraces, powerSums, infinityNorm}
  });
  expect(() => model.getCommandNodes(graph)).not.toThrow();
  // rho is the last table row (index predictorCount + 1), the stride is 4.
  expect(
    () =>
      new GPUSpatialLagImpacts({
        coefficients: props.output.table,
        coefficientStride: 4,
        rhoIndex: k + 1,
        predictorCount: k,
        rowCount: ROWS,
        maximumPower: 16,
        traces: {powerTraces, powerSums, infinityNorm},
        status: props.output.status.code,
        output: {impacts: view('float32', k * 3)}
      })
  ).not.toThrow();
});

it('GPUSpatialErrorMaximumLikelihood validates its inputs and declares its nodes', () => {
  const {graph, view, createProps} = createFixtures();
  const create = (overrides: Partial<GPUSpatialErrorMaximumLikelihoodProps> = {}) =>
    new GPUSpatialErrorMaximumLikelihood({...createProps(), ...overrides});
  const expectThrows = (
    overrides: Partial<GPUSpatialErrorMaximumLikelihoodProps>,
    message: RegExp
  ) => expect(() => create(overrides)).toThrow(message);
  expectThrows({predictorCount: 0}, /predictorCount/);
  expectThrows({predictorCount: 16}, /predictorCount/);
  expectThrows({maximumPower: 1}, /maximumPower/);
  expectThrows({traceMode: 'dense' as 'exact'}, /traceMode/);
  expectThrows({tileRowCount: 0}, /tileRowCount/);
  expectThrows({response: view('float32', ROWS - 1)}, /response length/);
  expectThrows({predictors: view('float32', ROWS * 2 - 1)}, /predictors length/);
  const output = createProps().output;
  expectThrows({output: {...output, table: view('float32', 15)}}, /output.table/);
  expectThrows({output: {...output, summary: view('float32', 21)}}, /output.summary/);
  expectThrows({output: {...output, residuals: view('float32', ROWS - 1)}}, /output.residuals/);
  const error = create({id: 'err', maximumPower: 4});
  const names = error.getCommandNodes(graph).map(node => node.id.replace('err-', ''));
  expect(names.filter(name => !name.startsWith('traces-'))).toEqual([
    'weights-norm-tiles',
    'means-tiles',
    'means-merge',
    'gram-tiles-0',
    'gram-tiles-1',
    'gram-tiles-2',
    'gram-merge',
    'lambda-search',
    'filtered-solve',
    'residuals-tiles',
    'residuals-merge',
    'tr3-batch-information',
    'tr3-probe-first-0-1',
    'tr3-probe-step-0-2',
    'tr3-probe-step-0-3',
    'tr3-probe-final-0-4',
    'tr3-probe-merge-0',
    'variance',
    'finish'
  ]);
  const cap = GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT;
  expect(cap).toBe(15);
  expect(() =>
    new GPUSpatialErrorMaximumLikelihood({...createProps(cap), maximumPower: 3}).getCommandNodes(
      graph
    )
  ).not.toThrow();
});

it('the maximum-likelihood nodes pick stochastic traces above 4096 rows and batch the probe pass', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = 5000;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `big-${serial++}`, format, length);
  const model = new GPUSpatialLagMaximumLikelihood({
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows * 4),
      weights: view('float32', rows * 4)
    },
    predictors: view('float32', rows * 2),
    response: view('float32', rows),
    predictorCount: 2,
    maximumPower: 3,
    output: {
      table: view('float32', 16),
      summary: view('float32', 22),
      status: {code: view('uint32', 1)}
    }
  });
  expect(model.traceMode).toBe('stochastic');
  const names = model.getCommandNodes(graph).map(node => node.id);
  // 64 Rademacher probes fit one batch (64 * 5000 <= 8M elements).
  expect(names.filter(name => /tr3-probe-merge/.test(name))).toHaveLength(1);
  expect(
    () =>
      new GPUSpatialLagMaximumLikelihood({
        weights: {
          offsets: view('uint32', 16386),
          neighbors: view('uint32', 4),
          weights: view('float32', 4)
        },
        predictors: view('float32', 32770),
        response: view('float32', 16385),
        predictorCount: 2,
        traceMode: 'exact',
        output: {
          table: view('float32', 16),
          summary: view('float32', 22),
          status: {code: view('uint32', 1)}
        }
      })
  ).toThrow(/16384/);
});

// Rule 12: neither the series length nor the number of probe batches may reach WGSL text or
// override constants, so building at N and N + 1 must give the same number of pipelines.
const getSeriesProbePipelineCount = (
  mode: 'exact' | 'stochastic',
  maximumPower: number,
  rowCount: number
): {pipelines: number; batchCount: number} => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `series-view-${serial++}`, format, length);
  const probes = createSeriesProbeNodes(graph, {
    id: `series-${serial++}`,
    operation: 'SeriesProbeTest',
    weights: {
      offsets: view('uint32', rowCount + 1),
      neighbors: view('uint32', 4),
      weights: view('float32', 4)
    },
    rowCount,
    maximumPower,
    mode,
    probeCount: 64,
    seedParameters: view('uint32', 1),
    tileRowCount: Math.ceil(rowCount / 64),
    tileCount: 64,
    workspace: view('float32', 8),
    parameterSlot: 1
  });
  const pipelines = getGPUCommandNodePipelineKeys(probes.nodes).size;
  const batchCount = probes.nodes.filter(node => /probe-merge-/.test(node.id)).length;
  device.destroy();
  return {pipelines, batchCount};
};

it('the series probe pipelines do not depend on the series length or the number of batches', () => {
  for (const mode of ['exact', 'stochastic'] as const) {
    const base = getSeriesProbePipelineCount(mode, 4, 3000);
    expect(base.pipelines).toBeGreaterThan(0);
    expect(getSeriesProbePipelineCount(mode, 5, 3000).pipelines, `${mode} power`).toBe(
      base.pipelines
    );
    expect(getSeriesProbePipelineCount(mode, 9, 3000).pipelines, `${mode} power`).toBe(
      base.pipelines
    );
    // Exact: probeTotal = rows, so more rows gives more batches. Stochastic: 64 probes in
    // batches of floor(8M / rows).
    const rowCounts = mode === 'exact' ? [3000, 5000, 9000] : [1_000_000, 1_700_000, 2_500_000];
    const counts = rowCounts.map(rowCount => getSeriesProbePipelineCount(mode, 4, rowCount));
    expect(new Set(counts.map(count => count.batchCount)).size, mode).toBeGreaterThan(1);
    expect(counts.every(count => count.batchCount > 1)).toBe(true);
    expect(new Set(counts.map(count => count.pipelines)).size, `${mode} batches`).toBe(1);
  }
});

it('the maximum likelihood contributors build the same pipelines at every series length', () => {
  const getCount = (kind: 'lag' | 'error', maximumPower: number) => {
    const {graph, createProps} = createFixtures();
    const props = {...createProps(), id: `${kind}-${serial++}`, maximumPower};
    const model =
      kind === 'lag'
        ? new GPUSpatialLagMaximumLikelihood(props)
        : new GPUSpatialErrorMaximumLikelihood(props);
    const nodes = model.getCommandNodes(graph);
    const count = getGPUCommandNodePipelineKeys(nodes).size;
    graph.device.destroy();
    return count;
  };
  for (const kind of ['lag', 'error'] as const) {
    expect(getCount(kind, 5), kind).toBe(getCount(kind, 4));
    expect(getCount(kind, 8), kind).toBe(getCount(kind, 4));
  }
});
