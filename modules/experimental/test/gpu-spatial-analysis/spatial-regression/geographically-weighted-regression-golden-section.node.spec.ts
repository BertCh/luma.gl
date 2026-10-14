// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPUGeographicallyWeightedRegression,
  type GPUGeographicallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';
import {createNonstationarityScene} from './geographically-weighted-regression-scenes';
import {
  getDistinctEvaluatedValues,
  getGPUGoldenSectionTolerance,
  getMgwrGoldenSectionTolerance,
  roundHalfToEven,
  simulateGoldenSectionSearch
} from './gwr-golden-section-reference';
import {
  MGWR_BANDWIDTH_SEARCH_REFERENCE,
  MGWR_GOLDEN_SECTION_REFERENCE,
  MGWR_INFERENCE_REFERENCE
} from './gwr-mgwr-inference-reference';

const STRIDE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE;
const ROWS = 10;
let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGeographicallyWeightedRegressionProps> = {}
): GPUGeographicallyWeightedRegressionProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `golden-view-${serial++}`, format, length);
  return {
    positions: view('float32x2', ROWS),
    predictors: view('float32', ROWS * 2),
    predictorCount: 2,
    response: view('float32', ROWS),
    parameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength()),
    bandwidthSearch: 'golden-section',
    output: {coefficients: view('float32', ROWS * 3)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUGeographicallyWeightedRegressionProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(
    () => new GPUGeographicallyWeightedRegression(createProps(graph, overrides(graph)))
  ).toThrow(message);
  device.destroy();
}

it('golden-section and cutoff props are validated', () => {
  expectThrows(() => ({bandwidthSearch: 'scipy' as never}), /bandwidthSearch/);
  expectThrows(() => ({maximumBandwidthCount: 1}), /maximumBandwidthCount >= 2/);
  expectThrows(() => ({maximumNeighborCount: 1}), /maximumNeighborCount >= 2/);
  expectThrows(() => ({maximumSearchIterations: 0}), /maximumSearchIterations/);
  expectThrows(() => ({maximumSearchIterations: 65}), /maximumSearchIterations/);
  expectThrows(() => ({gaussianCutoff: 0}), /gaussianCutoff/);
  expectThrows(() => ({gaussianCutoff: -3}), /gaussianCutoff/);
  expectThrows(() => ({gaussianCutoff: Number.NaN}), /gaussianCutoff/);
  expectThrows(() => ({gaussianCutoff: Number.POSITIVE_INFINITY}), /gaussianCutoff/);
  expectThrows(
    graph => ({
      bandwidthSearch: 'ladder',
      output: {
        coefficients: createTransientView(graph, `c-${serial++}`, 'float32', ROWS * 3),
        searchState: createTransientView(graph, `s-${serial++}`, 'float32', 12)
      }
    }),
    /searchState needs bandwidthSearch/
  );
  expectThrows(
    graph => ({
      maximumSearchIterations: 5,
      output: {
        coefficients: createTransientView(graph, `c-${serial++}`, 'float32', ROWS * 3),
        searchState: createTransientView(graph, `s-${serial++}`, 'float32', 11)
      }
    }),
    /searchState/
  );
  // The golden-section table has maximumSearchIterations + 1 rows, not the ladder layout.
  expectThrows(
    graph => ({
      maximumSearchIterations: 5,
      output: {
        coefficients: createTransientView(graph, `c-${serial++}`, 'float32', ROWS * 3),
        candidateTable: createTransientView(graph, `t-${serial++}`, 'float32', 5 * STRIDE)
      }
    }),
    /candidateTable/
  );
  // Ladder mode ignores the search props and keeps its own table layout.
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ladder = new GPUGeographicallyWeightedRegression(
    createProps(graph, {bandwidthSearch: 'ladder', maximumSearchIterations: 99})
  );
  expect(ladder.maximumSearchIterations).toBe(0);
  expect(ladder.candidateTableRowCount).toBe(3 * 32);
  const search = new GPUGeographicallyWeightedRegression(
    createProps(graph, {maximumSearchIterations: 7, refinement: 4})
  );
  expect(search.refinementPassCount).toBe(0);
  expect(search.maximumSearchIterations).toBe(7);
  expect(search.candidateWidth).toBe(2);
  expect(search.candidateTableRowCount).toBe(8);
  const defaults = new GPUGeographicallyWeightedRegression(createProps(graph));
  expect(defaults.maximumSearchIterations).toBe(30);
  device.destroy();
});

it('golden-section search builds the documented gated node sequence', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const regression = new GPUGeographicallyWeightedRegression(
    createProps(graph, {id: 'g', maximumSearchIterations: 3, indexGridSize: false})
  );
  const nodes = regression.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual([
    'g-validate',
    'g-ladder-init',
    'g-search-bounds-rows',
    'g-search-bounds-tiles',
    'g-search-init',
    'g-candidates',
    'g-tile-candidates',
    'g-select',
    'g-search-1-iterate',
    'g-search-1-candidates',
    'g-search-1-tile-candidates',
    'g-search-1-select',
    'g-search-2-iterate',
    'g-search-2-candidates',
    'g-search-2-tile-candidates',
    'g-search-2-select',
    'g-search-3-iterate',
    'g-choose',
    'g-fit',
    'g-tile-statistics',
    'g-summarize'
  ]);
  const byId = new Map(nodes.map(node => [node.id, node as {condition?: unknown}]));
  // Rule 13: the first pass and every state kernel run always; later candidate passes launch only
  // while the one-thread iterate kernel keeps the shared indirect dispatch triples armed.
  for (const always of ['g-candidates', 'g-tile-candidates', 'g-select']) {
    expect(byId.get(always)?.condition, always).toBeUndefined();
  }
  for (const iteration of [1, 2]) {
    for (const kind of ['candidates', 'tile-candidates', 'select']) {
      expect(byId.get(`g-search-${iteration}-${kind}`)?.condition, kind).toMatchObject({
        source: 'gpu',
        mode: 'indirect'
      });
    }
  }
  for (const state of [
    'g-search-bounds-rows',
    'g-search-bounds-tiles',
    'g-search-init',
    'g-search-1-iterate',
    'g-search-2-iterate',
    'g-search-3-iterate',
    'g-choose'
  ]) {
    expect(byId.get(state)?.condition, state).toBeUndefined();
  }
  // One iteration: a single comparison and no later candidate pass.
  const single = new GPUGeographicallyWeightedRegression(
    createProps(graph, {id: 's', maximumSearchIterations: 1, indexGridSize: false})
  );
  expect(single.getCommandNodes(graph).map(node => node.id)).toEqual([
    's-validate',
    's-ladder-init',
    's-search-bounds-rows',
    's-search-bounds-tiles',
    's-search-init',
    's-candidates',
    's-tile-candidates',
    's-select',
    's-search-1-iterate',
    's-choose',
    's-fit',
    's-tile-statistics',
    's-summarize'
  ]);
  device.destroy();
});

it('golden-section iterations share pipelines and do not depend on the iteration count (rule 12)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const getKeys = (maximumSearchIterations: number, idPattern?: RegExp, indexGridSize?: false) =>
    getGPUCommandNodePipelineKeys(
      new GPUGeographicallyWeightedRegression(
        createProps(graph, {
          id: `rule-12-${maximumSearchIterations}-${serial++}`,
          maximumSearchIterations,
          indexGridSize
        })
      ).getCommandNodes(graph),
      {idPattern}
    );
  for (const indexGridSize of [false, undefined] as const) {
    const counts = [2, 3, 4, 8].map(
      iterations => getKeys(iterations, undefined, indexGridSize).size
    );
    expect(new Set(counts).size, `distinct pipelines for N = 2, 3, 4, 8: ${counts}`).toBe(1);
  }
  for (const iterations of [3, 8]) {
    expect(getKeys(iterations, /(?<!tile)-candidates$/).size).toBe(1);
    expect(getKeys(iterations, /-select$/).size).toBe(1);
    expect(getKeys(iterations, /-iterate$/).size).toBe(1);
  }
  device.destroy();
});

const CONFIGURATIONS: {name: string; props: Partial<GPUGeographicallyWeightedRegressionProps>}[] = [
  {name: 'scan', props: {indexGridSize: false}},
  {name: 'grid', props: {indexGridSize: [4, 4]}},
  {name: 'scan with Gaussian cutoff', props: {indexGridSize: false, gaussianCutoff: 4}},
  {name: 'grid with Gaussian cutoff', props: {indexGridSize: [4, 4], gaussianCutoff: 5.5}},
  {
    name: 'great circle',
    props: {
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      }
    }
  },
  {
    name: 'minimum neighbours and capacity',
    props: {maximumBandwidthCount: 2, maximumNeighborCount: 2}
  }
];

it('golden-section kernels parse as WGSL and respect the binding limit', () => {
  for (const configuration of CONFIGURATIONS) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `golden-wgsl-${serial++}`, format, length);
    const iterations = 4;
    const rows = 40;
    const regression = new GPUGeographicallyWeightedRegression({
      positions: view('float32x2', rows),
      predictors: view('float32', rows * 2),
      predictorCount: 2,
      response: view('float32', rows),
      mask: view('uint32', rows),
      parameters: view(
        'float32',
        getGPUGeographicallyWeightedRegressionParameterLength(
          configuration.props.maximumBandwidthCount ?? 32
        )
      ),
      bandwidthSearch: 'golden-section',
      maximumSearchIterations: iterations,
      output: {
        coefficients: view('float32', rows * 3),
        standardErrors: view('float32', rows * 3),
        tValues: view('float32', rows * 3),
        significant: view('uint32', rows * 3),
        localR2: view('float32', rows),
        fitted: view('float32', rows),
        residuals: view('float32', rows),
        hatDiagonal: view('float32', rows),
        localStatus: view('uint32', rows),
        localConditionNumber: view('float32', rows),
        bandwidthScores: view('float32', configuration.props.maximumBandwidthCount ?? 32),
        bandwidthSingularRowCounts: view('uint32', configuration.props.maximumBandwidthCount ?? 32),
        candidateTable: view('float32', (iterations + 1) * STRIDE),
        selectedBandwidth: view('float32', 2),
        searchState: view('float32', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH),
        summary: view('float32', 12)
      },
      ...configuration.props
    });
    const nodes = regression.getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    device.destroy();
    expect(sources.length, `${configuration.name} kernel count`).toBe(nodes.length);
    for (const {id, source} of sources) {
      let reflection: WgslReflect | undefined;
      expect(() => {
        reflection = new WgslReflect(source);
      }, `${configuration.name}: ${id} parses`).not.toThrow();
      expect(
        reflection!.storage.length,
        `${configuration.name}: ${id} storage bindings`
      ).toBeLessThanOrEqual(8);
    }
    const hasCutoff = configuration.props.gaussianCutoff !== undefined;
    const fitSources = sources.filter(({id}) => /(?<!tile|publish)-candidates$|-fit$/.test(id));
    for (const {id, source} of fitSources) {
      expect(source.includes('const GAUSSIAN_CUTOFF'), `${configuration.name}: ${id} cutoff`).toBe(
        hasCutoff
      );
    }
  }
});

it('the Gaussian cutoff reaches the grid path only as a compile-time constant', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const getFitSource = (props: Partial<GPUGeographicallyWeightedRegressionProps>): string => {
    const nodes = new GPUGeographicallyWeightedRegression(
      createProps(graph, {id: `cutoff-${serial++}`, bandwidthSearch: 'ladder', ...props})
    ).getCommandNodes(graph);
    return getGPUCommandNodeKernelSources(nodes).find(({id}) => id.endsWith('-fit'))!.source;
  };
  const exact = getFitSource({indexGridSize: [4, 4]});
  expect(exact).toContain('return getIsBisquare() && bandwidth > 0.0');
  expect(exact).not.toContain('GAUSSIAN_CUTOFF');
  const truncated = getFitSource({indexGridSize: [4, 4], gaussianCutoff: 5});
  expect(truncated).toContain('const GAUSSIAN_CUTOFF: f32 = 5.0;');
  expect(truncated).toContain('if (distance > GAUSSIAN_CUTOFF * bandwidth)');
  expect(truncated).toContain(
    'return select(bandwidth * GAUSSIAN_CUTOFF, bandwidth, getIsBisquare());'
  );
  expect(truncated).toContain('getGridRange(origin, getSupportRadius(bandwidth))');
  expect(truncated).not.toContain('return getIsBisquare() && bandwidth > 0.0');
  // Without a grid the cutoff only truncates the weights.
  const scan = getFitSource({indexGridSize: false, gaussianCutoff: 3});
  expect(scan).toContain('const GAUSSIAN_CUTOFF: f32 = 3.0;');
  expect(scan).not.toContain('getSupportRadius');
  device.destroy();
});

const planar = createNonstationarityScene(12, 77);

/**
 * Score lookup of the bandwidths mgwr evaluated. Keys match within 1e-13 relative (a few ulp): the
 * bundler's JSON import can move a 17-digit fixture number by one ulp, and float32 state moves
 * fixed bandwidths by 6e-8, which `maximumRelativeDistance` widens for the float32 mirror.
 */
function getEvaluator(
  evaluated: [number, number][],
  maximumRelativeDistance = 1e-13
): (value: number) => {score: number; singularRowCount: number} {
  return value => {
    const match = evaluated.find(
      ([key]) => Math.abs(key - value) <= maximumRelativeDistance * Math.abs(key)
    );
    // A value mgwr never evaluated means the mirror's sequence has diverged from mgwr's.
    expect(match, `mgwr evaluated ${value}`).toBeDefined();
    return {score: match![1], singularRowCount: 0};
  };
}

it('the golden-section mirror reproduces the sequence of evaluated bandwidths of mgwr', () => {
  expect(MGWR_GOLDEN_SECTION_REFERENCE.length).toBeGreaterThanOrEqual(8);
  for (const reference of MGWR_GOLDEN_SECTION_REFERENCE) {
    const trace = simulateGoldenSectionSearch({
      lower: reference.lower,
      upper: reference.upper,
      isAdaptive: !reference.fixed,
      maximumIterations: 200,
      evaluate: getEvaluator(reference.evaluated),
      getTolerance: getMgwrGoldenSectionTolerance
    });
    // Same float64 arithmetic as numpy: the very same bandwidths in the very same order.
    const distinct = getDistinctEvaluatedValues(trace);
    expect(distinct.length, reference.label).toBe(reference.evaluated.length);
    distinct.forEach((value, index) => {
      expect(value, `${reference.label} evaluation ${index}`).toBeCloseTo(
        reference.evaluated[index][0],
        11
      );
    });
    expect(trace.iterationCount, `${reference.label} iterations`).toBe(reference.iterationCount);
    expect(trace.converged, `${reference.label} converged`).toBe(true);
    expect(trace.optimumValue, `${reference.label} optimum`).toBeCloseTo(
      reference.unroundedBandwidth,
      11
    );
    expect(trace.optimumScore, `${reference.label} score`).toBe(reference.aicc);
    // mgwr reports the optimum rounded to two decimals; adaptive optima are integers.
    expect(Math.round(trace.optimumValue * 100) / 100, reference.label).toBeCloseTo(
      reference.bandwidth,
      12
    );
    if (!reference.fixed) {
      expect(Number.isInteger(reference.bandwidth), reference.label).toBe(true);
    }
  }
});

it('the float32 mirror with the kernel tolerance stops early on the same adaptive path', () => {
  // Adaptive bandwidths are integers, so float32 state reproduces mgwr's evaluations exactly; the
  // fixed-bandwidth interior points differ from float64 in the last digits, so only the adaptive
  // cases are compared (the evaluator is keyed on the float64 values).
  for (const reference of MGWR_GOLDEN_SECTION_REFERENCE.filter(entry => !entry.fixed)) {
    const label = reference.label;
    const trace = simulateGoldenSectionSearch({
      lower: reference.lower,
      upper: reference.upper,
      isAdaptive: true,
      maximumIterations: 30,
      evaluate: getEvaluator(reference.evaluated),
      getTolerance: getGPUGoldenSectionTolerance,
      precision: 'float32'
    });
    expect(trace.iterationCount, `${label} iterations`).toBeLessThanOrEqual(
      reference.iterationCount
    );
    // The same path, stopped earlier or together: its distinct evaluations are a prefix of mgwr's.
    const distinct = getDistinctEvaluatedValues(trace);
    expect(distinct, label).toEqual(reference.evaluated.slice(0, distinct.length).map(([v]) => v));
    // It stops when the interior scores differ by at most the kernel tolerance, so its optimum is
    // within that of the final mgwr optimum on these curves (checked against the pinned trace).
    const scale = Math.max(Math.abs(trace.optimumScore), Math.abs(reference.aicc));
    const tolerance = Math.max(
      1e-6,
      GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE * scale
    );
    expect(Math.abs(trace.optimumScore - reference.aicc), label).toBeLessThanOrEqual(
      Math.max(tolerance, 0.05)
    );
  }
});

it('golden section stops on the tolerance, the iteration cap, singular rows and invalid scores', () => {
  const curve = (value: number) => (value - 70) ** 2 / 10 + 300;
  const run = (
    evaluate: (value: number) => {score: number; singularRowCount: number},
    maximumIterations = 30
  ) =>
    simulateGoldenSectionSearch({
      lower: 46,
      upper: 128,
      isAdaptive: true,
      maximumIterations,
      evaluate,
      getTolerance: getMgwrGoldenSectionTolerance
    });
  // A unimodal integer curve converges onto its minimum.
  const unimodal = run(value => ({score: curve(value), singularRowCount: 0}));
  expect(unimodal.converged).toBe(true);
  expect(Math.abs(unimodal.optimumValue - 70)).toBeLessThanOrEqual(1);
  // An iteration cap of one performs one comparison and evaluates exactly the first two points.
  const capped = run(value => ({score: curve(value), singularRowCount: 0}), 1);
  expect(capped.iterationCount).toBe(1);
  expect(capped.converged).toBe(false);
  expect(capped.evaluations.length).toBe(2);
  // N iterations evaluate at most N + 1 candidates (two in the first iteration, one per later one).
  for (const cap of [2, 3, 5, 8]) {
    const trace = run(value => ({score: curve(value), singularRowCount: 0}), cap);
    expect(trace.evaluations.length).toBeLessThanOrEqual(cap + 1);
    expect(Math.max(...trace.evaluations.map(({row}) => row))).toBeLessThanOrEqual(cap);
  }
  // Fewest singular rows first: a lower AICc with more singular rows loses.
  const singular = run(value => ({
    score: curve(value),
    singularRowCount: value < 60 ? 3 : 0
  }));
  expect(singular.optimumValue).toBeGreaterThanOrEqual(60);
  // An invalid score loses to a valid one, and two invalid scores stop the search.
  const invalidLow = run(value => ({
    score: value < 80 ? Number.NaN : curve(value),
    singularRowCount: 0
  }));
  expect(invalidLow.optimumValue).toBeGreaterThanOrEqual(80);
  const allInvalid = run(() => ({score: Number.NaN, singularRowCount: 0}));
  expect(allInvalid.iterationCount).toBe(1);
  expect(allInvalid.converged).toBe(true);
  expect(Number.isNaN(allInvalid.optimumScore)).toBe(true);
  // A flat curve has equal interior scores: one comparison and the tolerance stops it.
  const flat = run(() => ({score: 300, singularRowCount: 0}));
  expect(flat.iterationCount).toBe(1);
  expect(flat.converged).toBe(true);
  // numpy round: ties go to the even integer.
  expect([0.5, 1.5, 2.5, 3.5, -0.5, 77.32].map(roundHalfToEven)).toEqual([0, 2, 2, 4, 0, 77]);
});

it('golden section over the planar scene matches mgwr with the float64 oracle', () => {
  // The CPU oracle runs the same state machine on its own evaluations of the scene: with mgwr's
  // tolerance it must retrace mgwr's adaptive bisquare search from its bounds [46, 128].
  const reference = MGWR_GOLDEN_SECTION_REFERENCE.find(
    entry => entry.label === 'adaptive bisquare default lower, upper 128'
  )!;
  const oracle = computeGeographicallyWeightedRegressionOnCPU({
    ...planar,
    settings: {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [0, 0]},
    bandwidthSearch: 'golden-section',
    maximumNeighborCount: 128,
    maximumSearchIterations: 200,
    searchTolerance: 'mgwr'
  });
  // maximumNeighborCount 128 caps the default upper bound at 128 (n = 144), as in the fixture case.
  expect(oracle.searchBounds).toEqual([reference.lower, reference.upper]);
  expect(oracle.searchIterationCount).toBe(reference.iterationCount);
  expect(oracle.selectedValue).toBe(reference.bandwidth);
  const evaluatedValues = oracle.candidates
    .filter(candidate => Number.isFinite(candidate.value))
    .map(candidate => candidate.value);
  for (const [value] of reference.evaluated) {
    expect(evaluatedValues, `mgwr evaluated k = ${value}`).toContain(value);
  }
  // The AICc of every evaluated k matches mgwr's exhaustive curve (float64 versus float64).
  const adaptive = MGWR_BANDWIDTH_SEARCH_REFERENCE.adaptive;
  for (const candidate of oracle.candidates) {
    if (Number.isFinite(candidate.value) && adaptive.ks.includes(candidate.value)) {
      // The oracle scales the k-th neighbour distance by 1.00001 and mgwr by 1.0000001: weights
      // differ by (1.00001 / 1.0000001)^2 - 1 = 2e-5 relative in h^2, so the AICc by well under 2e-5.
      const want = adaptive.aicc[adaptive.ks.indexOf(candidate.value)];
      expect(Math.abs(candidate.aicc - want), `AICc at k = ${candidate.value}`).toBeLessThanOrEqual(
        2e-5 * Math.abs(want)
      );
    }
  }
  // Table layout: rows 0 and 1 are the first interior points, then one row per iteration.
  expect(oracle.candidates.length).toBe(201);
  expect(oracle.selectedIndex).toBeGreaterThanOrEqual(0);
  expect(oracle.candidates[oracle.selectedIndex].value).toBe(reference.bandwidth);
});

it('the Gaussian cutoff truncation error follows the documented bound on the planar scene', () => {
  // Exact Gaussian fit of mgwr, fixed bandwidth 1.5, versus the float64 oracle with a cutoff. The
  // omitted weight of a row beyond c bandwidths is below exp(-c^2 / 2) of the self weight; for a
  // locally uniform density the omitted share of the weight sum is exp(-c^2 / 2) and of the second
  // moment (1 + c^2 / 2) exp(-c^2 / 2). The coefficient error is bounded by that share times the
  // local conditioning, which this scene (a few unit-scale predictors) keeps below 20.
  const exact = MGWR_INFERENCE_REFERENCE.fixedGaussian15Full;
  const errorAt = (gaussianCutoff: number | undefined): number => {
    const oracle = computeGeographicallyWeightedRegressionOnCPU({
      ...planar,
      settings: {kernel: 'gaussian', bandwidthMode: 'fixed', bandwidths: [1.5]},
      maximumBandwidthCount: 1,
      gaussianCutoff
    });
    let worst = 0;
    exact.coefficients.forEach((want, index) => {
      worst = Math.max(
        worst,
        Math.abs(oracle.coefficients[index] - want) / Math.max(0.1, Math.abs(want))
      );
    });
    return worst;
  };
  // The float64 oracle without a cutoff reproduces mgwr to double rounding of the Cholesky solve.
  expect(errorAt(undefined)).toBeLessThan(1e-9);
  let previous = Number.POSITIVE_INFINITY;
  for (const cutoff of [2, 3, 4, 5, 6]) {
    const error = errorAt(cutoff);
    const secondMomentShare = (1 + (cutoff * cutoff) / 2) * Math.exp(-(cutoff * cutoff) / 2);
    expect(error, `cutoff ${cutoff}`).toBeLessThanOrEqual(20 * secondMomentShare + 1e-9);
    expect(error, `cutoff ${cutoff} decreases`).toBeLessThanOrEqual(previous);
    previous = error;
  }
});
