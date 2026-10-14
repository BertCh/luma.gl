// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOTS_CATEGORIES,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  type GPUEmergingHotSpotsOutput,
  type GPUEmergingHotSpotsProps
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import {
  getSpaceTimeDaysFromCivil,
  getSpaceTimeSliceIndex,
  type GPUSpaceTimeSlices
} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  classifyEmergingHotSpotSeries,
  computeAdjustedBinPValues,
  computeBinStates,
  computeEmergingHotSpotCells,
  computeEmergingHotSpotMoments,
  computeMannKendall,
  computeSpaceTimeGiStar,
  createDesignedCube,
  createDesignedSeries
} from './emerging-hot-spots-oracle';
import {
  EMERGING_HOT_SPOTS_TOLERANCES,
  MANN_KENDALL_REFERENCE,
  MULTIPLE_TESTING_ADJUSTED,
  MULTIPLE_TESTING_P,
  MULTIPLE_TESTING_SIGNIFICANT,
  MULTIPLE_TESTING_TOP_BINS,
  MULTIPLE_TESTING_VALUES,
  SLICE_CASES,
  SLICE_EVENTS
} from './emerging-hot-spots-reference';

let serial = 0;

const OUTPUT_NAMES = [
  'giZScores',
  'pValues',
  'adjustedPValues',
  'lastValidSlice',
  'trendZ',
  'trendP',
  'trendS',
  'category',
  'hotSliceCount',
  'coldSliceCount',
  'globalStatistics'
] as const;

/** Props overrides; output fields given at the top level are moved into `output`. */
type PropsOverrides = Partial<Omit<GPUEmergingHotSpotsProps, 'output'>> &
  Partial<GPUEmergingHotSpotsOutput>;

function createProps(
  graph: GPUCommandGraph,
  overrides: PropsOverrides = {}
): GPUEmergingHotSpotsProps {
  const props = createBaseProps(graph) as GPUEmergingHotSpotsProps & Record<string, unknown>;
  const output = {...props.output} as Record<string, unknown>;
  for (const [name, value] of Object.entries(overrides)) {
    if ((OUTPUT_NAMES as readonly string[]).includes(name)) {
      output[name] = value;
    } else {
      props[name] = value;
    }
  }
  return {...props, output: output as GPUEmergingHotSpotsOutput};
}

function createBaseProps(graph: GPUCommandGraph): GPUEmergingHotSpotsProps {
  const view = <Format extends 'uint32' | 'float32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  // 4 x 3 lattice, 5 slices.
  return {
    values: view('float32', 60),
    gridWidth: 4,
    gridHeight: 3,
    sliceCount: 5,
    parameters: view('float32', GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH),
    output: {
      giZScores: view('float32', 60),
      trendZ: view('float32', 12),
      trendP: view('float32', 12),
      trendS: view('sint32', 12),
      category: view('uint32', 12),
      hotSliceCount: view('uint32', 12),
      coldSliceCount: view('uint32', 12)
    }
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => PropsOverrides,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUEmergingHotSpots(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUEmergingHotSpotsParameterValues packs and validates the layout', () => {
  // Defaults: 95% confidence (was 90%), trend level 1 - confidence, minimum trend length 10.
  expect(Array.from(getGPUEmergingHotSpotsParameterValues({radius: 2, temporalWindow: 3}))).toEqual(
    [
      2,
      3,
      Math.fround(1.959963984540054),
      Math.fround(1 - 0.95),
      Math.fround(0.9),
      10,
      Math.fround(1 - 0.95),
      0
    ]
  );
  const strict = getGPUEmergingHotSpotsParameterValues({
    radius: 1.5,
    temporalWindow: 0,
    confidenceLevel: 0.99,
    persistentFraction: 0.8,
    minimumTrendSliceCount: 0
  });
  expect(strict[2]).toBeCloseTo(2.5758293, 6);
  expect(strict[3]).toBeCloseTo(0.01, 6);
  expect(strict[4]).toBeCloseTo(0.8, 6);
  expect(strict[5]).toBe(0);
  expect(strict[6]).toBeCloseTo(0.01, 6);
  // The tabulated critical z of the previous default.
  expect(
    getGPUEmergingHotSpotsParameterValues({
      temporalWindow: 0,
      confidenceLevel: 0.9,
      multipleTesting: 'none'
    })[2]
  ).toBeCloseTo(1.6448536, 6);
  // criticalZ only goes with 'none'.
  expect(
    getGPUEmergingHotSpotsParameterValues({
      radius: 0,
      temporalWindow: 1,
      criticalZ: 3,
      multipleTesting: 'none'
    })[2]
  ).toBe(3);
  // Any level is allowed with a correction.
  expect(
    getGPUEmergingHotSpotsParameterValues({temporalWindow: 0, confidenceLevel: 0.8})[6]
  ).toBeCloseTo(0.2, 6);
  const bad = (parameters: Parameters<typeof getGPUEmergingHotSpotsParameterValues>[0]) =>
    expect(() => getGPUEmergingHotSpotsParameterValues(parameters));
  bad({radius: -1, temporalWindow: 0}).toThrow(/radius/);
  bad({radius: NaN, temporalWindow: 0}).toThrow(/radius/);
  bad({radius: 1, temporalWindow: 1.5}).toThrow(/temporalWindow/);
  bad({radius: 1, temporalWindow: -1}).toThrow(/temporalWindow/);
  bad({radius: 1, temporalWindow: 0, criticalZ: 3}).toThrow(/requires multipleTesting 'none'/);
  bad({radius: 1, temporalWindow: 0, criticalZ: 0, multipleTesting: 'none'}).toThrow(/criticalZ/);
  bad({radius: 1, temporalWindow: 0, trendSignificanceLevel: 1}).toThrow(/trendSignificanceLevel/);
  bad({radius: 1, temporalWindow: 0, persistentFraction: 0.5}).toThrow(/persistentFraction/);
  bad({radius: 1, temporalWindow: 0, confidenceLevel: 0.8, multipleTesting: 'none'}).toThrow(
    /confidenceLevel/
  );
  bad({radius: 1, temporalWindow: 0, confidenceLevel: 1}).toThrow(/confidenceLevel/);
  bad({radius: 1, temporalWindow: 0, minimumTrendSliceCount: 2.5}).toThrow(
    /minimumTrendSliceCount/
  );
  bad({radius: 1, temporalWindow: 0, minimumTrendSliceCount: -1}).toThrow(/minimumTrendSliceCount/);
  expect(() =>
    getGPUEmergingHotSpotsParameterValues({radius: 1, temporalWindow: 0}, new Float32Array(2))
  ).toThrow(/target/);
});

it('GPUEmergingHotSpots validates its inputs', () => {
  expectThrows(() => ({gridWidth: 0}), /gridWidth/);
  expectThrows(() => ({gridHeight: undefined}), /gridHeight/);
  expectThrows(() => ({gridWidth: undefined, gridHeight: undefined}), /exactly one/);
  expectThrows(
    graph => ({
      weights: {
        offsets: createTransientView(graph, 'o', 'uint32', 13),
        neighbors: createTransientView(graph, 'n', 'uint32', 20),
        weights: createTransientView(graph, 'w', 'float32', 20)
      }
    }),
    /exactly one/
  );
  expectThrows(
    graph => ({
      gridWidth: undefined,
      gridHeight: undefined,
      selfWeight: -1,
      weights: {
        offsets: createTransientView(graph, 'o', 'uint32', 13),
        neighbors: createTransientView(graph, 'n', 'uint32', 20),
        weights: createTransientView(graph, 'w', 'float32', 20)
      }
    }),
    /selfWeight/
  );
  expectThrows(
    graph => ({
      gridWidth: undefined,
      gridHeight: undefined,
      weights: {
        offsets: createTransientView(graph, 'o', 'uint32', 11),
        neighbors: createTransientView(graph, 'n', 'uint32', 20),
        weights: createTransientView(graph, 'w', 'float32', 20)
      }
    }),
    /values length/
  );
  expectThrows(() => ({multipleTesting: 'sidak' as never}), /method must be one of/);
  expectThrows(() => ({finalSlice: 'first' as never}), /finalSlice/);
  expectThrows(
    graph => ({pValues: createTransientView(graph, 'p', 'float32', 59)}),
    /pValues length/
  );
  expectThrows(
    graph => ({adjustedPValues: createTransientView(graph, 'a', 'float32', 12)}),
    /adjustedPValues length/
  );
  expectThrows(
    graph => ({lastValidSlice: createTransientView(graph, 'l', 'uint32', 60)}),
    /lastValidSlice length/
  );
  expectThrows(graph => {
    const values = createTransientView(graph, 'alias-p', 'float32', 60);
    return {values, pValues: values};
  }, /must not share buffers/);
  expectThrows(() => ({sliceCount: 1.5}), /sliceCount/);
  expectThrows(() => ({sliceCount: 257}), /sliceCount must be an integer in \[1, 256\]/);
  expectThrows(() => ({maximumRadius: 33}), /maximumRadius/);
  expectThrows(() => ({maximumRadius: 1.5}), /maximumRadius/);
  expectThrows(
    graph => ({values: createTransientView(graph, 'short', 'float32', 59)}),
    /values length/
  );
  expectThrows(
    graph => ({values: createTransientView(graph, 'bad', 'float32x2', 60)}) as never,
    /values/
  );
  expectThrows(
    graph => ({parameters: createTransientView(graph, 'p', 'float32', 7)}),
    /parameters/
  );
  expectThrows(graph => ({mask: createTransientView(graph, 'm', 'uint32', 11)}), /mask length/);
  expectThrows(
    graph => ({giZScores: createTransientView(graph, 'z', 'float32', 12)}),
    /giZScores length/
  );
  expectThrows(
    graph => ({category: createTransientView(graph, 'c', 'uint32', 60)}),
    /category length/
  );
  expectThrows(
    graph => ({
      globalStatistics: createTransientView(graph, 'g', 'float32', 3)
    }),
    /globalStatistics/
  );
  expectThrows(graph => {
    const values = createTransientView(graph, 'alias', 'float32', 60);
    return {values, giZScores: values};
  }, /must not share buffers/);
});

it('GPUEmergingHotSpots creates deterministic nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const mask = createTransientView(graph, 'mask', 'uint32', 12);
  // Without a correction (and no p-value outputs) the p-value passes are not built.
  const contributor = new GPUEmergingHotSpots(
    createProps(graph, {id: 'eh', mask, multipleTesting: 'none'})
  );
  const nodes = contributor.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids).toEqual([
    'eh-block-sums',
    'eh-mean',
    'eh-block-squares',
    'eh-deviation',
    'eh-gi-star',
    'eh-slice-counts',
    // Mann-Kendall and the category classification share one pass over each cell's series.
    'eh-mann-kendall'
  ]);
  expect(nodes.at(-1)?.workload).toMatchObject({
    variant: 'mann-kendall-classify',
    maximumInvocationCount: 12
  });
  // The default is Benjamini-Hochberg: p-values and the adjustment run between Gi* and the trend.
  const defaultGraph = new GPUCommandGraph(device);
  const defaultNodes = new GPUEmergingHotSpots(
    createProps(defaultGraph, {id: 'eh'})
  ).getCommandNodes(defaultGraph);
  const defaultIds = defaultNodes.map(node => node.id);
  expect(defaultIds.slice(0, 5)).toEqual(ids.slice(0, 5));
  expect(defaultIds[5]).toBe('eh-p-values');
  expect(defaultIds.at(-2)).toBe('eh-slice-counts');
  expect(defaultIds.at(-1)).toBe('eh-mann-kendall');
  expect(defaultIds.length).toBeGreaterThan(7);
  expect(new Set(defaultIds).size).toBe(defaultIds.length);
  // 'none' with a requested p-value output still builds the p-values (and a pass-through adjustment).
  const outputGraph = new GPUCommandGraph(device);
  const outputIds = new GPUEmergingHotSpots(
    createProps(outputGraph, {
      id: 'eh',
      multipleTesting: 'none',
      pValues: createTransientView(outputGraph, 'out-p', 'float32', 60),
      lastValidSlice: createTransientView(outputGraph, 'out-last', 'uint32', 12)
    })
  )
    .getCommandNodes(outputGraph)
    .map(node => node.id);
  expect(outputIds).toContain('eh-p-values');
  const longSeriesGraph = new GPUCommandGraph(device);
  const longSeries = new GPUEmergingHotSpots(
    createProps(longSeriesGraph, {
      sliceCount: 64,
      values: createTransientView(longSeriesGraph, 'long-values', 'float32', 12 * 64),
      giZScores: createTransientView(longSeriesGraph, 'long-z', 'float32', 12 * 64)
    })
  );
  expect(longSeries.getCommandNodes(longSeriesGraph).at(-1)?.workload).toMatchObject({
    variant: 'mann-kendall-cooperative',
    maximumWorkgroupCount: 12,
    maximumInvocationCount: 12 * 64
  });
  const secondGraph = new GPUCommandGraph(device);
  const again = new GPUEmergingHotSpots(
    createProps(secondGraph, {id: 'eh', multipleTesting: 'none'})
  );
  expect(again.getCommandNodes(secondGraph).map(node => node.id)).toEqual(ids);
  // Views from another graph are rejected.
  const foreign = new GPUEmergingHotSpots(createProps(secondGraph, {id: 'foreign'}));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/must belong to the target graph/);
  // uint32 counts are accepted.
  const countsGraph = new GPUCommandGraph(device);
  const counts = new GPUEmergingHotSpots(
    createProps(countsGraph, {
      multipleTesting: 'none',
      values: createTransientView(countsGraph, 'counts', 'uint32', 60)
    })
  );
  expect(counts.getCommandNodes(countsGraph)).toHaveLength(7);
  device.destroy();
});

it('GPUEmergingHotSpots.getInferenceMetadata describes the inference', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const settings = {temporalWindow: 1};
  const metadata = new GPUEmergingHotSpots(createProps(graph)).getInferenceMetadata(settings);
  expect(metadata).toEqual({
    alternative: 'two-sided',
    permutationCount: 0,
    multipleTesting: 'benjamini-hochberg',
    significanceLevel: 1 - 0.95,
    testedRowsDefinition: 'valid space-time bins'
  });
  const corrected = new GPUEmergingHotSpots(
    createProps(graph, {multipleTesting: 'benjamini-yekutieli'})
  );
  expect(corrected.getInferenceMetadata({...settings, confidenceLevel: 0.9})).toMatchObject({
    multipleTesting: 'benjamini-yekutieli',
    significanceLevel: 1 - 0.9
  });
  // One source of truth: the packed values come from the node's own multipleTesting.
  expect(Array.from(corrected.getParameterValues({...settings, confidenceLevel: 0.8}))).toEqual(
    Array.from(
      getGPUEmergingHotSpotsParameterValues({
        ...settings,
        confidenceLevel: 0.8,
        multipleTesting: 'benjamini-yekutieli'
      })
    )
  );
  expect(() => corrected.getParameterValues({...settings, multipleTesting: 'none'})).toThrow(
    /must match the node/
  );
  const uncorrected = new GPUEmergingHotSpots(createProps(graph, {multipleTesting: 'none'}));
  expect(uncorrected.getInferenceMetadata(settings).multipleTesting).toBe('none');
  // Settings the packer rejects are rejected by the metadata too.
  expect(() => uncorrected.getInferenceMetadata({...settings, confidenceLevel: 0.8})).toThrow(
    /confidenceLevel/
  );
  device.destroy();
});

it('the Mann-Kendall oracle matches hand-computed statistics', () => {
  // Monotone series of 10: S = 45, Var = 10 * 9 * 25 / 18 = 125.
  const monotone = computeMannKendall(Array.from({length: 10}, (_, index) => index));
  expect(monotone.statistic).toBe(45);
  expect(monotone.variance).toBe(125);
  expect(monotone.z).toBeCloseTo(44 / Math.sqrt(125), 12);
  expect(monotone.p).toBeCloseTo(8.1e-5, 5);
  // Ties: [1, 2, 2, 3] has S = 5 and one tie group of size 2: (4 * 3 * 13 - 18) / 18 = 138 / 18.
  const tied = computeMannKendall([1, 2, NaN, 2, 3]);
  expect(tied.count).toBe(4);
  expect(tied.statistic).toBe(5);
  expect(tied.tieTerm).toBe(18);
  expect(tied.variance).toBeCloseTo(138 / 18, 12);
  expect(tied.z).toBeCloseTo(4 / Math.sqrt(138 / 18), 12);
  // Descending, constant and degenerate series.
  expect(computeMannKendall([5, 4, 3]).statistic).toBe(-3);
  expect(computeMannKendall([5, 4, 3]).z).toBeLessThan(0);
  const constant = computeMannKendall([2, 2, 2, 2]);
  expect([constant.statistic, constant.z, constant.p]).toEqual([0, 0, 1]);
  expect(computeMannKendall([NaN, NaN])).toMatchObject({
    count: 0,
    statistic: 0,
    z: 0,
    p: 1
  });
});

// Mann-Kendall reference: see MANN_KENDALL_REFERENCE (numpy implementation, scipy.stats.norm p).
it('the Mann-Kendall oracle matches the numpy and scipy reference', () => {
  for (const [name, reference] of Object.entries(MANN_KENDALL_REFERENCE)) {
    const {count, statistic, variance, z, p} = reference;
    const result = computeMannKendall(reference.series);
    expect(result.count, name).toBe(count);
    expect(result.statistic, name).toBe(statistic);
    const {mannKendallDecimals, mannKendallPValue} = EMERGING_HOT_SPOTS_TOLERANCES;
    expect(result.variance, name).toBeCloseTo(variance, mannKendallDecimals);
    expect(result.z, name).toBeCloseTo(z, mannKendallDecimals);
    // The oracle's p uses the same erfcc fit as the shader (relative error below 1.2e-7).
    expect(Math.abs(result.p - p), name).toBeLessThan(
      mannKendallPValue.relative * Math.max(p, mannKendallPValue.floor) + mannKendallPValue.absolute
    );
  }
});

it('multiple testing removes weakly significant bins and matches statsmodels', () => {
  const cube = {
    gridWidth: MULTIPLE_TESTING_VALUES.length,
    gridHeight: 1,
    sliceCount: 1,
    values: Float32Array.from(MULTIPLE_TESTING_VALUES)
  };
  const packed = getGPUEmergingHotSpotsParameterValues({radius: 0, temporalWindow: 0});
  const zScores = computeSpaceTimeGiStar(cube, packed);
  // The oracle's p uses the shader's erfcc fit (relative error below 1.2e-7).
  const pValues = computeAdjustedBinPValues(zScores, 'none');
  MULTIPLE_TESTING_TOP_BINS.forEach((bin, index) => {
    expect(Math.abs(pValues[bin] - MULTIPLE_TESTING_P[index])).toBeLessThan(
      EMERGING_HOT_SPOTS_TOLERANCES.pValue.relative * MULTIPLE_TESTING_P[index] +
        EMERGING_HOT_SPOTS_TOLERANCES.pValue.absolute
    );
  });
  for (const [method, expected] of Object.entries(MULTIPLE_TESTING_ADJUSTED)) {
    const adjusted = computeAdjustedBinPValues(zScores, method as 'holm');
    MULTIPLE_TESTING_TOP_BINS.forEach((bin, index) => {
      expect(Math.abs(adjusted[bin] - expected[index]), `${method} bin ${bin}`).toBeLessThan(
        EMERGING_HOT_SPOTS_TOLERANCES.adjustedPValue.relative * expected[index] +
          EMERGING_HOT_SPOTS_TOLERANCES.adjustedPValue.absolute
      );
    });
  }
  // Hot and cold bins: uncorrected 95% finds all six outliers (the three weak ones at p between
  // 0.02 and 0.04); every correction keeps only the three strong ones.
  for (const [method, significant] of Object.entries(MULTIPLE_TESTING_SIGNIFICANT)) {
    const level = getGPUEmergingHotSpotsParameterValues({
      temporalWindow: 0,
      multipleTesting: method as 'none'
    });
    const states = computeBinStates(zScores, level, {multipleTesting: method as 'none'});
    const flagged = Array.from(states, (state, bin) => (state > 0 ? bin : -1)).filter(
      bin => bin >= 0
    );
    expect(flagged, method).toEqual(significant);
  }
  const states = computeBinStates(zScores, packed, {multipleTesting: 'benjamini-hochberg'});
  // Strong outliers: 92 and 93 hot, 94 cold. Weak ones are dropped.
  expect([92, 93, 94, 95, 96, 97].map(bin => states[bin])).toEqual([1, 1, 2, 0, 0, 0]);
  // The family is every valid bin: an extra missing bin does not change anything.
  const withMissing = {...cube, values: Float32Array.from([...cube.values, NaN]), gridWidth: 101};
  const missingZ = computeSpaceTimeGiStar(withMissing, packed);
  expect(Number.isNaN(missingZ[100])).toBe(true);
  expect(computeBinStates(missingZ, packed).slice(0, 100)).toEqual(
    computeBinStates(zScores, packed)
  );
});

it('getSpaceTimeSliceIndex matches the python calendar reference across year boundaries', () => {
  for (const testCase of SLICE_CASES) {
    const [year, month, day, hour] = testCase.origin;
    const slices: GPUSpaceTimeSlices = {
      step: testCase.step,
      interval: testCase.interval,
      origin: {year, month, day, hour},
      count: testCase.count
    };
    const actual = SLICE_EVENTS.map(([eventYear, eventMonth, eventDay, eventHour]) =>
      getSpaceTimeSliceIndex(slices, {
        year: eventYear,
        month: eventMonth,
        day: eventDay,
        hour: eventHour
      })
    );
    expect(actual, `${testCase.step} x${testCase.interval}`).toEqual(testCase.expected);
  }
  // Spot checks of Hinnant's days_from_civil: the epoch, a leap day, a pre-epoch date.
  expect(getSpaceTimeDaysFromCivil(1970, 1, 1)).toBe(0);
  expect(getSpaceTimeDaysFromCivil(2024, 2, 29)).toBe(19782);
  expect(getSpaceTimeDaysFromCivil(1969, 12, 31)).toBe(-1);
  expect(getSpaceTimeDaysFromCivil(2000, 3, 1) - getSpaceTimeDaysFromCivil(2000, 2, 29)).toBe(1);
  expect(getSpaceTimeDaysFromCivil(1900, 3, 1) - getSpaceTimeDaysFromCivil(1900, 2, 28)).toBe(1);
});

it('the classifier oracle reaches all 17 categories on designed z series', () => {
  // 95% critical z without a correction; the default minimum trend length (10) applies.
  const packed = getGPUEmergingHotSpotsParameterValues({
    radius: 0,
    temporalWindow: 0,
    multipleTesting: 'none'
  });
  const hot = 3;
  const cold = -3;
  const series = (...segments: number[][]) => segments.flat();
  const ramp = (from: number, to: number, length = 20) =>
    Array.from({length}, (_, index) => from + ((to - from) * index) / (length - 1));
  const rep = (count: number, value: number) => new Array<number>(count).fill(value);
  const classify = (
    values: number[],
    finalSlice: 'require' | 'last-valid' = 'require',
    parameters = packed
  ) => {
    const trend = computeMannKendall(values);
    const known = trend.count >= parameters[5];
    return classifyEmergingHotSpotSeries(values, known ? trend : {z: NaN, p: NaN}, parameters, {
      finalSlice
    });
  };
  const categories = GPU_EMERGING_HOT_SPOTS_CATEGORIES;
  const cases: [string, number[], number][] = [
    ['new hot', series(rep(9, 0), [hot]), categories.NEW_HOT],
    ['consecutive hot', series(rep(7, 0), rep(3, hot)), categories.CONSECUTIVE_HOT],
    ['intensifying hot', ramp(2, 4), categories.INTENSIFYING_HOT],
    ['persistent hot', rep(20, hot), categories.PERSISTENT_HOT],
    ['diminishing hot', ramp(4, 2), categories.DIMINISHING_HOT],
    ['sporadic hot', series([hot], rep(3, 0), [hot], rep(5, 0), [hot, 0]), categories.SPORADIC_HOT],
    ['sporadic hot final hot', series([hot, 0, hot, 0, hot]), categories.SPORADIC_HOT],
    ['oscillating hot', series([cold, 0, hot, 0, hot, hot]), categories.OSCILLATING_HOT],
    ['historical hot', series(rep(19, hot), [0]), categories.HISTORICAL_HOT],
    ['new cold', series(rep(9, 0), [cold]), categories.NEW_COLD],
    ['consecutive cold', series(rep(7, 0), rep(3, cold)), categories.CONSECUTIVE_COLD],
    ['intensifying cold', ramp(-2, -4), categories.INTENSIFYING_COLD],
    ['persistent cold', rep(20, cold), categories.PERSISTENT_COLD],
    ['diminishing cold', ramp(-4, -2), categories.DIMINISHING_COLD],
    ['sporadic cold', series([cold, 0, 0, cold, 0]), categories.SPORADIC_COLD],
    ['oscillating cold', series([hot, 0, cold, 0, cold, cold]), categories.OSCILLATING_COLD],
    ['historical cold', series(rep(19, cold), [0]), categories.HISTORICAL_COLD],
    ['no pattern', rep(10, 0), categories.NO_PATTERN],
    ['mixed history, quiet end', series([hot, 0, cold, 0]), categories.NO_PATTERN],
    ['all missing', [NaN, NaN, NaN], categories.NO_PATTERN],
    // Persistent hot or cold with fewer than 10 finite slices has no trend decision.
    ['short persistent hot', rep(6, hot), categories.UNDETERMINED],
    ['short persistent cold', rep(6, cold), categories.UNDETERMINED],
    // A missing final slice (default 'require') makes any pattern undetermined.
    ['missing final slice', series(rep(9, 0), [hot, NaN]), categories.UNDETERMINED]
  ];
  const reached = new Set<number>();
  for (const [name, values, expected] of cases) {
    expect(classify(values), name).toBe(expected);
    reached.add(expected);
  }
  // The 17 ArcGIS categories plus UNDETERMINED (17).
  expect(reached.size).toBe(18);
  expect(Object.values(categories).sort((a, b) => a - b)).toEqual(
    Array.from({length: 18}, (_, i) => i)
  );
  expect(categories.UNDETERMINED).toBe(17);
  // finalSlice 'last-valid' keeps the old behavior: the last finite slice decides.
  expect(classify([0, 0, hot, NaN], 'last-valid')).toBe(categories.NEW_HOT);
  expect(classify([0, 0, hot, NaN])).toBe(categories.UNDETERMINED);
  // Interior missing slices are skipped under both modes.
  expect(classify([0, NaN, 0, hot])).toBe(categories.NEW_HOT);
  // A series with every pattern finite is unaffected by the mode.
  expect(classify(rep(20, hot), 'last-valid')).toBe(categories.PERSISTENT_HOT);
  // minimumTrendSliceCount 0 disables the guard: six constant hot slices are Persistent.
  const unguarded = getGPUEmergingHotSpotsParameterValues({
    temporalWindow: 0,
    multipleTesting: 'none',
    minimumTrendSliceCount: 0
  });
  expect(classify(rep(6, hot), 'require', unguarded)).toBe(categories.PERSISTENT_HOT);
  // Six slices below the minimum of ten give NaN trend z and p.
  const shortCells = computeEmergingHotSpotCells(
    [1, 2, 3, 4, 5, 6],
    {gridWidth: 1, gridHeight: 1, sliceCount: 6},
    packed,
    {multipleTesting: 'none'}
  );
  expect(shortCells.trendS).toEqual([15]);
  expect(Number.isNaN(shortCells.trendZ[0])).toBe(true);
  expect(Number.isNaN(shortCells.trendP[0])).toBe(true);
  // Hot 90% boundary: 9 of 10 is persistent, 8 of 10 is not.
  expect(classify(series(rep(1, 0), rep(9, hot)))).toBe(categories.PERSISTENT_HOT);
  expect(classify(series(rep(2, 0), rep(8, hot)))).toBe(categories.CONSECUTIVE_HOT);
});

it('the designed cube is classified as designed by the double precision oracle', () => {
  const {cube, designed, expectedCategories} = createDesignedCube();
  expect(designed.map(entry => entry.expected).sort((a, b) => a - b)).toEqual(
    expect.arrayContaining(Array.from({length: 17}, (_, index) => index))
  );
  const packed = getGPUEmergingHotSpotsParameterValues({
    radius: 0,
    temporalWindow: 0
  });
  const zScores = computeSpaceTimeGiStar(cube, packed);
  // The legend contract: the designed categories are the same under every correction method, and
  // the default is Benjamini-Hochberg.
  for (const multipleTesting of [
    'none',
    'bonferroni',
    'holm',
    'benjamini-hochberg',
    'benjamini-yekutieli'
  ] as const) {
    const methodPacked = getGPUEmergingHotSpotsParameterValues({
      radius: 0,
      temporalWindow: 0,
      multipleTesting
    });
    const methodCells = computeEmergingHotSpotCells(zScores, cube, methodPacked, {multipleTesting});
    for (const [cell, expected] of expectedCategories.entries()) {
      expect(
        methodCells.category[cell],
        `${multipleTesting} cell ${cell} ${designed[cell]?.name}`
      ).toBe(expected);
    }
  }
  const cells = computeEmergingHotSpotCells(zScores, cube, packed);
  for (const [cell, expected] of expectedCategories.entries()) {
    expect(cells.category[cell], `cell ${cell} ${designed[cell]?.name}`).toBe(expected);
  }
  // Radius 0 and window 0: z is the plain standard score of each valid bin.
  const moments = computeEmergingHotSpotMoments(cube);
  expect(zScores[0]).toBeCloseTo((cube.values[0] - moments.mean) / moments.standardDeviation, 12);
  // Masked and all-missing cells are excluded.
  expect(Number.isNaN(zScores[designedLength(designed) * 20])).toBe(true);
  expect(cells.category[designedLength(designed)]).toBe(0);
  expect(cells.category[designedLength(designed) + 1]).toBe(0);
  expect(createDesignedSeries()).toHaveLength(designed.length);
});

function designedLength(designed: unknown[]): number {
  return designed.length;
}

it('GPUEmergingHotSpots weights mode creates the same pipeline over weights bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `weights-mode-${serial++}`, format, length);
  const base = createProps(graph, {gridWidth: undefined, gridHeight: undefined});
  const contributor = new GPUEmergingHotSpots({
    ...base,
    weights: {
      offsets: view('uint32', 13),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    }
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('emerging-hot-spots-gi-star');
  expect(new Set(ids).size).toBe(ids.length);
  device.destroy();
});

it('GPUEmergingHotSpots refuses a cube larger than one storage binding (rule 16)', () => {
  const device = createNullWebGPUDevice();
  // 512 x 256 cells x 256 slices is exactly 2^25 four-byte bins, the 128 MiB binding limit.
  expect(512 * 256 * 256 * 4).toBe(device.limits.maxStorageBufferBindingSize);
  const sliceCount = 256;
  for (const [gridHeight, fits] of [
    [256, true],
    [257, false]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const cells = 512 * gridHeight;
    const view = <Format extends 'uint32' | 'float32' | 'sint32'>(format: Format, length: number) =>
      createTransientView(graph, `large-${serial++}`, format, length);
    const contributor = new GPUEmergingHotSpots(
      createProps(graph, {
        values: view('float32', cells * sliceCount),
        gridWidth: 512,
        gridHeight,
        sliceCount,
        multipleTesting: 'none',
        giZScores: view('float32', cells * sliceCount),
        trendZ: view('float32', cells),
        trendP: view('float32', cells),
        trendS: view('sint32', cells),
        category: view('uint32', cells),
        hotSliceCount: view('uint32', cells),
        coldSliceCount: view('uint32', cells)
      })
    );
    if (fits) {
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    } else {
      expect(() => contributor.getCommandNodes(graph)).toThrow(/maxStorageBufferBindingSize/);
    }
  }
  device.destroy();
});
