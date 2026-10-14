// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeClassBreaksOracle, getWithinClassCost} from './class-breaks-oracle';
import {computeNaturalBreaksMirror} from './natural-breaks-mirror';
import {
  CLASS_BREAKS_REFERENCE,
  getClassSizes,
  getGoodnessOfVarianceFit,
  getReferenceEdgeMismatches,
  getReferenceParameters,
  NATURAL_BREAKS_LOCAL_OPTIMA,
  NATURAL_BREAKS_LOCAL_OPTIMUM_FIT_TOLERANCE
} from './class-breaks-reference';

// The CPU mirror of GPUClassBreaks against mapclassify 2.11 and jenkspy 0.4.1 (Q4, B-21): the
// reference values come from a committed generator, not from this library's own conventions.
// Natural breaks are compared with the exact Fisher-Jenks partition: the lower edges must be equal
// data values, not merely a close goodness of fit.
const MAXIMUM_CLASS_COUNT = 10;
const NATURAL_BREAKS_BIN_COUNT = 2048;

it('class breaks fixture records its reference packages', () => {
  const {versions, generator} = CLASS_BREAKS_REFERENCE;
  expect(generator).toBe('test/oracles/dataframe-class-breaks.py');
  expect(Object.keys(versions).sort()).toEqual(['jenkspy', 'mapclassify', 'numpy', 'python']);
});

it('class breaks oracle matches mapclassify for every closed-form method', () => {
  const mismatches: string[] = [];
  let compared = 0;
  for (const [name, scene] of Object.entries(CLASS_BREAKS_REFERENCE.scenes)) {
    const values = Float32Array.from(scene.values);
    for (const referenceCase of scene.cases) {
      const result = computeClassBreaksOracle({
        values,
        parameters: getReferenceParameters(referenceCase),
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        naturalBreaksBinCount: NATURAL_BREAKS_BIN_COUNT
      });
      const label = `${name} ${referenceCase.method} k=${referenceCase.classCount}`;
      mismatches.push(
        ...getReferenceEdgeMismatches(label, result.breaks, result.classCount, referenceCase)
      );
      compared++;
    }
  }
  expect(mismatches).toEqual([]);
  expect(compared).toBe(48);
});

it('natural breaks mirror matches exact Fisher-Jenks except the two pinned local minima', () => {
  const mismatches: string[] = [];
  const localOptima: string[] = [];
  for (const [name, scene] of Object.entries(CLASS_BREAKS_REFERENCE.scenes)) {
    const values = Float32Array.from(scene.values);
    for (const reference of scene.naturalBreaks) {
      const result = computeClassBreaksOracle({
        values,
        parameters: {method: 'natural-breaks', classCount: reference.classCount},
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        naturalBreaksBinCount: NATURAL_BREAKS_BIN_COUNT
      });
      const label = `${name} ${reference.classCount}`;
      expect(result.classCount, label).toBe(reference.classCount);
      const inner = Array.from(result.breaks.slice(1, reference.classCount));
      // The fixture values are f32; the test transform of the JSON may perturb their f64 digits.
      const expected = reference.exactLowerEdges.map(Math.fround);
      if (inner.every((edge, index) => edge === expected[index])) {
        expect(getClassSizes(values, inner), label).toEqual(reference.exactClassSizes);
        continue;
      }
      const known = NATURAL_BREAKS_LOCAL_OPTIMA[label];
      if (!known) {
        mismatches.push(`${label}: ${inner.join(', ')} != ${expected.join(', ')}`);
        continue;
      }
      localOptima.push(label);
      if (known === 'mapclassify') {
        // mapclassify 2.11 stops in the same neighbouring optimum.
        expect(reference.mapclassifyAgrees, label).toBe(false);
        expect(inner, label).toEqual(reference.mapclassifyLowerEdges.map(Math.fround));
      }
      const shortfall =
        reference.exactGoodnessOfVarianceFit - getGoodnessOfVarianceFit(values, inner);
      expect(shortfall, label).toBeLessThanOrEqual(NATURAL_BREAKS_LOCAL_OPTIMUM_FIT_TOLERANCE);
    }
  }
  expect(mismatches).toEqual([]);
  // The known cases are the only ones that may differ from the exact partition.
  expect(localOptima.sort()).toEqual(Object.keys(NATURAL_BREAKS_LOCAL_OPTIMA).sort());
});

it('natural breaks mirror refuses nothing the references accept: every reference stays classifiable', () => {
  // mapclassify and jenkspy refuse a class count above the distinct values; GPUClassBreaks answers
  // with one class per distinct value, the sorted distinct values after the first.
  for (const [name, scene] of Object.entries(CLASS_BREAKS_REFERENCE.scenes)) {
    if (!scene.distinctValueEdges) {
      continue;
    }
    const values = Float32Array.from(scene.values);
    const distinct = scene.distinctValueEdges.map(Math.fround);
    if (distinct.length + 1 > MAXIMUM_CLASS_COUNT) {
      continue;
    }
    const result = computeClassBreaksOracle({
      values,
      parameters: {method: 'natural-breaks', classCount: MAXIMUM_CLASS_COUNT},
      maximumClassCount: MAXIMUM_CLASS_COUNT,
      naturalBreaksBinCount: NATURAL_BREAKS_BIN_COUNT
    });
    expect(result.classCount, name).toBe(distinct.length + 1);
    expect(Array.from(result.breaks.slice(1, result.classCount)), name).toEqual(distinct);
  }
});

it('the two Fisher-Jenks deviations require joint cuts, not more rounds or a wider window', () => {
  for (const [name, classCount, sizes, excess] of [
    ['lognormal', 7, [391, 214, 106, 59, 22, 4, 4], 0.0001094573368],
    ['outlier-pair', 5, [1, 92, 179, 129, 1], 0.0138836166317]
  ] as const) {
    const scene = CLASS_BREAKS_REFERENCE.scenes[name];
    const values = Float32Array.from(scene.values);
    const reference = scene.naturalBreaks.find(item => item.classCount === classCount)!;
    const run = (maximumRoundCount: number) =>
      computeNaturalBreaksMirror(
        values,
        Math.min(...values),
        Math.max(...values),
        classCount,
        2048,
        {maximumRoundCount}
      );
    const result = run(8);
    expect(run(64).edges, name).toEqual(result.edges);
    expect(getClassSizes(values, result.edges), name).toEqual(sizes);
    const cost = getWithinClassCost(values, result.edges);
    const optimum = getWithinClassCost(values, reference.exactLowerEdges);
    // These are measured costs, checked to 1e-10 absolute (well above f64 accumulation noise).
    expect(cost - optimum, name).toBeCloseTo(excess, 10);
    const distinct = Array.from(new Set(values)).sort((left, right) => left - right);
    for (let cut = 0; cut < result.edges.length; cut++) {
      for (const value of distinct) {
        if (
          value <= (result.edges[cut - 1] ?? -Infinity) ||
          value >= (result.edges[cut + 1] ?? Infinity)
        )
          continue;
        const candidate = [...result.edges];
        candidate[cut] = value;
        expect(getWithinClassCost(values, candidate), `${name} cut ${cut}`).toBeGreaterThanOrEqual(
          cost
        );
      }
    }
    expect(
      result.edges.filter((edge, index) => edge !== Math.fround(reference.exactLowerEdges[index]))
    ).toHaveLength(2);
    expect(optimum).toBeLessThan(cost);
  }
});
