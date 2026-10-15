// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUEmergingHotSpots,
  getGPUEmergingHotSpotsParameterValues
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import type {GPUMultipleTestingMethod} from '../../../src/analysis-core/contracts/inference';
import fixtureJson from '../../oracles/fixtures/parity-wp1-emerging-weighted.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import {getConditionalTrendReference} from '../../oracles/parity-wp1-trend-reference';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

type EmergingCase = {
  name: string;
  reference: string;
  n: number;
  width: number;
  height: number;
  sliceCount: number;
  values: number[];
  mask: number[];
  temporalWindow: number;
  multipleTesting: GPUMultipleTestingMethod;
  finalSlice: 'require' | 'last-valid';
  expected: Record<string, number[]>;
  offsets: number[];
  neighbors: number[];
  weights: number[];
  confidence: number;
};
const fixture = decodeRound4Fixture<{cases: EmergingCase[]}>(fixtureJson);

function compareValues(
  testCase: EmergingCase,
  metric: string,
  actual: number[],
  exact: boolean
): void {
  const expected = testCase.expected[metric];
  const scale = Math.max(1, ...expected.filter(Number.isFinite).map(Math.abs));
  const absolute = exact ? 0 : 1e-6 * scale;
  const relative = exact ? 0 : 1e-4;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  expected.forEach((value, row) => {
    if (!Number.isFinite(value) || !Number.isFinite(actual[row])) {
      if (!Object.is(actual[row], value)) mismatches++;
      return;
    }
    const error = Math.abs(actual[row] - value);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(value), Number.MIN_VALUE));
    if (error > absolute + relative * Math.abs(value)) mismatches++;
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUEmergingHotSpots',
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.n,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {profile: exact ? 'E' : 'P/observed-S', absolute, relative}
    })
  );
  expect.soft(mismatches, `${testCase.name}: ${metric}`).toBe(0);
}

for (let caseIndex = 0; caseIndex < 54; caseIndex++) {
  it(`GPUEmergingHotSpots weighted published-reference parity: ${caseIndex}`, async () => {
    const testCase = (await fixture).cases[caseIndex];
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device, 'parity-wp1-emerging');
    try {
      const cellCount = testCase.width * testCase.height;
      const outputs = {
        giZScores: rig.output('float32', testCase.n),
        pValues: rig.output('float32', testCase.n),
        adjustedPValues: rig.output('float32', testCase.n),
        globalStatistics: rig.output('float32', 4),
        trendZ: rig.output('float32', cellCount),
        trendP: rig.output('float32', cellCount),
        trendS: rig.output('sint32', cellCount),
        category: rig.output('uint32', cellCount),
        hotSliceCount: rig.output('uint32', cellCount),
        coldSliceCount: rig.output('uint32', cellCount),
        lastValidSlice: rig.output('uint32', cellCount)
      };
      rig.run(
        new GPUEmergingHotSpots({
          values: rig.input(Uint32Array.from(testCase.values), 'uint32'),
          mask: rig.input(Uint32Array.from(testCase.mask), 'uint32'),
          weights: {
            offsets: rig.input(new Uint32Array(testCase.offsets), 'uint32'),
            neighbors: rig.input(new Uint32Array(testCase.neighbors), 'uint32'),
            weights: rig.input(new Float32Array(testCase.weights), 'float32')
          },
          selfWeight: 1,
          sliceCount: testCase.sliceCount,
          multipleTesting: testCase.multipleTesting,
          finalSlice: testCase.finalSlice,
          parameters: rig.input(
            getGPUEmergingHotSpotsParameterValues({
              radius: 1,
              temporalWindow: testCase.temporalWindow,
              multipleTesting: testCase.multipleTesting,
              confidenceLevel: testCase.confidence,
              trendSignificanceLevel: 0.05,
              minimumTrendSliceCount: 10
            }),
            'float32'
          ),
          output: {
            giZScores: outputs.giZScores.view,
            pValues: outputs.pValues.view,
            adjustedPValues: outputs.adjustedPValues.view,
            globalStatistics: outputs.globalStatistics.view,
            trendZ: outputs.trendZ.view,
            trendP: outputs.trendP.view,
            trendS: outputs.trendS.view,
            category: outputs.category.view,
            hotSliceCount: outputs.hotSliceCount.view,
            coldSliceCount: outputs.coldSliceCount.view,
            lastValidSlice: outputs.lastValidSlice.view
          }
        })
      );
      // Gi* retains its original S bound. Exact Kendall ranks must refer to
      // the actual f32 stage boundary, not a separately rounded f64 Gi*.
      const conditionalTrend = await getConditionalTrendReference(
        testCase.name,
        await outputs.giZScores.readFloat32(),
        testCase.sliceCount
      );
      const comparisonCase = {
        ...testCase,
        reference: `${testCase.reference}; conditional SciPy Kendall rank table`,
        expected: {...testCase.expected, ...conditionalTrend}
      };
      for (const [metric, output] of Object.entries(outputs)) {
        const exact = [
          'trendS',
          'category',
          'hotSliceCount',
          'coldSliceCount',
          'lastValidSlice'
        ].includes(metric);
        const actual = exact ? await output.readUint32() : await output.readFloat32();
        compareValues(
          comparisonCase,
          metric,
          metric === 'trendS' ? actual.map(value => value | 0) : actual,
          exact
        );
      }
    } finally {
      rig.destroy();
    }
  });
}
