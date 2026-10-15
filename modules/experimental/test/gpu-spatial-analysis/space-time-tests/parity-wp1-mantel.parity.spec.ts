// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUMantelTest,
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests';
import fixture from '../../oracles/fixtures/parity-wp1-mantel.json';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {makeMantelPairs} from './parity-wp1-mantel-input';

for (const testCase of fixture.cases) {
  for (const seed of testCase.seeds) {
    it(`GPUMantelTest independent parity: ${testCase.name}/${seed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      const comparison = {...testCase, name: `${testCase.name}/${seed}`};
      try {
        const pairs = makeMantelPairs(testCase.positions);
        const statistics = rig.output('float32', testCase.permutationCount + 3);
        const summary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
        rig.run(
          new GPUMantelTest({
            pairs: {
              offsets: rig.input(pairs.offsets, 'uint32'),
              neighbors: rig.input(pairs.neighbors, 'uint32'),
              weights: rig.input(new Float32Array(pairs.neighbors.length).fill(1), 'float32'),
              distances: rig.input(pairs.distances, 'float32')
            },
            times: rig.input(Float32Array.from(testCase.times), 'float32'),
            parameters: rig.input(
              getGPUSpaceTimeParameterValues({seed, permutationCount: testCase.permutationCount}),
              'uint32'
            ),
            maximumPermutations: testCase.permutationCount + 2,
            output: {statistics, summary}
          })
        );
        const actual = await rig.readFloat(statistics);
        const actualSummary = await rig.readFloat(summary);
        compareParity(
          'GPUMantelTest',
          comparison,
          'observed-r',
          [actual[0], actualSummary[0]],
          [testCase.observed, testCase.observed]
        );
        compareParity(
          'GPUMantelTest',
          comparison,
          'unused-slots',
          actual.slice(testCase.permutationCount + 1),
          [0, 0],
          'E'
        );
        const draws = actual.slice(1, testCase.permutationCount + 1);
        const mean = draws.reduce((sum, value) => sum + value, 0) / draws.length;
        const variance =
          draws.length > 1
            ? draws.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (draws.length - 1)
            : 0;
        const greater = draws.filter(value => value >= actual[0]).length;
        const lesser = draws.filter(value => value <= actual[0]).length;
        const upper = (greater + 1) / (draws.length + 1);
        const lower = (lesser + 1) / (draws.length + 1);
        compareParity(
          'GPUMantelTest',
          comparison,
          'summary-counts',
          [
            actualSummary[1],
            actualSummary[2],
            actualSummary[3],
            actualSummary[6],
            actualSummary[7]
          ],
          [pairs.neighbors.length, 0, 0, greater, lesser],
          'E'
        );
        compareParity(
          'GPUMantelTest',
          comparison,
          'summary-moments-and-tails',
          [
            actualSummary[4],
            actualSummary[5],
            actualSummary[8],
            actualSummary[9],
            actualSummary[10],
            actualSummary[11]
          ],
          [
            mean,
            variance,
            upper,
            lower,
            Math.min(1, 2 * Math.min(upper, lower)),
            variance > 0 ? (actual[0] - mean) / Math.sqrt(variance) : NaN
          ]
        );
        compareParity(
          'GPUMantelTest',
          comparison,
          'finite-in-range-draws',
          draws.map(value => (Number.isFinite(value) && Math.abs(value) <= 1 ? 1 : 0)),
          draws.map(() => 1),
          'E'
        );
        if (testCase.constant) {
          compareParity(
            'GPUMantelTest',
            comparison,
            'constant-null',
            draws,
            draws.map(() => 0),
            'E'
          );
        } else if (testCase.null.length > 0) {
          const expected = testCase.null;
          // DKW: P(sup|F_n-F| > e) <= 2 exp(-2 n e^2).
          // Union of two samples at alpha=1e-5 (per comparison). Exact null needs
          // only one bound. Across six nonconstant seed tests family risk <=6e-5.
          const alpha = 1e-5;
          const bound =
            Math.sqrt(Math.log(4 / alpha) / (2 * draws.length)) +
            (testCase.exactNull ? 0 : Math.sqrt(Math.log(4 / alpha) / (2 * expected.length)));
          const thresholds = [...new Set([...expected, ...draws])].sort(
            (first, second) => first - second
          );
          const actualCdf = thresholds.map(
            value => draws.filter(draw => draw <= value).length / draws.length
          );
          const expectedCdf = thresholds.map(
            value => expected.filter(draw => draw <= value).length / expected.length
          );
          compareParity(
            'GPUMantelTest',
            comparison,
            'full-null-empirical-cdf',
            actualCdf,
            expectedCdf,
            'P:DKW',
            bound,
            0
          );
          const referenceUpper =
            expected.filter(value => value >= testCase.observed).length / expected.length;
          // Binomial standard error of TWO independent samples, with 1/n floor
          // for zero observed counts; five sigma plus the pseudo-p resolution.
          const error = Math.sqrt(
            Math.max(referenceUpper * (1 - referenceUpper), 1 / expected.length) *
              (1 / draws.length + (testCase.exactNull ? 0 : 1 / expected.length))
          );
          compareParity(
            'GPUMantelTest',
            comparison,
            'greater-tail-independent-null',
            [upper],
            [referenceUpper],
            'P:binomial',
            5 * error + 1 / (draws.length + 1),
            0
          );
        }
      } finally {
        rig.destroy();
      }
    }, 120000);
  }
}
