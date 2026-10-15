// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUCompositeScore} from '../../../src/gpu-dataframe/composite-indicators/gpu-composite-score';
import {
  getGPUCompositeScoreParameterValues,
  type GPUCompositeScoreSettings
} from '../../../src/gpu-dataframe/composite-indicators/composite-score-parameters';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-composite-score.json';

import round8MinMax from '../../oracles/fixtures/parity-wp6-round8-composite-min-max.json';
import round8ZScore from '../../oracles/fixtures/parity-wp6-round8-composite-z-score.json';
import round8Rank from '../../oracles/fixtures/parity-wp6-round8-composite-rank.json';

import round8Pca from '../../oracles/fixtures/parity-wp6-round8-composite-pca-selectors.json';

type Scene = {
  name: string;
  indicators: number[];
  indicatorCount: number;
  mask: number[];
  principal: boolean;
  settings: GPUCompositeScoreSettings;
  expected: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);

for (const scene of [
  ...fixture.cases,
  ...[round8MinMax, round8ZScore, round8Rank, round8Pca].flatMap(
    json => decodeOracleFixture<{cases: Scene[]}>(json).cases
  )
]) {
  it(`GPUCompositeScore sklearn/SciPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new ParityReductionHarness(device);
    harness.graph.add(
      new GPUCompositeScore({
        indicators: harness.input('indicators', Float32Array.from(scene.indicators), 'float32'),
        indicatorCount: scene.indicatorCount,
        mask: scene.name.startsWith('showcase')
          ? undefined
          : harness.input('mask', Uint32Array.from(scene.mask), 'uint32'),
        parameters: harness.input(
          'parameters',
          getGPUCompositeScoreParameterValues(scene.settings),
          'float32'
        ),
        enableRank: scene.name.startsWith('showcase') || scene.settings.scaler === 'rank',
        enablePrincipalComponent: scene.principal,
        output: {
          score: harness.output('score', scene.mask.length, 'float32'),
          scaled: harness.output('scaled', scene.indicators.length, 'float32'),
          columnStatistics: harness.output('columnStatistics', scene.indicatorCount * 4, 'float32'),
          ...(scene.principal
            ? {
                loadings: harness.output('loadings', scene.indicatorCount, 'float32'),
                principalComponentSummary: harness.output('principalComponentSummary', 3, 'float32')
              }
            : {})
        }
      })
    );
    let compiled: ReturnType<typeof harness.graph.compile> | undefined;
    try {
      compiled = harness.graph.compile();
      for (let encoding = 0; encoding < 2; encoding++) {
        submitGraph(device, compiled, undefined);
        const actual = await harness.read();
        for (const [metric, expected] of Object.entries(scene.expected)) {
          compareParity(
            'GPUCompositeScore',
            `${scene.name}/encoding=${encoding}`,
            fixture.reference,
            scene.mask.length,
            metric,
            actual[metric],
            expected
          );
        }
      }
    } finally {
      compiled?.destroy();
      harness.destroy();
    }
  }, 120000);
}
