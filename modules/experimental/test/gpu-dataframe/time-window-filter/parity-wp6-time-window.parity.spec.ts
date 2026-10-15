// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUTimeWindowFilter,
  getGPUTimeWindowFilterParameterValues,
  getGPUTimeWindowFilterWordParameterValues,
  getInt64TimeWords
} from '../../../src/gpu-dataframe/time-window-filter';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {
  ParityReductionHarness,
  decodeParityPayload,
  getParityEarthquakes
} from '../../oracles/parity-wp6-reduction-harness';
import realFixtureJson from '../../oracles/fixtures/parity-wp6-time-window-real.json';
import edgeFixtureJson from '../../oracles/fixtures/parity-wp6-time-window-edges.json';

type Payload = {base64: string; dtype: string; length: number};
type Scene = {
  name: string;
  mode: 'float' | 'word';
  real: boolean;
  values: (number | string)[];
  start: number | string;
  end: number | string;
  capacity: number;
  rowCount: number;
  sortedEligible: boolean;
  expected: {
    mask: Payload;
    ids: Payload;
    count: number[];
    requiredCount: number[];
    overflow: number[];
  };
};
const fixtures = [realFixtureJson, edgeFixtureJson].map(json =>
  decodeOracleFixture<{reference: string; cases: Scene[]}>(json)
);
for (const fixture of fixtures) {
  for (const scene of fixture.cases) {
    for (const sorted of scene.sortedEligible ? [false, true] : [false]) {
      it(`GPUTimeWindowFilter pandas parity: ${scene.name}/sorted=${sorted}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const harness = new ParityReductionHarness(device);
        const source = scene.real ? getParityEarthquakes() : undefined;
        const timestamps =
          scene.mode === 'word'
            ? harness.input(
                'timestamps',
                getInt64TimeWords(
                  BigInt64Array.from(
                    source
                      ? source.seconds.map(
                          second => BigInt(second) * 1000n + BigInt(source.timeOriginMs)
                        )
                      : scene.values.map(value => BigInt(value))
                  )
                ),
                'uint32x2'
              )
            : harness.input(
                'timestamps',
                Float32Array.from(source?.seconds ?? scene.values.map(Number)),
                'float32'
              );
        const window =
          scene.mode === 'word'
            ? harness.input(
                'window',
                getGPUTimeWindowFilterWordParameterValues({
                  start: BigInt(scene.start),
                  end: BigInt(scene.end)
                }),
                'uint32'
              )
            : harness.input(
                'window',
                getGPUTimeWindowFilterParameterValues({
                  start: Number(scene.start),
                  end: Number(scene.end)
                }),
                'float32'
              );
        harness.graph.add(
          new GPUTimeWindowFilter({
            timestamps,
            window,
            sorted,
            validateOrder: sorted,
            output: {
              ids: harness.output('ids', scene.capacity, 'uint32'),
              status: {
                count: harness.output('count', 1, 'uint32'),
                requiredCount: harness.output('requiredCount', 1, 'uint32'),
                overflow: harness.output('overflow', 1, 'uint32')
              },
              outputMask: harness.output('mask', scene.rowCount, 'uint32'),
              fadeWeights: harness.output('fadeWeights', scene.rowCount, 'float32'),
              drawInstanceCount: harness.output('drawInstanceCount', 1, 'uint32'),
              ...(sorted ? {orderViolations: harness.output('orderViolations', 1, 'uint32')} : {})
            }
          })
        );
        const expected: Record<string, number[]> = {
          ...scene.expected,
          mask: decodeParityPayload(scene.expected.mask),
          ids: decodeParityPayload(scene.expected.ids),
          fadeWeights: decodeParityPayload(scene.expected.mask),
          drawInstanceCount: scene.expected.count,
          orderViolations: [0]
        };
        let compiled: ReturnType<typeof harness.graph.compile> | undefined;
        try {
          compiled = harness.graph.compile();
          for (let encoding = 0; encoding < 2; encoding++) {
            submitGraph(device, compiled, undefined);
            const actual = await harness.read();
            for (const [metric, observed] of Object.entries(actual)) {
              compareParity(
                'GPUTimeWindowFilter',
                `${scene.name}/sorted=${sorted}/encoding=${encoding}`,
                fixture.reference,
                scene.rowCount,
                metric,
                metric === 'ids' ? observed.slice(0, actual['count'][0]) : observed,
                expected[metric],
                true
              );
            }
          }
        } finally {
          compiled?.destroy();
          harness.destroy();
        }
      }, 120000);
    }
  }
}
