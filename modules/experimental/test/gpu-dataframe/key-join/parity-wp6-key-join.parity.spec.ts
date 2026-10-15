// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUKeyJoin, type GPUKeyJoinAggregate} from '../../../src/gpu-dataframe/key-join';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-key-join.json';

type Scene = {
  name: string;
  bits: 32 | 64;
  leftKeys: string[];
  rightKeys: string[];
  values: number[];
  leftMask: number[];
  rightMask: number[];
  capacity: number;
  expected: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);

for (const scene of fixture.cases) {
  for (const kind of ['left', 'inner'] as const) {
    it(`GPUKeyJoin pandas parity: ${scene.name}/${kind}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new ParityReductionHarness(device);
      const keyView = (name: string, keys: string[]) => {
        const words = keys.flatMap(key => {
          const value = BigInt(key);
          return scene.bits === 64
            ? [Number(value & 0xffffffffn), Number(value >> 32n)]
            : [Number(value)];
        });
        return harness.input(
          name,
          Uint32Array.from(words),
          scene.bits === 64 ? 'uint32x2' : 'uint32'
        );
      };
      const leftCount = scene.leftKeys.length;
      const values = harness.input('values', Float32Array.from(scene.values), 'float32');
      const aggregates: GPUKeyJoinAggregate[] = [
        {operation: 'count', output: harness.output('countAll', leftCount, 'float32')},
        {
          operation: 'count',
          column: values,
          output: harness.output('countFinite', leftCount, 'float32')
        },
        ...(['sum', 'mean', 'minimum', 'maximum'] as const).map(operation => ({
          operation,
          column: values,
          output: harness.output(operation, leftCount, 'float32'),
          ...(operation === 'sum' || operation === 'mean'
            ? {sums: harness.output(`${operation}Words`, leftCount, 'uint32x2')}
            : {})
        }))
      ];
      harness.graph.add(
        new GPUKeyJoin({
          leftKeys: keyView('left', scene.leftKeys),
          rightKeys: keyView('right', scene.rightKeys),
          leftMask: harness.input('left-mask', Uint32Array.from(scene.leftMask), 'uint32'),
          rightMask: harness.input('right-mask', Uint32Array.from(scene.rightMask), 'uint32'),
          kind,
          sumScale: 65536,
          gather: [{column: values, output: harness.output('gathered', leftCount, 'float32')}],
          aggregates,
          output: {
            rightRows: harness.output('rightRows', leftCount, 'uint32'),
            matchCounts: harness.output('matchCounts', leftCount, 'uint32'),
            matched: harness.output('matched', leftCount, 'uint32'),
            rightMatched: harness.output('rightMatched', scene.rightKeys.length, 'uint32'),
            ...(kind === 'inner'
              ? {
                  rows: {
                    ids: harness.output('ids', scene.capacity, 'uint32'),
                    status: {
                      count: harness.output('count', 1, 'uint32'),
                      requiredCount: harness.output('requiredCount', 1, 'uint32'),
                      overflow: harness.output('overflow', 1, 'uint32')
                    }
                  }
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
          for (const [metric, observed] of Object.entries(actual)) {
            const expected = scene.expected[metric];
            const exact = !['sum', 'mean'].includes(metric);
            compareParity(
              'GPUKeyJoin',
              `${scene.name}/${kind}/encoding=${encoding}`,
              fixture.reference,
              scene.rightKeys.length,
              metric,
              metric === 'ids' ? observed.slice(0, actual['count'][0]) : observed,
              ['gathered', 'minimum', 'maximum'].includes(metric)
                ? expected.map(Math.fround)
                : expected,
              exact
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
