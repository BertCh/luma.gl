// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUTemporalReduction,
  getGPUTemporalReductionParameterValues,
  getGPUTemporalReductionWordParameterValues
} from '../../../src/gpu-dataframe/temporal-reduction';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {
  ParityReductionHarness,
  getParityEarthquakes
} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-temporal-reduction.json';

import round8Json from '../../oracles/fixtures/parity-wp6-round8-temporal.json';

type Scene = {
  name: string;
  omitMask?: boolean;
  mode: 'float' | 'word';
  source: boolean;
  cellCount: number;
  bucketCount: number;
  origin: string;
  width: string;
  capacity: number;
  cells: number[];
  timestamps: (string | number)[];
  values: number[];
  mask: number[];
  expected: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);

const round8 = decodeOracleFixture<{cases: Scene[]}>(round8Json).cases;
for (const scene of [...fixture.cases, ...round8]) {
  for (const sorted of scene.source ? [false, true] : [false]) {
    it(`GPUTemporalReduction pandas parity: ${scene.name}/sorted=${sorted}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new ParityReductionHarness(device);
      const source = scene.source ? getParityEarthquakes() : undefined;
      const values = source?.values ?? scene.values;
      const cells = source?.cells ?? scene.cells;
      const times = source
        ? source.seconds.map(seconds =>
            scene.mode === 'word' ? BigInt(seconds) * 1000n + BigInt(source.timeOriginMs) : seconds
          )
        : scene.timestamps;
      const timestamps =
        scene.mode === 'word'
          ? harness.input(
              'timestamps',
              getInt64TimeWords(BigInt64Array.from(times.map(value => BigInt(value)))),
              'uint32x2'
            )
          : harness.input('timestamps', Float32Array.from(times.map(Number)), 'float32');
      const parameters =
        scene.mode === 'word'
          ? harness.input(
              'parameters',
              getGPUTemporalReductionWordParameterValues(BigInt(scene.origin), BigInt(scene.width)),
              'uint32'
            )
          : harness.input(
              'parameters',
              getGPUTemporalReductionParameterValues(Number(scene.origin), Number(scene.width)),
              'float32'
            );
      const parameterBuffer = harness.buffers[harness.buffers.length - 1];
      const length = scene.cellCount * scene.bucketCount;
      harness.graph.add(
        new GPUTemporalReduction({
          cellIds: harness.input('cells', Uint32Array.from(cells), 'uint32'),
          timestamps,
          parameters,
          values: harness.input('values', Float32Array.from(values), 'float32'),
          mask: scene.omitMask
            ? undefined
            : harness.input(
                'mask',
                source ? new Uint32Array(values.length).fill(1) : Uint32Array.from(scene.mask),
                'uint32'
              ),
          cellCount: scene.cellCount,
          bucketCount: scene.bucketCount,
          sorted,
          output: {
            counts: harness.output('counts', length, 'uint32'),
            minimums: harness.output('minimums', length, 'float32'),
            maximums: harness.output('maximums', length, 'float32'),
            first: harness.output('first', length, 'float32'),
            last: harness.output('last', length, 'float32'),
            occupiedSlots: {
              ids: harness.output('ids', scene.capacity, 'uint32'),
              status: {
                count: harness.output('count', 1, 'uint32'),
                overflow: harness.output('overflow', 1, 'uint32'),
                requiredCount: harness.output('requiredCount', 1, 'uint32')
              }
            }
          }
        })
      );
      let compiled: ReturnType<typeof harness.graph.compile> | undefined;
      try {
        compiled = harness.graph.compile();
        // Round-8 cases share inputs/topology. Re-bin, then restore, on one compiled graph.
        const compatible = round8.filter(candidate => candidate.bucketCount === scene.bucketCount);
        const runs = round8.includes(scene)
          ? [scene, compatible[(compatible.indexOf(scene) + 1) % compatible.length], scene]
          : [scene];
        for (const run of runs) {
          parameterBuffer.write(
            scene.mode === 'word'
              ? getGPUTemporalReductionWordParameterValues(BigInt(run.origin), BigInt(run.width))
              : getGPUTemporalReductionParameterValues(Number(run.origin), Number(run.width))
          );
          submitGraph(device, compiled, undefined);
          const actual = await harness.read();
          const compare = (metric: string, observed: number[], expected: number[]) =>
            compareParity(
              'GPUTemporalReduction',
              `${scene.name}/run=${run.name}/sorted=${sorted}`,
              fixture.reference,
              values.length,
              metric,
              observed,
              expected,
              true
            );
          for (const field of ['counts', 'minimums', 'maximums', 'first', 'last']) {
            // Selection outputs retain input f32 bits. Normalize JSON/module decimal decoding to
            // that representation; this keeps E exact (one f32 ULP still fails), with no tolerance.
            const expected =
              field === 'counts' ? run.expected[field] : run.expected[field].map(Math.fround);
            compare(field, actual[field], expected);
            compare(
              `${field}.negativeZero`,
              actual[field].map(value => Number(Object.is(value, -0))),
              expected.map(value => Number(Object.is(value, -0)))
            );
          }
          compare(
            'status',
            [...actual['count'], ...actual['overflow'], ...actual['requiredCount']],
            run.expected['status']
          );
          compare('ids', actual['ids'].slice(0, actual['count'][0]), run.expected['ids']);
        }
      } finally {
        compiled?.destroy();
        harness.destroy();
      }
    }, 120000);
  }
}
