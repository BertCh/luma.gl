// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-round8-weights.json';

type WeightsCase = {
  name: string;
  queries: number[];
  features: number[];
  selfJoin: boolean;
  k: number;
  radius?: number;
  absolute: number;
  expected: {offsets: number[]; neighbors: number[]; distances: number[]};
  rules: {
    weightType: 'binary' | 'inverse-distance';
    power: number;
    floor: number;
    weights: number[];
  }[];
};
const fixture = decodeOracleFixture<{reference: string; cases: WeightsCase[]}>(fixtureJson);
for (const entry of fixture.cases) {
  it(`GPUNearestFeatureWeights SciPy parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    try {
      for (const rule of entry.rules)
        for (const capacity of [entry.expected.neighbors.length + 1]) {
          const rig = new WeightsRig(device);
          const rows = entry.queries.length / 2;
          const joinCount = entry.k + Number(entry.selfJoin);
          const ids = rig.output('uint32', rows * joinCount);
          const distances = rig.output('float32', rows * joinCount);
          const counts = rig.output('uint32', rows);
          const joinOverflow = rig.output('uint32', 1);
          const overflow = rig.output('uint32', 1);
          const output = rig.weightsOutput(rows, capacity, true);
          try {
            rig.run(
              new GPUNearestFeatureJoin({
                queries: {
                  kind: 'points',
                  positions: rig.input(Float32Array.from(entry.queries), 'float32x2', rows)
                },
                features: {
                  kind: 'points',
                  positions: rig.input(
                    Float32Array.from(entry.features),
                    'float32x2',
                    entry.features.length / 2
                  )
                },
                k: joinCount,
                radius:
                  entry.radius === undefined
                    ? undefined
                    : rig.input(Float32Array.of(entry.radius), 'float32', 1),
                output: {
                  neighborIds: ids.view,
                  neighborDistances: distances.view,
                  neighborCounts: counts.view,
                  status: {overflow: joinOverflow.view}
                }
              }),
              new GPUNearestFeatureWeights({
                neighborIds: ids.view,
                neighborDistances: distances.view,
                neighborCounts: counts.view,
                slotCapacity: joinCount,
                k: entry.k,
                excludeSelf: entry.selfJoin,
                weightType: rule.weightType,
                power: rule.power,
                distanceFloor: rule.floor,
                output: {...output.spatialWeights, status: {overflow: overflow.view}}
              })
            );
            const actual = await output.read();
            const label = `${entry.name}/${rule.weightType}/power=${rule.power}/capacity=${capacity}`;
            const compare = (
              metric: string,
              measured: number[],
              expected: number[],
              absolute = 0,
              relative = 0
            ) =>
              compareParity(
                'GPUNearestFeatureWeights',
                label,
                fixture.reference,
                metric,
                measured,
                expected,
                absolute,
                relative
              );
            compare(
              'join and CSR overflow',
              [
                (await readUint32(joinOverflow.buffer, 1))[0],
                (await readUint32(overflow.buffer, 1))[0]
              ],
              [0, Number(capacity < entry.expected.neighbors.length)]
            );
            // Check the producer before CSR conversion so an invalid join radius cannot
            // masquerade as an offsets-layout failure. Self rows have one extra zero-distance slot.
            compare(
              'every input join count',
              await readUint32(counts.buffer, rows),
              entry.expected.offsets
                .slice(1)
                .map((end, row) => end - entry.expected.offsets[row] + Number(entry.selfJoin))
            );
            compare(
              'every CSR offset',
              actual.offsets,
              entry.expected.offsets.map(value => Math.min(value, capacity))
            );
            compare(
              'every neighbor ID',
              actual.neighbors,
              entry.expected.neighbors.slice(0, capacity)
            );
            compare(
              'every distance',
              actual.distances,
              entry.expected.distances.slice(0, capacity),
              entry.absolute
            );
            compare(
              'every unstandardized weight',
              actual.weights,
              rule.weights.slice(0, capacity),
              0,
              rule.weightType === 'binary' ? 0 : 2e-6
            );
          } finally {
            rig.destroy();
          }
        }
    } finally {
      device.destroy?.();
    }
  }, 120_000);
}
