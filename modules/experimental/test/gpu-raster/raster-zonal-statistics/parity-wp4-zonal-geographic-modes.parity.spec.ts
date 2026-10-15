// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPURasterZonalStatistics,
  getGPURasterZonalStatisticsParameterValues
} from '../../../src/gpu-raster/raster-zonal-statistics';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-zonal-modes.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
type GeographicCase = Omit<
  (typeof fixtureJSON.cases)[number],
  'rowDirection' | 'origin' | 'cellSize' | 'expected'
> & {
  rowDirection: 'north' | 'south';
  origin: [number, number];
  cellSize: [number, number];
  expected: Record<keyof (typeof fixtureJSON.cases)[number]['expected'], number[]>;
};
const fixtureJSONDecoded = decodeOracleFixture<{reference: string; cases: GeographicCase[]}>(
  fixtureJSON
);
const zonalCases = fixtureJSONDecoded.cases;
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of zonalCases) {
  for (const sumOrder of ['sorted', 'atomic'] as const) {
    it(`GPURasterZonalStatistics SciPy parity: ${scene.name}/${sumOrder}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {zones, values, validity} = scene;
      const grid = makeGPURasterMetadata(scene);
      const fixture = new ParityRasterGraph(device, scene.name);
      const {graph} = fixture;
      const outputs = {
        cellCounts: fixture.output('cell-counts', 'uint32', scene.zoneCapacity),
        valueCounts: fixture.output('value-counts', 'uint32', scene.zoneCapacity),
        sums: fixture.output('sums', 'float32', scene.zoneCapacity),
        means: fixture.output('means', 'float32', scene.zoneCapacity),
        minimums: fixture.output('minimums', 'float32', scene.zoneCapacity),
        maximums: fixture.output('maximums', 'float32', scene.zoneCapacity),
        weightSums: fixture.output('weight-sums', 'float32', scene.zoneCapacity),
        variances: fixture.output('variances', 'float32', scene.zoneCapacity),
        standardDeviations: fixture.output('standard-deviations', 'float32', scene.zoneCapacity)
      };
      const overflow = fixture.output('overflow', 'uint32', 1);
      graph.add(
        new GPURasterZonalStatistics({
          grid,
          cellSizeMode: 'geographic',
          parameters: fixture.input(
            'parameters',
            'float32',
            getGPURasterZonalStatisticsParameterValues(grid)
          ),
          zones: fixture.input('zones', 'uint32', new Uint32Array(zones)),
          values: {
            id: 'values',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: fixture.input('values', 'float32', new Float32Array(values))
            },
            validity: fixture.input('validity', 'uint32', new Uint32Array(validity)),
            scale: 0.5,
            offset: 2,
            noDataValue: -9999
          },
          zoneCapacity: scene.zoneCapacity,
          sumOrder,
          output: {
            cellCounts: outputs.cellCounts.view,
            valueCounts: outputs.valueCounts.view,
            sums: outputs.sums.view,
            means: outputs.means.view,
            minimums: outputs.minimums.view,
            maximums: outputs.maximums.view,
            weightSums: outputs.weightSums.view,
            variances: outputs.variances.view,
            standardDeviations: outputs.standardDeviations.view,
            status: {overflow: overflow.view}
          }
        })
      );
      const compiled = graph.compile();
      const compare = (
        metric: string,
        actual: ArrayLike<number>,
        expected: ArrayLike<number>,
        profile: 'E' | 'S'
      ) =>
        compareRasterParity(
          'GPURasterZonalStatistics',
          `${scene.name}/${sumOrder}`,
          fixtureJSONDecoded.reference,
          metric,
          actual,
          expected,
          profile
        );
      try {
        submitGraph(device, compiled, undefined);
        for (const name of ['cellCounts', 'valueCounts'] as const) {
          compare(
            name,
            await readUint32(outputs[name].buffer, scene.zoneCapacity),
            scene.expected[name],
            'E'
          );
        }
        for (const name of [
          'sums',
          'means',
          'minimums',
          'maximums',
          'weightSums',
          'variances',
          'standardDeviations'
        ] as const) {
          compare(
            name,
            await readFloat32(outputs[name].buffer, scene.zoneCapacity),
            // JSON null and string values were decoded at the fixture boundary.
            scene.expected[name],
            'S'
          );
        }
        compare('overflow', await readUint32(overflow.buffer, 1), [0], 'E');
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
