// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPURasterZonalStatistics} from '../../../src/gpu-raster/raster-zonal-statistics';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {zonalCases, decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of zonalCases) {
  for (const sumOrder of ['sorted', 'atomic'] as const) {
    for (const outputNames of [
      ['cellCounts'],
      ['valueCounts', 'sums'],
      ['cellCounts', 'valueCounts', 'sums', 'maximums'],
      ['cellCounts', 'valueCounts', 'sums', 'means', 'minimums', 'maximums']
    ] as const) {
      it(`GPURasterZonalStatistics SciPy parity: ${scene.name}/${sumOrder}/${outputNames.join(',')}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const [zones, values, validity] = await Promise.all([
          decodePackedNumbers(scene.zones),
          decodePackedNumbers(scene.values),
          decodePackedNumbers(scene.validity)
        ]);
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
            grid: makeGPURasterMetadata({width: scene.width, height: scene.height}),
            zones: fixture.input('zones', 'uint32', new Uint32Array(zones)),
            values: {
              id: 'values',
              format: 'float32',
              storage: {
                kind: 'buffer',
                values: fixture.input('values', 'float32', new Float32Array(values))
              },
              validity: fixture.input('validity', 'uint32', new Uint32Array(validity)),
              scale: scene.scale,
              offset: scene.offset,
              noDataValue: scene.noDataValue
            },
            zoneCapacity: scene.zoneCapacity,
            ignoredZone: scene.ignoredZone,
            sumOrder,
            output: {
              ...Object.fromEntries(outputNames.map(name => [name, outputs[name].view])),
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
            `${scene.name}/${sumOrder}/${outputNames.join(',')}`,
            scene.reference,
            metric,
            actual,
            expected,
            profile
          );
        try {
          submitGraph(device, compiled, undefined);
          for (const name of ['cellCounts', 'valueCounts'] as const) {
            if (!outputNames.some(selected => selected === name)) continue;
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
            if (!outputNames.some(selected => selected === name)) continue;
            compare(
              name,
              await readFloat32(outputs[name].buffer, scene.zoneCapacity),
              // JSON null and string values were decoded at the fixture boundary.
              scene.expected[name],
              'S'
            );
          }
          compare('overflow', await readUint32(overflow.buffer, 1), [scene.overflow], 'E');
        } finally {
          compiled.destroy();
          fixture.destroy();
        }
      });
    }
  }
}
