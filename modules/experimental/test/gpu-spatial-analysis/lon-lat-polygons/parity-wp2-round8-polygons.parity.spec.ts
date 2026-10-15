// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPULonLatPolygonPreparation,
  getGPULonLatPolygonPreparationParameterValues
} from '../../../src/gpu-spatial-analysis/lon-lat-polygons';
import fixtureJson from '../../oracles/fixtures/parity-wp2-round8-polygons.json';
const lonLatFixtures = [fixtureJson];
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of lonLatFixtures)
  for (const entry of fixture.cases) {
    it(`GPULonLatPolygonPreparation GEOS parity: ${entry.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createParityGraph(device);
      const capacity = 2 * (entry.positions.length / 2 + 6 * entry.ringFlags.length);
      const positions = rig.output('output-positions', capacity, 'float32x2');
      const featureOffsets = rig.output(
        'output-features',
        entry.expected.featureOffsets.length,
        'uint32'
      );
      const polygonOffsets = rig.output(
        'output-polygons',
        entry.expected.polygonOffsets.length,
        'uint32'
      );
      const ringOffsets = rig.output('output-rings', entry.expected.ringOffsets.length, 'uint32');
      const flags = rig.output('flags', entry.ringFlags.length, 'uint32');
      const count = rig.output('count', 1, 'uint32');
      const required = rig.output('required', 1, 'uint32');
      const overflow = rig.output('overflow', 1, 'uint32');
      const invalid = rig.output('invalid', 1, 'uint32');
      const approximation = rig.output('approximation', 1, 'uint32');
      rig.graph.add(
        new GPULonLatPolygonPreparation({
          positions: rig.input('positions', entry.positions, 'float32x2').view,
          featureOffsets: rig.input('features', entry.featureOffsets, 'uint32').view,
          polygonOffsets: rig.input('polygons', entry.polygonOffsets, 'uint32').view,
          ringOffsets: rig.input('rings', entry.ringOffsets, 'uint32').view,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'none',
            edgeModel: 'great-circle'
          },
          antimeridian: entry.reject ? 'reject' : 'split',
          parameters: rig.input(
            'parameters',
            Array.from(
              getGPULonLatPolygonPreparationParameterValues('great-circle', {
                maximumSegmentDegrees: entry.maximumSegmentDegrees
              })
            ),
            'float32'
          ).view,
          output: {
            positions: positions.view,
            featureOffsets: featureOffsets.view,
            polygonOffsets: polygonOffsets.view,
            ringOffsets: ringOffsets.view,
            ringFlags: flags.view,
            status: {
              count: count.view,
              requiredCount: required.view,
              overflow: overflow.view,
              invalidCount: invalid.view,
              approximation: approximation.view
            }
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (let repeat = 0; repeat < 2; repeat++) {
          submitGraph(device, compiled, undefined);
          const compare = (metric: string, actual: number[], expected: number[], absolute = 0) =>
            compareParity(
              'GPULonLatPolygonPreparation',
              `${entry.name}/repeat=${repeat}`,
              fixture.reference,
              metric,
              actual,
              expected,
              absolute
            );
          const total = entry.expected.positions.length / 2;
          compare(
            'all status fields',
            [
              ...(await readUint32(count.buffer, 1)),
              ...(await readUint32(required.buffer, 1)),
              ...(await readUint32(overflow.buffer, 1)),
              ...(await readUint32(invalid.buffer, 1)),
              ...(await readUint32(approximation.buffer, 1))
            ],
            [total, total, 0, entry.invalidCount, 0]
          );
          compare(
            'all feature offsets',
            await readUint32(featureOffsets.buffer, entry.expected.featureOffsets.length),
            entry.expected.featureOffsets
          );
          compare(
            'all polygon offsets',
            await readUint32(polygonOffsets.buffer, entry.expected.polygonOffsets.length),
            entry.expected.polygonOffsets
          );
          compare(
            'all ring offsets',
            await readUint32(ringOffsets.buffer, entry.expected.ringOffsets.length),
            entry.expected.ringOffsets
          );
          compare(
            'all ring flags',
            await readUint32(flags.buffer, entry.ringFlags.length),
            entry.ringFlags
          );
          const actual = total ? await readFloat32(positions.buffer, total * 2) : [];
          const aligned: number[] = [];
          // GEOS may rotate the starting vertex. Only cyclic rotation is permitted:
          // orientation, every edge, every hole role and all empty ring slots stay fixed.
          for (let ring = 0; ring + 1 < entry.expected.ringOffsets.length; ring++) {
            const start = entry.expected.ringOffsets[ring];
            const end = entry.expected.ringOffsets[ring + 1];
            let closest = start;
            for (let row = start; row < end; row++) {
              const distance = (index: number) =>
                Math.hypot(
                  actual[2 * start] - entry.expected.positions[2 * index],
                  actual[2 * start + 1] - entry.expected.positions[2 * index + 1]
                );
              if (distance(row) < distance(closest)) closest = row;
            }
            for (let row = start; row < end; row++) {
              const index = start + ((closest - start + row - start) % (end - start));
              aligned.push(
                entry.expected.positions[2 * index],
                entry.expected.positions[2 * index + 1]
              );
            }
          }
          compare('every cyclic ring coordinate with winding', actual, aligned, entry.absolute);
        }
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy?.();
      }
    }, 120_000);
  }
