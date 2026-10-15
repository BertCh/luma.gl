// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {compareBoundedParity} from '../../oracles/parity-wp2-round4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/parity-wp2-line-length-per-polygon.json';

for (const entry of fixture.cases) {
  it(`GPULineLengthPerPolygon GEOS parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const features = entry.polygons.featureOffsets.length - 1;
    try {
      // GEOS counts nondegenerate segment candidates; reserve additional slots for
      // GPU bounds of stationary segments before exact filtering; terminal rows are invalid.
      const fullCapacity = entry.candidateCount + entry.stationarySegmentCount * features + 1;
      for (const pairCapacity of [fullCapacity, 1]) {
        const rig = createParityGraph(device);
        const lengths = rig.output('lengths', features, 'float32');
        const weighted = rig.output('weighted', features, 'float32');
        const counts = rig.output('counts', features, 'uint32');
        const overflow = rig.output('overflow', 1, 'uint32');
        rig.graph.add(
          new GPULineLengthPerPolygon({
            positions: rig.input('positions', entry.positions, 'float32x2').view,
            lineOffsets: rig.input('line-offsets', entry.lineOffsets, 'uint32').view,
            lineWeights: rig.input('weights', entry.lineWeights, 'float32').view,
            polygons: {
              kind: 'polygons',
              positions: rig.input('polygon-positions', entry.polygons.positions, 'float32x2').view,
              ringOffsets: rig.input('rings', entry.polygons.ringOffsets, 'uint32').view,
              polygonOffsets: rig.input('polygons', entry.polygons.polygonOffsets, 'uint32').view,
              featureOffsets: rig.input('features', entry.polygons.featureOffsets, 'uint32').view
            },
            pairCapacity,
            output: {
              lengths: lengths.view,
              weightedLengths: weighted.view,
              segmentCounts: counts.view,
              status: {overflow: overflow.view}
            }
          })
        );
        const compiled = rig.graph.compile();
        try {
          let previous: number[][] | undefined;
          for (let encoding = 0; encoding < 2; encoding++) {
            submitGraph(device, compiled, undefined);
            const label = `${entry.name}/capacity=${pairCapacity}/encoding=${encoding}`;
            const compare = (metric: string, actual: number[], expected: number[]) =>
              compareParity(
                'GPULineLengthPerPolygon',
                label,
                fixture.reference,
                metric,
                actual,
                expected
              );
            compare('overflow', await readUint32(overflow.buffer, 1), [
              Number(pairCapacity < entry.candidateCount)
            ]);
            const actualLengths = await readFloat32(lengths.buffer, features);
            const actualWeighted = await readFloat32(weighted.buffer, features);
            const actualCounts = await readUint32(counts.buffer, features);
            if (pairCapacity > 1) {
              compareBoundedParity(
                'GPULineLengthPerPolygon',
                label,
                fixture.reference,
                'every length',
                actualLengths,
                entry.expected.lengths,
                entry.lengthBounds
              );
              compareBoundedParity(
                'GPULineLengthPerPolygon',
                label,
                fixture.reference,
                'every weighted length',
                actualWeighted,
                entry.expected.weightedLengths,
                entry.weightedBounds
              );
              compare(
                'every contributing segment count',
                actualCounts,
                entry.expected.segmentCounts
              );
            } else {
              // Truncated candidate order is not a public contract. Check all rows against
              // independently bounded full sums, preserving exact zero for empty features.
              compare(
                'every truncated length is finite nonnegative and bounded',
                actualLengths.map((value, index) =>
                  Number(
                    Number.isFinite(value) &&
                      value >= 0 &&
                      value <= entry.expected.lengths[index] + entry.lengthBounds[index]
                  )
                ),
                Array(features).fill(1)
              );
              compare(
                'every truncated weighted length is finite nonnegative and bounded',
                actualWeighted.map((value, index) =>
                  Number(
                    Number.isFinite(value) &&
                      value >= 0 &&
                      value <= entry.expected.weightedLengths[index] + entry.weightedBounds[index]
                  )
                ),
                Array(features).fill(1)
              );
              compare(
                'every truncated count is bounded',
                actualCounts.map((value, index) =>
                  Number(value <= entry.expected.segmentCounts[index])
                ),
                Array(features).fill(1)
              );
            }
            if (previous) {
              compare('repeat lengths', actualLengths, previous[0]);
              compare('repeat weighted lengths', actualWeighted, previous[1]);
              compare('repeat counts', actualCounts, previous[2]);
            }
            previous = [actualLengths, actualWeighted, actualCounts];
          }
        } finally {
          compiled.destroy();
          rig.destroy();
        }
      }
    } finally {
      device.destroy?.();
    }
  }, 120_000);
}
