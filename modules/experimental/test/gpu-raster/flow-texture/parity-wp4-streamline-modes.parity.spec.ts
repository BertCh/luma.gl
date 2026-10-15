// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUStreamlines,
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/gpu-raster/flow-texture';
import {streamlineCases, decodeStreamlineCase} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

import first from '../../oracles/fixtures/parity-wp4-streamline-modes-0.json';
import second from '../../oracles/fixtures/parity-wp4-streamline-modes-99.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
const additional = [first, second].flatMap(
  fixture => decodeOracleFixture<{cases: typeof streamlineCases}>(fixture).cases
);
for (const encoded of [...streamlineCases, ...additional]) {
  for (const capacityMode of ['full', 'lines', 'points']) {
    it(`GPUStreamlines showcase 16 rounds/compact-only independent parity: ${encoded.name}`, async () => {
      const original = await decodeStreamlineCase(encoded);
      const lineCapacity = capacityMode === 'lines' ? 1 : original.config.lineCapacity;
      const pointCapacity =
        capacityMode === 'points'
          ? Math.max(1, Math.floor(original.points.length / 4))
          : original.config.pointCapacity;
      let published = Math.min(original.ids.length, lineCapacity);
      while (published > 0 && original.offsets[published] > pointCapacity) published--;
      const scene = {
        ...original,
        name: `${original.name}/${capacityMode}`,
        config: {...original.config, lineCapacity, pointCapacity},
        ids: original.ids.slice(0, published),
        offsets: original.offsets.slice(0, published + 1),
        points: original.points.slice(0, original.offsets[published] * 2)
      };
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = new ParityRasterGraph(device, scene.name);
      const {graph} = fixture;
      const {config} = scene;
      const ids = fixture.output('ids', 'uint32', config.lineCapacity);
      const count = fixture.output('count', 'uint32', 1);
      const overflow = fixture.output('overflow', 'uint32', 1);
      const requiredCount = fixture.output('required-count', 'uint32', 1);
      const offsets = fixture.output('offsets', 'uint32', config.lineCapacity + 1);
      const points = fixture.output('points', 'float32x2', config.pointCapacity);
      const pointCount = fixture.output('point-count', 'uint32', 1);
      const unconverged = fixture.output('unconverged', 'uint32', 1);
      graph.add(
        new GPUStreamlines({
          fieldWidth: scene.fieldWidth,
          fieldHeight: scene.fieldHeight,
          gridWidth: config.gridWidth,
          gridHeight: config.gridHeight,
          seedColumns: config.seedColumns,
          seedRows: config.seedRows,
          stepsPerDirection: config.stepsPerDirection,
          maximumIterations: 16,
          velocities: fixture.input('field', 'float32x2', new Float32Array(scene.field)),
          parameters: fixture.input(
            'parameters',
            'float32',
            getGPUStreamlinesParameterValues(scene.settings)
          ),
          uint32Parameters: fixture.input(
            'words',
            'uint32',
            getGPUStreamlinesUint32ParameterValues(scene.settings)
          ),
          output: {
            lines: {
              ids: ids.view,
              status: {
                count: count.view,
                overflow: overflow.view,
                requiredCount: requiredCount.view
              }
            },
            lineOffsets: offsets.view,
            positions: points.view,
            positionCount: pointCount.view,
            status: {iterationLimitReached: unconverged.view}
          }
        })
      );
      const compiled = graph.compile();
      const compare = (
        metric: string,
        actual: ArrayLike<number>,
        expected: ArrayLike<number>,
        profile: 'E' | 'L'
      ) =>
        compareRasterParity(
          'GPUStreamlines',
          scene.name,
          scene.reference,
          metric,
          actual,
          expected,
          profile
        );
      try {
        submitGraph(device, compiled, undefined);
        compare(
          'counts-overflow-convergence',
          [
            ...(await readUint32(count.buffer, 1)),
            ...(await readUint32(requiredCount.buffer, 1)),
            ...(await readUint32(pointCount.buffer, 1)),
            ...(await readUint32(overflow.buffer, 1)),
            ...(await readUint32(unconverged.buffer, 1))
          ],
          [
            scene.ids.length,
            original.ids.length,
            scene.points.length / 2,
            Number(scene.ids.length < original.ids.length),
            0
          ],
          'E'
        );
        compare(
          'all-published-ids',
          await readUint32(ids.buffer, scene.ids.length),
          scene.ids,
          'E'
        );
        compare(
          'all-csr-offsets',
          await readUint32(offsets.buffer, scene.offsets.length),
          scene.offsets,
          'E'
        );
        compare(
          'all-published-points',
          await readFloat32(points.buffer, scene.points.length),
          scene.points,
          'L'
        );
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
