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

for (const encoded of streamlineCases) {
  it(`GPUStreamlines independent ODE and spacing parity: ${encoded.name}`, async () => {
    const scene = await decodeStreamlineCase(encoded);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = new ParityRasterGraph(device, scene.name);
    const {graph} = fixture;
    const {config} = scene;
    const seedCount = config.seedColumns * config.seedRows;
    const stride = 2 * config.stepsPerDirection + 1;
    const ids = fixture.output('ids', 'uint32', config.lineCapacity);
    const count = fixture.output('count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const requiredCount = fixture.output('required-count', 'uint32', 1);
    const offsets = fixture.output('offsets', 'uint32', config.lineCapacity + 1);
    const points = fixture.output('points', 'float32x2', config.pointCapacity);
    const pointCount = fixture.output('point-count', 'uint32', 1);
    const unconverged = fixture.output('unconverged', 'uint32', 1);
    const candidates = fixture.output('candidates', 'float32x2', seedCount * stride);
    const spans = fixture.output('spans', 'uint32', seedCount * 4);
    graph.add(
      new GPUStreamlines({
        fieldWidth: scene.fieldWidth,
        fieldHeight: scene.fieldHeight,
        gridWidth: config.gridWidth,
        gridHeight: config.gridHeight,
        seedColumns: config.seedColumns,
        seedRows: config.seedRows,
        stepsPerDirection: config.stepsPerDirection,
        maximumIterations: 64,
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
            status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view}
          },
          lineOffsets: offsets.view,
          positions: points.view,
          positionCount: pointCount.view,
          status: {iterationLimitReached: unconverged.view},
          candidates: {positions: candidates.view, spans: spans.view}
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
        [scene.ids.length, scene.ids.length, scene.points.length / 2, 0, 0],
        'E'
      );
      compare(
        'all-candidates-and-invalid-mask',
        await readFloat32(candidates.buffer, seedCount * stride * 2),
        scene.candidates,
        'L'
      );
      const actualSpans = await readUint32(spans.buffer, seedCount * 4);
      compare(
        'all-traced-spans',
        actualSpans.filter((_, index) => index % 4 < 2),
        scene.spans,
        'E'
      );
      compare(
        'accepted-kept-spans',
        scene.ids.flatMap(seed => actualSpans.slice(4 * seed + 2, 4 * seed + 4)),
        scene.ids.flatMap(seed => scene.kept.slice(2 * seed, 2 * seed + 2)),
        'E'
      );
      compare('all-published-ids', await readUint32(ids.buffer, scene.ids.length), scene.ids, 'E');
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
