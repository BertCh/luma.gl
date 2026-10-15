// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPURectangleClip} from '../../../src/gpu-spatial-analysis/rectangle-clip';
import {rectangleFixtures} from '../../oracles/parity-wp2-round5-rectangle-fixtures';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of rectangleFixtures)
  for (const entry of fixture.cases) {
    it(`GPURectangleClip GEOS parity: ${entry.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const full = Math.max(...entry.frames.map(frame => frame.positions.length / 2));
      const lineCapacity = Math.max(...entry.frames.map(frame => frame.sourceLineIndices.length));
      try {
        for (const capacity of [full, 3]) {
          const rig = createParityGraph(device);
          const output = rig.output('output', capacity, 'float32x2');
          const offsets = rig.output('offsets', lineCapacity + 1, 'uint32');
          const owners = rig.output('owners', lineCapacity, 'uint32');
          const count = rig.output('count', 1, 'uint32');
          const required = rig.output('required', 1, 'uint32');
          const overflow = rig.output('overflow', 1, 'uint32');
          const lineCount = rig.output('line-count', 1, 'uint32');
          const parameters = rig.input('parameters', [0, 0, 0, 0], 'float32');
          rig.graph.add(
            new GPURectangleClip({
              geometryType: 'lines',
              positions: rig.input('positions', entry.positions, 'float32x2').view,
              lineOffsets: rig.input('input-offsets', entry.lineOffsets, 'uint32').view,
              parameters: parameters.view,
              output: {
                positions: output.view,
                lineOffsets: offsets.view,
                sourceLineIndices: owners.view,
                status: {
                  count: count.view,
                  requiredCount: required.view,
                  overflow: overflow.view,
                  lineCount: lineCount.view
                }
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            for (const frame of [...entry.frames, entry.frames[0]]) {
              parameters.buffer.write(Float32Array.from(frame.rectangle));
              submitGraph(device, compiled, undefined);
              const total = frame.positions.length / 2;
              const used = Math.min(capacity, total);
              const compare = (
                metric: string,
                actual: number[],
                expected: number[],
                absolute = 0
              ) =>
                compareParity(
                  'GPURectangleClip',
                  `${entry.name}/${frame.rectangle}/capacity=${capacity}`,
                  fixture.reference,
                  metric,
                  actual,
                  expected,
                  absolute
                );
              compare(
                'count/required/overflow/lineCount',
                [
                  ...(await readUint32(count.buffer, 1)),
                  ...(await readUint32(required.buffer, 1)),
                  ...(await readUint32(overflow.buffer, 1)),
                  ...(await readUint32(lineCount.buffer, 1))
                ],
                [used, total, Number(total > capacity), frame.sourceLineIndices.length]
              );
              compare(
                'every clipped coordinate',
                await readFloat32(output.buffer, used * 2),
                frame.positions.slice(0, used * 2),
                entry.absolute
              );
              compare(
                'every offset and unused tail',
                await readUint32(offsets.buffer, lineCapacity + 1),
                Array.from({length: lineCapacity + 1}, (_, index) =>
                  Math.min(capacity, frame.lineOffsets[index] ?? total)
                )
              );
              compare(
                'every owner and unused tail',
                await readUint32(owners.buffer, lineCapacity),
                Array.from(
                  {length: lineCapacity},
                  (_, index) => frame.sourceLineIndices[index] ?? 0xffffffff
                )
              );
            }
          } finally {
            compiled.destroy();
            rig.destroy();
          }
        }
      } finally {
        device.destroy?.();
      }
    });
  }
