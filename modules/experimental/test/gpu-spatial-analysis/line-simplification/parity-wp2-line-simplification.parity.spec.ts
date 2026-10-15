// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPULineSimplification,
  getGPULineSimplificationParameterValues
} from '../../../src/gpu-spatial-analysis/line-simplification';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {simplificationFixtures} from '../../oracles/parity-wp2-line-simplification-fixtures';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of simplificationFixtures)
  for (const entry of fixture.cases) {
    it(`GPULineSimplification GEOS parity: ${entry.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rows = entry.positions.length / 2;
      const lines = entry.lineOffsets.length - 1;
      try {
        // Every real track is complete; a second small capacity tests exact truncation on edge cases.
        for (const capacity of entry.frames.length > 2 ? [rows, 3] : [rows]) {
          const rig = createParityGraph(device);
          const positions = rig.input('positions', entry.positions, 'float32x2');
          const offsets = rig.input('offsets', entry.lineOffsets, 'uint32');
          const parameters = rig.input('parameters', [0, 0, 0, 0], 'float32');
          const importance = rig.output('importance', rows, 'float32');
          const ids = rig.output('ids', capacity, 'uint32');
          const count = rig.output('count', 1, 'uint32');
          const overflow = rig.output('overflow', 1, 'uint32');
          const required = rig.output('required', 1, 'uint32');
          const mask = rig.output('mask', rows, 'uint32');
          const lineCounts = rig.output('line-counts', lines, 'uint32');
          const lineStarts = rig.output('line-starts', lines, 'uint32');
          const converged = rig.output('converged', 1, 'uint32');
          rig.graph.add(
            new GPULineSimplification({
              positions: positions.view,
              lineOffsets: offsets.view,
              importance: importance.view,
              parameters: parameters.view,
              output: {
                ids: ids.view,
                outputMask: mask.view,
                lineCounts: lineCounts.view,
                lineStarts: lineStarts.view,
                status: {
                  converged: converged.view,
                  count: count.view,
                  overflow: overflow.view,
                  requiredCount: required.view
                }
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            for (const frame of [...entry.frames, entry.frames[0]]) {
              parameters.buffer.write(
                getGPULineSimplificationParameterValues({tolerance: frame.tolerance})
              );
              submitGraph(device, compiled, undefined);
              const compare = (metric: string, actual: number[], expected: number[]) =>
                compareParity(
                  'GPULineSimplification',
                  `${entry.name}/tolerance=${frame.tolerance}/capacity=${capacity}`,
                  fixture.reference,
                  metric,
                  actual,
                  expected
                );
              const used = Math.min(capacity, frame.ids.length);
              compare(
                'status count/required/overflow/converged',
                [
                  ...(await readUint32(count.buffer, 1)),
                  ...(await readUint32(required.buffer, 1)),
                  ...(await readUint32(overflow.buffer, 1)),
                  ...(await readUint32(converged.buffer, 1))
                ],
                [used, frame.ids.length, Number(frame.ids.length > capacity), 1]
              );
              // Exact retained rows imply exact retained coordinates and zero Hausdorff difference to GEOS.
              compare(
                'every retained source row',
                await readUint32(ids.buffer, used),
                frame.ids.slice(0, capacity)
              );
              compare('every selection mask', await readUint32(mask.buffer, rows), frame.mask);
              compare(
                'every line count',
                await readUint32(lineCounts.buffer, lines),
                frame.lineCounts
              );
              compare(
                'every line start',
                await readUint32(lineStarts.buffer, lines),
                frame.lineStarts
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
    }, 120_000);
  }
