// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUCellAggregation,
  GPUCellPyramid
} from '../../../src/gpu-spatial-analysis/cell-aggregation';
import {cellFixtures} from '../../oracles/parity-wp2-cells-fixtures';
import {compareCellTable, createCellRig} from '../../oracles/parity-wp2-cells-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of cellFixtures)
  for (const pyramid of [false, true])
    for (const bounded of [false, true])
      for (const wordOrder of ['little-endian', 'high-low'] as const) {
        const className = pyramid ? 'GPUCellPyramid' : 'GPUCellAggregation';
        it(`${className} H3/pandas ${fixture.expected.name} bounded=${bounded} ${wordOrder}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const rig = createCellRig(device);
          try {
            const {input, expected} = fixture;
            const capacity = bounded ? 3 : expected.capacity;
            const frames = bounded ? expected.boundedFrames : expected.frames;
            const keys =
              wordOrder === 'little-endian'
                ? input.keys
                : input.keys.map((_, index) => input.keys[index ^ 1]);
            const cells = rig.input('keys', keys, 'uint32x2');
            const values = rig.input('values', input.values, 'float32');
            const mask = rig.input('mask', input.masks[0], 'uint32');
            const levels = Array.from({length: pyramid ? 3 : 1}, (_, index) =>
              rig.table(`level-${index}`, capacity)
            );
            const levelCounts = rig.output('level-counts', levels.length, 'uint32');
            const properties = {
              family: 'h3' as const,
              cells: cells.view,
              values: values.view,
              mask: mask.view,
              wordOrder,
              sumScale: 65536
            };
            rig.graph.add(
              pyramid
                ? new GPUCellPyramid({
                    ...properties,
                    resolutions: [9, 7, 5],
                    output: {levels: levels.map(level => level.view), levelCounts: levelCounts.view}
                  })
                : new GPUCellAggregation({...properties, resolution: 9, output: levels[0].view})
            );
            const compiled = rig.graph.compile();
            try {
              // Entire input remains resident; encode nonempty, changing, empty, restored, repeat.
              for (const frameIndex of [0, 1, 2, 3, 0, 0]) {
                mask.buffer.write(Uint32Array.from(input.masks[frameIndex]));
                submitGraph(device, compiled, undefined);
                const label = `${expected.name}/${frames[frameIndex].name}/capacity=${capacity}/${wordOrder}`;
                for (const [levelIndex, table] of levels.entries())
                  await compareCellTable(
                    className,
                    `${label}/resolution=${[9, 7, 5][levelIndex]}`,
                    expected.reference,
                    table,
                    frames[frameIndex].levels[levelIndex]
                  );
                if (pyramid)
                  compareParity(
                    className,
                    label,
                    expected.reference,
                    'packed level counts',
                    await readUint32(levelCounts.buffer, levels.length),
                    frames[frameIndex].levels.map(level => level.count)
                  );
              }
            } finally {
              compiled.destroy();
            }
          } finally {
            rig.destroy();
            device.destroy?.();
          }
        }, 120_000);
      }
