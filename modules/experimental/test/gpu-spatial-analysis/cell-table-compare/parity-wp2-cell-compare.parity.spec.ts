// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPUCellTableCompare} from '../../../src/gpu-spatial-analysis/cell-table-compare';
import {cellFixtures} from '../../oracles/parity-wp2-cells-fixtures';
import {createCellRig, writeCellTable} from '../../oracles/parity-wp2-cells-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of cellFixtures)
  for (const measure of ['count', 'sum'] as const)
    for (const bounded of [false, true])
      it(`GPUCellTableCompare pandas ${fixture.expected.name}/${measure}/bounded=${bounded}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rig = createCellRig(device);
        const capacity = bounded ? 3 : fixture.expected.capacity;
        try {
          const beforeTable = rig.table('before', fixture.expected.capacity);
          const afterTable = rig.table('after', fixture.expected.capacity);
          const cells = rig.output('cells', capacity, 'uint32x2');
          const presence = rig.output('presence', capacity, 'uint32');
          const mask = rig.output('mask', capacity, 'uint32');
          const columns = {
            before: rig.output('comparison-before-values', capacity, 'float32'),
            after: rig.output('comparison-after-values', capacity, 'float32'),
            delta: rig.output('delta', capacity, 'float32'),
            ratio: rig.output('ratio', capacity, 'float32'),
            percentChange: rig.output('percent', capacity, 'float32'),
            zScore: rig.output('score', capacity, 'float32')
          };
          const status = {
            count: rig.output('count', 1, 'uint32'),
            requiredCount: rig.output('required', 1, 'uint32'),
            overflow: rig.output('overflow', 1, 'uint32')
          };
          rig.graph.add(
            new GPUCellTableCompare({
              before: beforeTable.view,
              after: afterTable.view,
              measure,
              sumScale: 65536,
              output: {
                cells: cells.view,
                presence: presence.view,
                mask: mask.view,
                before: columns.before.view,
                after: columns.after.view,
                delta: columns.delta.view,
                ratio: columns.ratio.view,
                percentChange: columns.percentChange.view,
                zScore: columns.zScore.view,
                status: {
                  count: status.count.view,
                  requiredCount: status.requiredCount.view,
                  overflow: status.overflow.view
                }
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            const comparisons = fixture.comparison.comparisons.filter(
              entry =>
                entry.measure === measure && (bounded ? entry.capacity === 3 : entry.capacity !== 3)
            );
            // Empty/empty has capacity 3 in both generator modes; include it explicitly for full graphs.
            if (!bounded)
              comparisons.push(
                fixture.comparison.comparisons.find(
                  entry => entry.measure === measure && entry.before === 3 && entry.after === 3
                )!
              );
            for (const [iteration, entry] of [
              ...comparisons,
              comparisons[0],
              comparisons[0]
            ].entries()) {
              const inheritedOverflow = Number(iteration === comparisons.length);
              writeCellTable(
                beforeTable,
                fixture.expected.frames[entry.before].levels[0],
                inheritedOverflow
              );
              writeCellTable(afterTable, fixture.expected.frames[entry.after].levels[0]);
              submitGraph(device, compiled, undefined);
              const expected = entry.expected;
              const label = `${fixture.expected.name}/${entry.before}->${entry.after}/${measure}/capacity=${capacity}/inherited=${inheritedOverflow}`;
              const compare = (
                metric: string,
                actual: number[],
                wanted: number[],
                absolute = 0,
                relative = 0
              ) =>
                compareParity(
                  'GPUCellTableCompare',
                  label,
                  fixture.comparison.reference,
                  metric,
                  actual,
                  wanted,
                  absolute,
                  relative
                );
              for (const field of ['count', 'requiredCount', 'overflow'] as const)
                compare(field, await readUint32(status[field].buffer, 1), [
                  field === 'overflow'
                    ? Number(expected.overflow || inheritedOverflow)
                    : expected[field]
                ]);
              const tail = capacity - expected.count;
              compare('every key and tail', await readUint32(cells.buffer, capacity * 2), [
                ...expected.cells,
                ...Array(tail * 2).fill(0xffffffff)
              ]);
              compare('presence and tail', await readUint32(presence.buffer, capacity), [
                ...expected.presence,
                ...Array(tail).fill(0)
              ]);
              compare('mask and tail', await readUint32(mask.buffer, capacity), [
                ...Array(expected.count).fill(1),
                ...Array(tail).fill(0)
              ]);
              for (const field of [
                'before',
                'after',
                'delta',
                'ratio',
                'percentChange',
                'zScore'
              ] as const) {
                const values = [
                  ...expected[field],
                  ...Array(tail).fill(field === 'before' || field === 'after' ? 0 : NaN)
                ];
                // Counts and count differences fit exactly in f32 for these fixtures.
                const exact = measure === 'count' && ['before', 'after', 'delta'].includes(field);
                compare(
                  `every ${field} and nonfinite mask`,
                  await readFloat32(columns[field].buffer, capacity),
                  values,
                  exact ? 0 : 1e-6,
                  exact ? 0 : 1e-4
                );
              }
            }
          } finally {
            compiled.destroy();
          }
        } finally {
          rig.destroy();
          device.destroy?.();
        }
      }, 120_000);
