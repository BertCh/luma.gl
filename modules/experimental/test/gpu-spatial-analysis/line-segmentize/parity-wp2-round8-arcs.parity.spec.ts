// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUGreatCircleArcs} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-great-circle-arcs';
import {getGPUGreatCircleArcsParameterValues} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-parameters';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {compareBoundedParity} from '../../oracles/parity-wp2-round4-compare';
import fixture0 from '../../oracles/fixtures/parity-wp2-round8-arcs-0.json';
import fixture1 from '../../oracles/fixtures/parity-wp2-round8-arcs-1.json';
import fixture2 from '../../oracles/fixtures/parity-wp2-round8-arcs-2.json';
// Expand independent catalog rows; no values come from a GPU or local geometry oracle.
const arcsFixtures = [fixture0, fixture1, fixture2].flatMap(fixture =>
  fixture.cases.flatMap(entry => {
    const groups = new Map<number, typeof entry.settings>();
    for (const setting of entry.settings) {
      const group = groups.get(setting.maximumSegmentLength) ?? [];
      group.push(setting);
      groups.set(setting.maximumSegmentLength, group);
    }
    return Array.from(groups, ([maximumLength, settings]) => ({
      reference: fixture.reference,
      tolerance: {
        measureAbsolute: entry.units === 'kilometers' ? 0.000001 : 0.001,
        measureRelative: 2e-6
      },
      conventions: {
        sphereRadius: entry.sphereRadius,
        units: entry.units === 'kilometers' ? ('kilometers' as const) : ('meters' as const)
      },
      cases: [
        {
          name: `${entry.name}/maximum=${maximumLength}`,
          sources: entry.sources,
          targets: entry.targets,
          frames: settings.map(setting => {
            const positions: number[] = [];
            const measures: number[] = [];
            const sourceRows: number[] = [];
            const positionBounds: number[] = [];
            const lineOffsets = [0];
            setting.counts.forEach((segments, row) => {
              const reference = entry.catalog[segments - 1];
              const start = reference.lineOffsets[row];
              const stop = reference.lineOffsets[row + 1];
              positions.push(...reference.positions.slice(2 * start, 2 * stop));
              positionBounds.push(...reference.positionBounds.slice(2 * start, 2 * stop));
              measures.push(...reference.measures.slice(start, stop));
              sourceRows.push(...reference.sourceRows.slice(start, stop));
              lineOffsets.push(sourceRows.length);
            });
            return {
              ...setting,
              maximumSegmentCount: 64,
              positions,
              positionBounds,
              measures,
              sourceRows,
              lineOffsets
            };
          })
        }
      ]
    }));
  })
);
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Bound each test's readback work while retaining every live parameter update.
// R9 timed out inside a 64-frame sweep; 16 frames keep the same 120-second limit.
for (const fixture of arcsFixtures)
  for (const completeEntry of fixture.cases)
    for (let frameStart = 0; frameStart < completeEntry.frames.length; frameStart += 16) {
      const entry = {
        ...completeEntry,
        name: `${completeEntry.name}/frames=${frameStart}-${Math.min(frameStart + 16, completeEntry.frames.length) - 1}`,
        frames: completeEntry.frames.slice(frameStart, frameStart + 16)
      };
      it(`GPUGreatCircleArcs PROJ parity: ${entry.name}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const pairs = entry.sources.length / 2;
        const fullCapacity = Math.max(...entry.frames.map(frame => frame.sourceRows.length));
        for (const capacity of [fullCapacity]) {
          const rig = createParityGraph(device);
          const positions = rig.output('positions', capacity, 'float32x2');
          const offsets = rig.output('offsets', pairs + 1, 'uint32');
          const owners = rig.output('owners', capacity, 'uint32');
          const measures = rig.output('measures', capacity, 'float32');
          const sourceLines = rig.output('source-lines', pairs, 'uint32');
          const count = rig.output('count', 1, 'uint32');
          const required = rig.output('required', 1, 'uint32');
          const overflow = rig.output('overflow', 1, 'uint32');
          const lineCount = rig.output('line-count', 1, 'uint32');
          const parameters = rig.input('parameters', [0, 1, 0, 0], 'float32');
          rig.graph.add(
            new GPUGreatCircleArcs({
              sources: rig.input('sources', entry.sources, 'float32x2').view,
              targets: rig.input('targets', entry.targets, 'float32x2').view,
              parameters: parameters.view,
              maximumSegmentCount: entry.frames[0].maximumSegmentCount,
              spatialContext: {
                coordinateSpace: 'longitude-latitude',
                metric: 'great-circle',
                units: fixture.conventions.units,
                sphereRadius: fixture.conventions.sphereRadius
              },
              output: {
                positions: positions.view,
                lineOffsets: offsets.view,
                sourceRows: owners.view,
                sourceLineIndices: sourceLines.view,
                measures: measures.view,
                status: {
                  count: count.view,
                  requiredCount: required.view,
                  overflow: overflow.view,
                  lineCount: lineCount.view
                }
              }
            })
          );
          const compiled = await rig.graph.compileAsync();
          try {
            for (const frame of entry.frames) {
              parameters.buffer.write(getGPUGreatCircleArcsParameterValues(frame));
              submitGraph(device, compiled, undefined);
              const total = frame.sourceRows.length;
              const used = Math.min(capacity, total);
              const label = `${entry.name}/length=${frame.maximumSegmentLength}/minimum=${frame.minimumSegments}/capacity=${capacity}`;
              const compare = (
                metric: string,
                actual: number[],
                expected: number[],
                absolute = 0,
                relative = 0
              ) =>
                compareParity(
                  'GPUGreatCircleArcs',
                  label,
                  fixture.reference,
                  metric,
                  actual,
                  expected,
                  absolute,
                  relative
                );
              compare(
                'count/required/overflow/lineCount',
                [
                  ...(await readUint32(count.buffer, 1)),
                  ...(await readUint32(required.buffer, 1)),
                  ...(await readUint32(overflow.buffer, 1)),
                  ...(await readUint32(lineCount.buffer, 1))
                ],
                [used, total, Number(total > capacity), pairs]
              );
              compare(
                'every line offset',
                await readUint32(offsets.buffer, pairs + 1),
                frame.lineOffsets.map(value => Math.min(value, capacity))
              );
              compare(
                'every source row',
                await readUint32(owners.buffer, used),
                frame.sourceRows.slice(0, used)
              );
              compare(
                'every source line',
                await readUint32(sourceLines.buffer, pairs),
                Array.from({length: pairs}, (_, index) => index)
              );
              compareBoundedParity(
                'GPUGreatCircleArcs',
                label,
                fixture.reference,
                'every unwrapped coordinate',
                await readFloat32(positions.buffer, used * 2),
                frame.positions.slice(0, used * 2),
                frame.positionBounds.slice(0, used * 2)
              );
              compare(
                'every arc measure',
                await readFloat32(measures.buffer, used),
                frame.measures.slice(0, used),
                fixture.tolerance.measureAbsolute,
                fixture.tolerance.measureRelative
              );
            }
          } finally {
            compiled.destroy();
            rig.destroy();
          }
        }
        // getWebGPUTestDevice returns a borrowed, cached device; only release our resources.
      }, 120_000);
    }
