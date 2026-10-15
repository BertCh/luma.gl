// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUBufferSelection} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import fixture from '../../oracles/fixtures/parity-wp2-buffer-selection.json';

for (const entry of fixture.cases) {
  it(`GPUBufferSelection Shapely parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queryCount = entry.frames[0].queries.length / 2;
    try {
      for (const capacity of [queryCount + 1, 1]) {
        const fixtureGraph = createParityGraph(device);
        const {graph, input, output} = fixtureGraph;
        const points = input('points', entry.frames[0].queries, 'float32x2');
        const starts = input('starts', entry.frames[0].starts, 'float32x2');
        const ends = input('ends', entry.frames[0].ends, 'float32x2');
        const radius = input('radius', [entry.radius], 'float32');
        const mask = output('mask', queryCount, 'uint32');
        const ids = output('ids', capacity, 'uint32');
        const count = output('count', 1, 'uint32');
        const requiredCount = output('required', 1, 'uint32');
        const overflow = output('overflow', 1, 'uint32');
        const joinOverflow = output('join-overflow', 1, 'uint32');
        const drawCount = output('draw-count', 1, 'uint32');
        const distances = output('distances', queryCount, 'float32');
        const nearest = output('nearest', queryCount, 'uint32');
        graph.add(
          new GPUBufferSelection({
            points: {
              kind: 'points',
              positions: points.view,
              sourceIds: input(
                'point-ids',
                Array.from({length: queryCount}, (_, index) => 10000 + index),
                'uint32'
              ).view
            },
            features: {
              kind: 'segments',
              starts: starts.view,
              ends: ends.view,
              sourceIds: input('feature-ids', entry.sourceIds, 'uint32').view
            },
            distance: radius.view,
            candidateCapacity: Math.max(...entry.frames.map(frame => frame.candidateCount)) + 1,
            output: {
              outputMask: mask.view,
              matches: {
                ids: ids.view,
                status: {
                  count: count.view,
                  requiredCount: requiredCount.view,
                  overflow: overflow.view
                }
              },
              distances: distances.view,
              nearestFeatureIds: nearest.view,
              drawInstanceCount: drawCount.view,
              status: {overflow: joinOverflow.view}
            }
          })
        );
        const compiled = graph.compile();
        try {
          for (const frame of entry.frames) {
            points.buffer.write(Float32Array.from(frame.queries));
            starts.buffer.write(Float32Array.from(frame.starts));
            ends.buffer.write(Float32Array.from(frame.ends));
            for (const threshold of [entry.radius, 0, -1, NaN, Infinity, entry.radius]) {
              radius.buffer.write(Float32Array.of(threshold));
              submitGraph(device, compiled, undefined);
              // Complete Shapely nearest tie sets suffice at every smaller radius: if the
              // nearest is outside, every feature is outside. NaN/negative/infinite radii
              // are explicitly empty under the public API contract.
              const neighbors = frame.neighbors.map(rows =>
                Number.isFinite(threshold) && threshold >= 0 && rows[0]?.distance <= threshold
                  ? rows[0]
                  : undefined
              );
              const expectedMask = neighbors.map(neighbor => Number(Boolean(neighbor)));
              const expectedIds = neighbors.flatMap((neighbor, index) =>
                neighbor ? [10000 + index] : []
              );
              const kept = Math.min(capacity, expectedIds.length);
              const caseName = `${entry.name}/${frame.name}/capacity=${capacity}/radius=${threshold}`;
              const compare = (
                metric: string,
                actual: ArrayLike<number>,
                expected: ArrayLike<number>,
                absolute = 0
              ) =>
                compareParity(
                  'GPUBufferSelection',
                  caseName,
                  fixture.reference,
                  metric,
                  actual,
                  expected,
                  absolute
                );
              compare(
                'every selection mask',
                await readUint32(mask.buffer, queryCount),
                expectedMask
              );
              compare(
                'count/required/output-overflow/join-overflow/draw-count',
                [
                  ...(await readUint32(count.buffer, 1)),
                  ...(await readUint32(requiredCount.buffer, 1)),
                  ...(await readUint32(overflow.buffer, 1)),
                  ...(await readUint32(joinOverflow.buffer, 1)),
                  ...(await readUint32(drawCount.buffer, 1))
                ],
                [kept, expectedIds.length, Number(expectedIds.length > capacity), 0, kept]
              );
              compare(
                'every compact source ID',
                await readUint32(ids.buffer, kept),
                expectedIds.slice(0, kept)
              );
              compare(
                'every nearest feature source ID and sentinel',
                await readUint32(nearest.buffer, queryCount),
                neighbors.map(neighbor => neighbor?.id ?? 0xffffffff)
              );
              compare(
                'every distance and unmatched sentinel',
                await readFloat32(distances.buffer, queryCount),
                neighbors.map(neighbor => neighbor?.distance ?? -1),
                entry.absolute
              );
            }
          }
        } finally {
          compiled.destroy();
          fixtureGraph.destroy();
        }
      }
      // Candidate scratch overflow is separate from final output capacity and invalidates
      // completeness; compare only flags, never an incomplete set to the full oracle.
      const fixtureGraph = createParityGraph(device);
      const {graph, input, output} = fixtureGraph;
      const overflow = output('join-overflow', 1, 'uint32');
      graph.add(
        new GPUBufferSelection({
          points: {
            kind: 'points',
            positions: input('points', entry.frames[0].queries, 'float32x2').view
          },
          features: {
            kind: 'segments',
            starts: input('starts', entry.frames[0].starts, 'float32x2').view,
            ends: input('ends', entry.frames[0].ends, 'float32x2').view
          },
          distance: input('radius', [entry.radius], 'float32').view,
          candidateCapacity: 1,
          output: {
            outputMask: output('mask', queryCount, 'uint32').view,
            status: {overflow: overflow.view}
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        compareParity(
          'GPUBufferSelection',
          entry.name,
          fixture.reference,
          'candidate scratch overflow',
          await readUint32(overflow.buffer, 1),
          [1]
        );
      } finally {
        compiled.destroy();
        fixtureGraph.destroy();
      }
    } finally {
      device.destroy?.();
    }
  }, 120_000);
}
