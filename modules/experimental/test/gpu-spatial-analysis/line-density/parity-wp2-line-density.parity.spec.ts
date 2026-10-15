// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/parity-wp2-line-density.json';

function compareCells(
  caseName: string,
  metric: string,
  actual: number[],
  expected: number[],
  bounds: number[]
) {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    if (!Number.isFinite(difference)) {
      mismatches++;
      continue;
    }
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(expected[index]), 1e-30));
    mismatches += Number(difference > bounds[index]);
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPULineDensity',
      case: caseName,
      reference: fixture.reference,
      n: expected.length,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {
        absolute: {minimum: Math.min(...bounds), maximum: Math.max(...bounds)},
        relative: 0,
        perCell: fixture.tolerance.length
      }
    })
  );
  expect(mismatches, `${caseName}/${metric}`).toBe(0);
}

for (const entry of fixture.cases) {
  it(`GPULineDensity GEOS parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const requiredCounts: number[] = [];
    try {
      for (const capacity of [(entry.positions.length / 2) * (entry.columns + entry.rows + 1), 1]) {
        const rig = createParityGraph(device);
        const cells = entry.columns * entry.rows;
        const lengths = rig.output('lengths', cells, 'float32');
        const densities = rig.output('densities', cells, 'float32');
        const overflow = rig.output('overflow', 1, 'uint32');
        const required = rig.output('required', 1, 'uint32');
        const parameters = rig.input('parameters', entry.frames[0].parameters, 'float32');
        rig.graph.add(
          new GPULineDensity({
            positions: rig.input('positions', entry.positions, 'float32x2').view,
            lineOffsets: rig.input('offsets', entry.lineOffsets, 'uint32').view,
            columns: entry.columns,
            rows: entry.rows,
            recordCapacity: capacity,
            parameters: parameters.view,
            spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
            output: {
              lengths: lengths.view,
              densities: densities.view,
              status: {overflow: overflow.view, requiredCount: required.view}
            }
          })
        );
        const compiled = rig.graph.compile();
        try {
          for (const [frameIndex, frame] of entry.frames.entries()) {
            parameters.buffer.write(Float32Array.from(frame.parameters));
            submitGraph(device, compiled, undefined);
            const actualLengths = await readFloat32(lengths.buffer, cells);
            const actualDensities = await readFloat32(densities.buffer, cells);
            const requiredCount = (await readUint32(required.buffer, 1))[0];
            const label = `${entry.name}/frame=${frameIndex}/capacity=${capacity}`;
            const compare = (metric: string, actual: number[], expected: number[]) =>
              compareParity('GPULineDensity', label, fixture.reference, metric, actual, expected);
            compare('overflow flag', await readUint32(overflow.buffer, 1), [
              Number(capacity === 1)
            ]);
            if (capacity > 1) {
              requiredCounts.push(requiredCount);
              compare(
                'reserved records cover all positive GEOS pieces',
                [Number(requiredCount >= frame.positivePieces)],
                [1]
              );
              compareCells(
                label,
                'every cell length',
                actualLengths,
                frame.lengths,
                frame.lengthBounds
              );
              compareCells(
                label,
                'every cell density',
                actualDensities,
                frame.densities,
                frame.densityBounds
              );
              submitGraph(device, compiled, undefined);
              compare(
                'repeat encoding is bitwise identical',
                await readFloat32(lengths.buffer, cells),
                actualLengths
              );
            } else {
              compare(
                'unclamped record demand is capacity independent',
                [requiredCount],
                [requiredCounts[frameIndex]]
              );
              compare(
                'truncated cells finite and nonnegative',
                actualLengths.map(value => Number(Number.isFinite(value) && value >= 0)),
                new Array(cells).fill(1)
              );
              compare(
                'truncated lengths do not exceed complete reference',
                actualLengths.map((value, index) =>
                  Number(value <= frame.lengths[index] + frame.lengthBounds[index])
                ),
                new Array(cells).fill(1)
              );
              compareCells(
                label,
                'truncated density follows retained length',
                actualDensities,
                actualLengths.map(value => value / (frame.parameters[2] * frame.parameters[3])),
                frame.densityBounds
              );
            }
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
