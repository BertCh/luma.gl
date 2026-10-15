// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUProfilePeaks,
  getGPUProfilePeaksParameterValues
} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  PEAK_CASES,
  PEAK_REFERENCE,
  EXACT_TOLERANCE,
  compareRound5Values
} from '../parity-wp5-round5-fixtures';

for (const {fixture, capacity} of PEAK_CASES.flatMap(fixture =>
  (fixture.source.kind === 'synthetic' ? [fixture.values.length, 1] : [fixture.values.length]).map(
    capacity => ({fixture, capacity})
  )
)) {
  const caseName = `${fixture.name}/capacity-${capacity}`;
  it(`GPUProfilePeaks SciPy/composite reference: ${caseName}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = fixture.values.length;
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-profile-peaks'});
    const inputs = [
      createInputBuffer(device, Float32Array.from(fixture.values)),
      createInputBuffer(device, Uint32Array.from(fixture.validity)),
      createInputBuffer(device, Uint32Array.from(fixture.offsets))
    ];
    const outputs = {
      prominence: createOutputBuffer(device, count),
      refinedIndex: createOutputBuffer(device, count),
      refinedValue: createOutputBuffer(device, count),
      mask: createOutputBuffer(device, count),
      ids: createOutputBuffer(device, capacity),
      count: createOutputBuffer(device, 1),
      overflow: createOutputBuffer(device, 1),
      converged: createOutputBuffer(device, 1)
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'settings',
      format: 'float32',
      length: 1,
      values: getGPUProfilePeaksParameterValues(fixture)
    });
    try {
      graph.add(
        new GPUProfilePeaks({
          values: importGraphBuffer(graph, 'values', inputs[0], 'float32', count),
          validity: importGraphBuffer(graph, 'validity', inputs[1], 'uint32', count),
          offsets: importGraphBuffer(graph, 'offsets', inputs[2], 'uint32', fixture.offsets.length),
          parameters: parameters.importToGraph(graph),
          window: fixture.window,
          minimumSide: 1,
          suppressionRadius: fixture.suppressionRadius,
          wrap: fixture.wrap,
          maximumSuppressionIterations: 64,
          output: {
            prominence: importGraphBuffer(
              graph,
              'prominence',
              outputs.prominence,
              'float32',
              count
            ),
            refinedIndex: importGraphBuffer(
              graph,
              'refined-index',
              outputs.refinedIndex,
              'float32',
              count
            ),
            refinedValue: importGraphBuffer(
              graph,
              'refined-value',
              outputs.refinedValue,
              'float32',
              count
            ),
            peakMask: importGraphBuffer(graph, 'mask', outputs.mask, 'uint32', count),
            peaks: {
              ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', capacity),
              status: {
                count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
                overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
              }
            },
            status: {
              converged: importGraphBuffer(graph, 'converged', outputs.converged, 'uint32', 1)
            }
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        for (const metric of ['prominence', 'refinedIndex', 'refinedValue'] as const) {
          compareRound5Values(
            'GPUProfilePeaks',
            caseName,
            PEAK_REFERENCE,
            metric,
            await readFloat32(outputs[metric], count),
            fixture.expected[metric],
            fixture.tolerances[metric]
          );
        }
        compareRound5Values(
          'GPUProfilePeaks',
          caseName,
          PEAK_REFERENCE,
          'mask',
          await readUint32(outputs.mask, count),
          fixture.expectedMask,
          EXACT_TOLERANCE
        );
        const [peakCount] = await readUint32(outputs.count, 1);
        compareRound5Values(
          'GPUProfilePeaks',
          caseName,
          PEAK_REFERENCE,
          'count',
          [peakCount],
          [Math.min(capacity, fixture.expectedIds.length)],
          EXACT_TOLERANCE
        );
        compareRound5Values(
          'GPUProfilePeaks',
          caseName,
          PEAK_REFERENCE,
          'compact-ids',
          (await readUint32(outputs.ids, capacity)).slice(0, peakCount),
          fixture.expectedIds.slice(0, capacity),
          EXACT_TOLERANCE
        );
        for (const [metric, expected] of [
          ['overflow', Number(fixture.expectedIds.length > capacity)],
          ['converged', 1]
        ] as const) {
          compareRound5Values(
            'GPUProfilePeaks',
            caseName,
            PEAK_REFERENCE,
            metric,
            await readUint32(outputs[metric], 1),
            [expected],
            EXACT_TOLERANCE
          );
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      parameters.destroy();
      for (const buffer of [...inputs, ...Object.values(outputs)]) buffer.destroy();
    }
  });
}
