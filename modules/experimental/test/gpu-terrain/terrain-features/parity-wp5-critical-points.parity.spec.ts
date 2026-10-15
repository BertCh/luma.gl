// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTerrainCriticalPoints} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-critical-points';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  CRITICAL_CASES,
  CRITICAL_REFERENCE,
  EXACT_TOLERANCE,
  compareValues
} from '../parity-wp5-round6-fixtures';

for (const fixture of CRITICAL_CASES) {
  it(`GPUTerrainCriticalPoints independent lower-link parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = fixture.width * fixture.height;
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-critical'});
    const elevation = createInputBuffer(device, Float32Array.from(fixture.elevation));
    const validity = createInputBuffer(device, Uint32Array.from(fixture.validity));
    const classes = createOutputBuffer(device, count);
    const signChanges = createOutputBuffer(device, count);
    const counts = createOutputBuffer(device, 6);
    graph.add(
      new GPUTerrainCriticalPoints({
        grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
        connectivity: fixture.connectivity,
        elevation: {
          id: 'elevation',
          format: 'float32',
          noDataValue: -9999,
          scale: fixture.scale,
          offset: fixture.offset,
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', elevation, 'float32', count)
          },
          validity: importGraphBuffer(graph, 'validity', validity, 'uint32', count)
        },
        output: {
          classes: importGraphBuffer(graph, 'classes', classes, 'uint32', count),
          signChanges: importGraphBuffer(graph, 'sign-changes', signChanges, 'uint32', count),
          counts: importGraphBuffer(graph, 'counts', counts, 'uint32', 6)
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (let encoding = 0; encoding < 2; encoding++) {
        submitGraph(device, compiled, undefined);
        for (const [metric, buffer, expected] of [
          ['classes', classes, fixture.classes],
          ['signChanges', signChanges, fixture.signChanges],
          ['counts', counts, fixture.counts]
        ] as const) {
          compareValues(
            'GPUTerrainCriticalPoints',
            fixture.name,
            CRITICAL_REFERENCE,
            `${metric}/encoding-${encoding}`,
            await readUint32(buffer, expected.length),
            expected,
            EXACT_TOLERANCE
          );
        }
      }
    } finally {
      compiled.destroy();
      for (const buffer of [elevation, validity, classes, signChanges, counts]) buffer.destroy();
    }
  });
}
