// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainPeakSnap,
  getGPUTerrainPeakSnapParameterValues
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  SNAP_CASES,
  SNAP_REFERENCE,
  DISTANCE_TOLERANCE,
  compareValues
} from '../parity-wp5-round7-fixtures';

for (const fixture of SNAP_CASES) {
  it(`GPUTerrainPeakSnap SciPy neighborhood parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const pixelCount = fixture.width * fixture.height;
    const candidateCount = fixture.candidates.length / 2;
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-snap'});
    const elevation = createInputBuffer(device, Float32Array.from(fixture.elevation));
    const validity = createInputBuffer(device, Uint32Array.from(fixture.validity));
    const candidates = createInputBuffer(device, Float32Array.from(fixture.candidates));
    const candidateHeights = fixture.candidateHeights
      ? createInputBuffer(device, Float32Array.from(fixture.candidateHeights))
      : undefined;
    const candidateRadii = fixture.candidateRadii
      ? createInputBuffer(device, Float32Array.from(fixture.candidateRadii))
      : undefined;
    const outputs = {
      positions: createOutputBuffer(device, candidateCount * 2),
      heights: createOutputBuffer(device, candidateCount),
      status: createOutputBuffer(device, candidateCount),
      snapDistance: createOutputBuffer(device, candidateCount),
      radiusClamped: createOutputBuffer(device, 1)
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8,
      values: getGPUTerrainPeakSnapParameterValues(fixture.settings, grid)
    });
    graph.add(
      new GPUTerrainPeakSnap({
        grid,
        maximumRadiusPixels: fixture.maximumRadiusPixels,
        elevation: {
          id: 'elevation',
          format: 'float32',
          scale: fixture.scale,
          offset: fixture.offset,
          noDataValue: -9999,
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', elevation, 'float32', pixelCount)
          },
          validity: importGraphBuffer(graph, 'validity', validity, 'uint32', pixelCount)
        },
        candidates: importGraphBuffer(graph, 'candidates', candidates, 'float32x2', candidateCount),
        candidateHeights: candidateHeights
          ? importGraphBuffer(
              graph,
              'candidate-heights',
              candidateHeights,
              'float32',
              candidateCount
            )
          : undefined,
        candidateRadii: candidateRadii
          ? importGraphBuffer(graph, 'candidate-radii', candidateRadii, 'float32', candidateCount)
          : undefined,
        parameters: parameters.importToGraph(graph),
        output: {
          positions: importGraphBuffer(
            graph,
            'positions',
            outputs.positions,
            'float32x2',
            candidateCount
          ),
          heights: importGraphBuffer(graph, 'heights', outputs.heights, 'float32', candidateCount),
          snapStatus: importGraphBuffer(graph, 'status', outputs.status, 'uint32', candidateCount),
          snapDistance: importGraphBuffer(
            graph,
            'distance',
            outputs.snapDistance,
            'float32',
            candidateCount
          ),
          status: {
            radiusClamped: importGraphBuffer(
              graph,
              'radius-clamped',
              outputs.radiusClamped,
              'uint32',
              1
            )
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const compare = (
        metric: string,
        actual: number[],
        expected: number[],
        tolerance?: typeof DISTANCE_TOLERANCE
      ) =>
        compareValues(
          'GPUTerrainPeakSnap',
          fixture.name,
          SNAP_REFERENCE,
          metric,
          actual,
          expected,
          tolerance
        );
      compare(
        'positions',
        await readFloat32(outputs.positions, candidateCount * 2),
        fixture.expected.positions
      );
      compare(
        'heights',
        await readFloat32(outputs.heights, candidateCount),
        fixture.expected.heights,
        fixture.expected.heightTolerance
      );
      compare('status', await readUint32(outputs.status, candidateCount), fixture.expected.status);
      compare(
        'snapDistance',
        await readFloat32(outputs.snapDistance, candidateCount),
        fixture.expected.snapDistance,
        DISTANCE_TOLERANCE
      );
      compare('radiusClamped', await readUint32(outputs.radiusClamped, 1), [
        fixture.expected.radiusClamped
      ]);
    } finally {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of [
        elevation,
        validity,
        candidates,
        candidateHeights,
        candidateRadii,
        ...Object.values(outputs)
      ])
        buffer?.destroy();
    }
  });
}
