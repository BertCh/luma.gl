// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTerrainContours} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  CONTOUR_CASES,
  CONTOUR_REFERENCE,
  EXACT_TOLERANCE,
  compareValues
} from '../parity-wp5-round6-fixtures';
import {compareContourSegments, compareBoundedContours} from './parity-wp5-contour-comparison';

for (const fixture of CONTOUR_CASES)
  for (const bounded of [false, true]) {
    it(`GPUTerrainContours skimage parity: ${fixture.name}/${bounded ? 'capacity-one' : 'full'}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const pixelCount = fixture.width * fixture.height;
      const capacity = bounded
        ? 1
        : Math.max(1, ...fixture.expected.map(level => level.segments.length));
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-contours'});
      const elevation = createInputBuffer(device, Float32Array.from(fixture.elevation));
      const validity = createInputBuffer(device, Uint32Array.from(fixture.validity));
      const overflow = createOutputBuffer(device, 1);
      const levels = new GPUParameterBuffer(device, {
        id: 'levels',
        format: 'float32',
        length: fixture.expected.length,
        values: Float32Array.from(fixture.expected, level => level.level)
      });
      const levelsView = levels.importToGraph(graph);
      const outputs = fixture.expected.map(() => ({
        vertices: createOutputBuffer(device, capacity * 4),
        count: createOutputBuffer(device, 1),
        required: createOutputBuffer(device, 1),
        overflow: createOutputBuffer(device, 1)
      }));
      graph.add(
        new GPUTerrainContours({
          grid: {
            ...makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
            pixelInterpretation: fixture.point ? 'point' : 'area'
          },
          elevation: {
            id: 'elevation',
            format: 'float32',
            noDataValue: -9999,
            scale: fixture.scale,
            offset: fixture.offset,
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', elevation, 'float32', pixelCount)
            },
            validity: importGraphBuffer(graph, 'validity', validity, 'uint32', pixelCount)
          },
          levels: outputs.map((output, index) => ({
            level: graph.createDataView(levelsView.buffer, {
              format: 'float32',
              length: 1,
              byteOffset: index * 4
            }),
            vertices: importGraphBuffer(
              graph,
              `vertices-${index}`,
              output.vertices,
              'float32x2',
              capacity * 2
            ),
            status: {
              count: importGraphBuffer(graph, `count-${index}`, output.count, 'uint32', 1),
              requiredCount: importGraphBuffer(
                graph,
                `required-${index}`,
                output.required,
                'uint32',
                1
              ),
              overflow: importGraphBuffer(graph, `overflow-${index}`, output.overflow, 'uint32', 1)
            }
          })),
          output: {status: {overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)}}
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        for (const [index, output] of outputs.entries()) {
          const reference = fixture.expected[index];
          const [count] = await readUint32(output.count, 1);
          compareValues(
            'GPUTerrainContours',
            fixture.name,
            CONTOUR_REFERENCE,
            `level-${index}/capacity-${capacity}/status`,
            [
              count,
              ...(await readUint32(output.required, 1)),
              ...(await readUint32(output.overflow, 1))
            ],
            [
              Math.min(capacity, reference.segments.length),
              reference.segments.length,
              Number(reference.segments.length > capacity)
            ],
            EXACT_TOLERANCE
          );
          (bounded ? compareBoundedContours : compareContourSegments)(
            fixture,
            reference,
            count ? await readFloat32(output.vertices, Math.min(count, capacity) * 4) : []
          );
        }
        compareValues(
          'GPUTerrainContours',
          fixture.name,
          CONTOUR_REFERENCE,
          `capacity-${capacity}/aggregate-overflow`,
          await readUint32(overflow, 1),
          [Number(fixture.expected.some(level => level.segments.length > capacity))],
          EXACT_TOLERANCE
        );
        // The same compiled graph must clear all statuses when GPU levels change.
        levels.write(new Float32Array(fixture.expected.length).fill(1e20));
        submitGraph(device, compiled, undefined);
        const cleared = [...(await readUint32(overflow, 1))];
        for (const output of outputs)
          for (const buffer of [output.count, output.required, output.overflow])
            cleared.push(...(await readUint32(buffer, 1)));
        compareValues(
          'GPUTerrainContours',
          fixture.name,
          CONTOUR_REFERENCE,
          `capacity-${capacity}/empty-frame-reset`,
          cleared,
          cleared.map(() => 0),
          EXACT_TOLERANCE
        );
      } finally {
        compiled.destroy();
        levels.destroy();
        for (const buffer of [
          elevation,
          validity,
          overflow,
          ...outputs.flatMap(output => Object.values(output))
        ])
          buffer.destroy();
      }
    });
  }
