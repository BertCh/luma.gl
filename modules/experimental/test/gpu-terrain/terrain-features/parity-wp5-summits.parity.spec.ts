// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainSummits,
  getGPUTerrainSummitsParameterValues
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {SUMMITS_CASES, SUMMITS_REFERENCE, compareValues} from '../parity-wp5-round7-fixtures';

for (const fixture of SUMMITS_CASES) {
  for (const capacity of new Set([fixture.width * fixture.height, 1])) {
    it(`GPUTerrainSummits SciPy morphology parity: ${fixture.name}/capacity-${capacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const pixelCount = fixture.width * fixture.height;
      const grid = makeGPURasterMetadata({
        width: fixture.width,
        height: fixture.height,
        cellSize: fixture.cellSize
      });
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-summits'});
      const elevation = createInputBuffer(device, Float32Array.from(fixture.elevation));
      const validity = createInputBuffer(device, Uint32Array.from(fixture.validity));
      const outputs = {
        mask: createOutputBuffer(device, pixelCount),
        drop: createOutputBuffer(device, pixelCount),
        ids: createOutputBuffer(device, capacity),
        summitDrop: createOutputBuffer(device, capacity),
        count: createOutputBuffer(device, 1),
        overflow: createOutputBuffer(device, 1),
        requiredCount: createOutputBuffer(device, 1),
        radiusClamped: createOutputBuffer(device, 1)
      };
      const parameters = new GPUParameterBuffer(device, {
        id: 'parameters',
        format: 'float32',
        length: 8,
        values: getGPUTerrainSummitsParameterValues(
          {radius: fixture.radius, minimumDrop: fixture.minimumDrop},
          grid
        )
      });
      graph.add(
        new GPUTerrainSummits({
          grid,
          maximumRadiusPixels: fixture.maximumRadiusPixels,
          incompleteNeighborhood: fixture.policy,
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
          parameters: parameters.importToGraph(graph),
          output: {
            summitMask: importGraphBuffer(graph, 'mask', outputs.mask, 'uint32', pixelCount),
            drop: importGraphBuffer(graph, 'drop', outputs.drop, 'float32', pixelCount),
            summits: {
              ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', capacity),
              status: {
                count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
                overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
                requiredCount: importGraphBuffer(
                  graph,
                  'required-count',
                  outputs.requiredCount,
                  'uint32',
                  1
                )
              }
            },
            summitDrop: importGraphBuffer(
              graph,
              'summit-drop',
              outputs.summitDrop,
              'float32',
              capacity
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
        // Repeat execution tests count/overflow resets with the same graph.
        for (const iteration of [0, 1]) {
          submitGraph(device, compiled, undefined);
          const compare = (metric: string, actual: number[], expected: number[]) =>
            compareValues(
              'GPUTerrainSummits',
              `${fixture.name}/capacity-${capacity}/run-${iteration}`,
              SUMMITS_REFERENCE,
              metric,
              actual,
              expected
            );
          const expectedIds = fixture.expected.ids.slice(0, capacity);
          compare('mask', await readUint32(outputs.mask, pixelCount), fixture.expected.mask);
          compare('drop', await readFloat32(outputs.drop, pixelCount), fixture.expected.drop);
          const actualCount = await readUint32(outputs.count, 1);
          compare('count', actualCount, [expectedIds.length]);
          compare(
            'ids',
            (await readUint32(outputs.ids, capacity)).slice(0, actualCount[0]),
            expectedIds
          );
          compare(
            'summitDrop',
            await readFloat32(outputs.summitDrop, capacity),
            Array.from({length: capacity}, (_, index) =>
              index < expectedIds.length ? fixture.expected.drop[expectedIds[index]] : NaN
            )
          );
          compare('overflow', await readUint32(outputs.overflow, 1), [
            Number(fixture.expected.ids.length > capacity)
          ]);
          compare('requiredCount', await readUint32(outputs.requiredCount, 1), [
            fixture.expected.ids.length
          ]);
          compare('radiusClamped', await readUint32(outputs.radiusClamped, 1), [
            fixture.radiusClamped
          ]);
        }
      } finally {
        compiled.destroy();
        parameters.destroy();
        for (const buffer of [elevation, validity, ...Object.values(outputs)]) buffer.destroy();
      }
    });
  }
}
