// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUTerrainWatersheds} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WATERSHED_CASES, EXACT_TOLERANCE, compareRound4Values} from '../parity-wp5-round4-fixtures';

for (const fixture of WATERSHED_CASES) {
  for (const mode of ['basins', 'nested-pour-points', 'no-valid-pour-points'] as const) {
    it(`GPUTerrainWatersheds pysheds parity: ${fixture.name}/${mode}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const count = fixture.width * fixture.height;
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-watersheds'});
      const directions = createInputBuffer(device, Uint32Array.from(fixture.directions));
      const points =
        mode === 'basins'
          ? undefined
          : createInputBuffer(
              device,
              Uint32Array.from(
                mode === 'nested-pour-points' ? fixture.pourPoints : [0, count + 7, 0xffffffff]
              )
            );
      const labels = createOutputBuffer(device, count);
      const converged = createOutputBuffer(device, 1);
      try {
        graph.add(
          new GPUTerrainWatersheds({
            grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
            flowDirections: importGraphBuffer(graph, 'directions', directions, 'uint32', count),
            pourPoints: points
              ? importGraphBuffer(
                  graph,
                  'points',
                  points,
                  'uint32',
                  mode === 'nested-pour-points' ? fixture.pourPoints.length : 3
                )
              : undefined,
            output: {
              labels: importGraphBuffer(graph, 'labels', labels, 'uint32', count),
              status: {converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1)}
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          compareRound4Values(
            'GPUTerrainWatersheds',
            `${fixture.name}/${mode}`,
            'pysheds.Grid.catchment; smallest containing nested catchment, duplicate index precedence',
            'labels',
            await readUint32(labels, count),
            mode === 'basins'
              ? fixture.basinLabels
              : mode === 'nested-pour-points'
                ? fixture.pourPointLabels
                : new Array<number>(count).fill(0xffffffff),
            EXACT_TOLERANCE
          );
          compareRound4Values(
            'GPUTerrainWatersheds',
            `${fixture.name}/${mode}`,
            'Complete finite acyclic D8 graph',
            'converged',
            await readUint32(converged, 1),
            [1],
            EXACT_TOLERANCE
          );
        } finally {
          compiled.destroy();
        }
      } finally {
        directions.destroy();
        points?.destroy();
        labels.destroy();
        converged.destroy();
      }
    });
  }
}
