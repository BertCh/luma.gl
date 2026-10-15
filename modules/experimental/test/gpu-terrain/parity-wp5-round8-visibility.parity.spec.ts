// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainLineOfSightParameterValues
} from '../../src/gpu-terrain/terrain-analysis';
import {
  GPUPointHorizonVisibility,
  getGPUPointHorizonVisibilityParameterValues
} from '../../src/gpu-terrain/point-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {measureComparison, assertComparisons} from './parity-wp5-round8-fixtures';
import {MATCHED_VISIBILITY_FIXTURES} from './parity-wp5-deep-visibility-fixtures';

for (const fixture of MATCHED_VISIBILITY_FIXTURES)
  for (const settings of fixture.cases)
    for (const traversal of ['march', 'pyramid'] as const) {
      for (const className of [
        'GPUTerrainViewshed',
        'GPUTerrainLineOfSight',
        'GPUTerrainCumulativeViewshed',
        'GPUPointHorizonVisibility'
      ] as const) {
        it(`${className} sampler-matched GDAL ${settings.name}/${traversal}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const {width, height, cellSize, elevation} = fixture.input;
          const pixelCount = width * height;
          const grid = makeGPURasterMetadata({width, height, cellSize: [cellSize, cellSize]});
          const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-visibility'});
          const input = createInputBuffer(device, Float32Array.from(elevation));
          const output = createOutputBuffer(device, pixelCount);
          const observers = createInputBuffer(device, Float32Array.from(settings.observers.flat()));
          const pairs = createInputBuffer(
            device,
            Float32Array.from(
              Array.from({length: pixelCount}, (_, index) => [
                ...settings.observers[0],
                index % width,
                Math.floor(index / width)
              ]).flat()
            )
          );
          const pointObservers = createInputBuffer(
            device,
            Float32Array.from(
              settings.observers.flatMap(position => [...position, settings.observerHeight, 0])
            )
          );
          const pointTargets = createInputBuffer(
            device,
            Float32Array.from(
              Array.from({length: pixelCount}, (_, index) => [
                index % width,
                Math.floor(index / width),
                settings.targetHeight,
                0
              ]).flat()
            )
          );
          const values =
            className === 'GPUTerrainViewshed'
              ? getGPUTerrainViewshedParameterValues(
                  {...settings, observer: settings.observers[0]},
                  grid
                )
              : className === 'GPUPointHorizonVisibility'
                ? getGPUPointHorizonVisibilityParameterValues(
                    {
                      ...settings,
                      toleranceDegrees: 0,
                      sigmaZ: 0,
                      targetIgnoreDistance: 0,
                      targetIgnoreFraction: 0
                    },
                    grid
                  )
                : getGPUTerrainLineOfSightParameterValues(settings, grid);
          const parameters = new GPUParameterBuffer(device, {
            id: 'settings',
            format: 'float32',
            length: values.length,
            values
          });
          const common = {
            grid,
            elevation: {
              id: 'elevation',
              format: 'float32' as const,
              storage: {
                kind: 'buffer' as const,
                values: importGraphBuffer(graph, 'elevation', input, 'float32', pixelCount)
              }
            },
            parameters: parameters.importToGraph(graph),
            traversal
          };
          const visibility = importGraphBuffer(graph, 'visibility', output, 'uint32', pixelCount);
          if (className === 'GPUTerrainViewshed')
            graph.add(new GPUTerrainViewshed({...common, output: {visibility}}));
          else if (className === 'GPUTerrainLineOfSight')
            graph.add(
              new GPUTerrainLineOfSight({
                ...common,
                pairs: importGraphBuffer(graph, 'pairs', pairs, 'float32x4', pixelCount),
                output: {visibility}
              })
            );
          else if (className === 'GPUTerrainCumulativeViewshed')
            graph.add(
              new GPUTerrainCumulativeViewshed({
                ...common,
                observers: importGraphBuffer(
                  graph,
                  'observers',
                  observers,
                  'float32x2',
                  settings.observers.length
                ),
                observersPerDispatch: 1,
                output: {visibleCount: visibility}
              })
            );
          else
            graph.add(
              new GPUPointHorizonVisibility({
                ...common,
                projection: 'planar',
                cellSize,
                minimumDistance: 1,
                maximumDistance: Math.ceil(Math.hypot(width, height) * cellSize),
                observers: importGraphBuffer(
                  graph,
                  'observers',
                  pointObservers,
                  'float32x4',
                  settings.observers.length
                ),
                targets: importGraphBuffer(graph, 'targets', pointTargets, 'float32x4', pixelCount),
                output: {visibility}
              })
            );
          const compiled = graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            const actual = await readUint32(output, pixelCount);
            const counts = className === 'GPUTerrainCumulativeViewshed';
            // Preserve native 2D comparisons as explicitly unmatched diagnostics. Neither
            // reference exposes these sample sets; their differing answers are not a tolerance.
            for (const reference of ['grass', 'gdal'])
              measureComparison(
                className,
                `${settings.name}/${traversal}`,
                reference === 'grass'
                  ? 'DIAGNOSTIC: native GRASS 8.5 r.viewshed (unmatched sampling)'
                  : 'DIAGNOSTIC: native GDAL 3.12 gdal_viewshed (unmatched sampling)',
                counts ? 'visibleCount' : 'visibility',
                actual,
                reference === 'grass'
                  ? counts
                    ? settings.grassCounts
                    : settings.grass[0]
                  : counts
                    ? settings.gdalCounts
                    : settings.gdal[0]
              );
            // z_i = i * apparentSlope(d_i) on a one-row DEM preserves every slope.
            // Wang's collinear recurrence computes their maximum exactly in real arithmetic.
            // Thus the derived semantic bound is ZERO, including cumulative counts. No
            // percentage allowance, discarded boundary cells, or relaxed integer tolerance.
            assertComparisons([
              measureComparison(
                className,
                `${settings.name}/${traversal}`,
                'SciPy bilinear samples + GDAL 3.12 radial Wang recurrence',
                counts ? 'visibleCount' : 'visibility',
                actual,
                counts
                  ? settings.coarseCounts
                  : className === 'GPUPointHorizonVisibility'
                    ? settings.point[0]
                    : settings.coarse[0]
              )
            ]);
          } finally {
            compiled.destroy();
            parameters.destroy();
            for (const buffer of [input, output, observers, pairs, pointObservers, pointTargets])
              buffer.destroy();
          }
        });
      }
    }
