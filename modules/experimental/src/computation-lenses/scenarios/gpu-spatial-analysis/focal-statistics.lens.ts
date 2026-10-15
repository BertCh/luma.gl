// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.focal-statistics';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUFocalStatistics',
  title: 'Focal statistics: roughness of a 32 by 32 terrain lattice',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's1-dixie-lattice/1',
    source:
      'AWS Terrain Tiles (Terrarium, USGS 3DEP ~10 m), central block of the Dixie DEM at 120 m',
    licence: 'Public domain (USGS 3DEP); terrain tiles by Mapzen/Tilezen on AWS Open Data'
  },
  async load() {
    const [{runExample}, utilities, spatial, {STATISTICS_OPERATORS}, data] = await Promise.all([
      import('./focal-statistics.example'),
      import('./spatial-scenario-utils'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics'),
      import('../../data/s1-dixie-lattice.json').then(module => module.default)
    ]);
    const {width, height} = data;
    const settings = {radius: 3, shape: 'circle', minimumCount: 1} as const;
    return {
      async run(device) {
        const retained: string[] = [];
        const graph = utilities.createSpatialRecordingGraph(device, SCENARIO_ID, retained);
        const elevation = utilities.uploadSpatialView(
          graph,
          'elevation',
          Float32Array.from(data.values),
          'float32'
        );
        const parameters = utilities.uploadSpatialView(
          graph,
          'parameters',
          spatial.getGPUFocalStatisticsParameterValues(settings),
          'float32'
        );
        try {
          const execution = await runExample(device, {
            graph,
            elevation: elevation.view,
            parameters: parameters.view,
            width,
            height,
            cellSize: data.cellSize,
            origin: [data.origin[0], data.origin[1]]
          });
          // Retain the example's transients by buffer id (capture names are not resource ids).
          retained.push(...utilities.getRetainedViewIds(execution.views));
          const record = await utilities.runSpatialScenario({
            id: SCENARIO_ID,
            graph,
            encodeOptions: {parameters: undefined},
            views: execution.views,
            entry: STATISTICS_OPERATORS['GPUFocalStatistics'],
            choices: settings,
            metadata: {coordinateModel: 'projected', valueColumn: 'elevation'},
            datasetVersion: data.id,
            seed: 1,
            axes: {
              axes: [
                {
                  id: 'entity',
                  label: 'Lattice cell',
                  kind: 'data',
                  keySpace: 'cell',
                  size: width * height
                }
              ],
              keySpaces: [
                {
                  id: 'cell',
                  label: '120 m cell, rows from north',
                  size: width * height,
                  indexSpace: '2d',
                  shape: [width, height]
                }
              ]
            },
            tags: {
              elevation: {
                role: 'input',
                keySpace: 'cell',
                meaning: 'Elevation of the 120 m cell (m)'
              },
              mean: {
                role: 'output',
                keySpace: 'cell',
                meaning: 'Mean elevation of the circular window, radius 3 cells (m)'
              },
              standardDeviation: {
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'Population deviation of the window (m)'
              },
              windowCounts: {
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'Valid cells in the window; fewer at the raster edge'
              }
            }
          });
          const raster = {
            kind: 'raster' as const,
            shape: [width, height] as const,
            coordinateModel: 'projected' as const,
            cellSize: data.cellSize,
            unit: 'm'
          };
          return utilities.finishSpatialWeightsRecord(record, {
            elevation: {role: 'input', geometry: raster},
            mean: {role: 'output', geometry: raster}
          });
        } finally {
          elevation.destroy();
          parameters.destroy();
        }
      }
    };
  }
});
