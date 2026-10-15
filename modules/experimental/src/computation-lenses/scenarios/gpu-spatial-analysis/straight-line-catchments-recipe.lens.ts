// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

/** Catchment raster cell size in metres. */
const CELL_SIZE = 250;

registerLensScenario({
  id: 'gpu-spatial-analysis.straight-line-catchments-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addStraightLineCatchmentsRecipe',
  title: 'Straight-line catchments of Chicago North Side facilities',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.facilities,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, snapS2Extent},
      {runExample},
      facilityData,
      nature
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./straight-line-catchments-recipe.example'),
      import('../../data/s2-chicago-north-facilities.json').then(module => module.default),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const facilities = projectS2Positions(frame, facilityData.positions);
        const observations = projectS2Positions(frame, nature.positions);
        const extent = snapS2Extent(
          [
            Math.min(facilities.extent[0], observations.extent[0]) - CELL_SIZE,
            Math.min(facilities.extent[1], observations.extent[1]) - CELL_SIZE,
            Math.max(facilities.extent[2], observations.extent[2]) + CELL_SIZE,
            Math.max(facilities.extent[3], observations.extent[3]) + CELL_SIZE
          ],
          CELL_SIZE
        );
        const width = Math.round((extent[2] - extent[0]) / CELL_SIZE);
        const height = Math.round((extent[3] - extent[1]) / CELL_SIZE);
        // Observations per cell, row-major from the top row: the value raster the catchments sum.
        const counts = new Float32Array(width * height);
        for (let index = 0; index < observations.positions.length; index += 2) {
          const column = Math.floor((observations.positions[index] - extent[0]) / CELL_SIZE);
          const row = Math.floor((extent[3] - observations.positions[index + 1]) / CELL_SIZE);
          counts[row * width + column]++;
        }
        const facilityCount = facilities.positions.length / 2;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.straight-line-catchments-recipe',
          element: 'addStraightLineCatchmentsRecipe',
          datasets: ['s2-chicago-north-facilities', 's2-chicago-north-nature'],
          retain: ['allocation', 'distances', 'cellCounts', 'valueCounts', 'sums', 'means'],
          keySpaces: [
            {
              id: 'cell',
              label: 'grid cell',
              size: width * height,
              indexSpace: '2d',
              shape: [width, height]
            },
            {id: 'facility', label: 'facility catchment', size: facilityCount, indexSpace: '1d'}
          ],
          choices: {cellSize: CELL_SIZE, mode: 'exact'},
          async build(graph, uploads) {
            const facilityView = uploads.view('facilities', facilities.positions, 'float32x2');
            const observationCounts = uploads.view('observationCounts', counts, 'float32');
            const result = await runExample(graph, {
              facilities: facilityView,
              observationCounts,
              spatialContext: frame.spatialContext,
              extent,
              cellSize: CELL_SIZE
            });
            uploads.own(result.parameters);
            const raster = {
              kind: 'raster' as const,
              shape: [width, height] as const,
              coordinateModel: 'projected' as const,
              cellSize: CELL_SIZE,
              unit: 'm',
              // Row 0 is the northern row (row = (maxY - y) / cell size).
              origin: [extent[0], extent[3]] as const,
              rowDirection: 'south' as const
            };
            const captures: Record<string, S2CaptureSpec> = {
              facilities: {
                view: facilityView,
                role: 'input',
                keySpace: 'facility',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              observationCounts: {
                view: observationCounts,
                role: 'input',
                keySpace: 'cell',
                geometry: raster
              },
              allocation: {
                view: result.allocation,
                role: 'output',
                keySpace: 'cell',
                geometry: raster
              },
              distances: {
                view: result.distances,
                role: 'output',
                keySpace: 'cell',
                geometry: raster
              },
              resultFacilities: {
                view: facilityView,
                role: 'output',
                keySpace: 'facility',
                geometry: {
                  kind: 'points',
                  valuesCapture: 'sums',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              }
            };
            for (const name of ['cellCounts', 'valueCounts', 'sums', 'means'] as const) {
              captures[name] = {
                view: result.statistics[name],
                role: 'output',
                keySpace: 'facility'
              };
            }
            return captures;
          }
        });
      }
    };
  }
});
