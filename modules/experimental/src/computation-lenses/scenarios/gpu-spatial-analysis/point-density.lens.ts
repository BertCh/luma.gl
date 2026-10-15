// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.point-density',
  entry: 'gpu-spatial-analysis',
  element: 'GPUPointDensity',
  title: 'Observation counts on a 125 m Chicago grid',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, snapS2Extent, closeS2Rings},
      {runExample},
      data,
      areas
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./point-density.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default),
      import('../../data/s2-chicago-north-areas.json').then(module => module.default)
    ]);
    return {
      run: device => {
        // GPUPointDensity declares measure: 'area' and rejects AEQD contexts, including count mode.
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const projected = projectS2Positions(frame, data.positions);
        const extent = snapS2Extent(projected.extent, 125);
        const context = closeS2Rings(
          frame.projectArray(areas.polygons.vertices),
          areas.polygons.ringOffsets
        );
        const width = Math.round((extent[2] - extent[0]) / 125);
        const height = Math.round((extent[3] - extent[1]) / 125);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.point-density',
          element: 'GPUPointDensity',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-areas'],
          retain: ['counts'],
          keySpaces: [
            {
              id: 'cell',
              label: '125 m cell, rows from south',
              size: width * height,
              indexSpace: '2d',
              shape: [width, height]
            },
            {
              id: 'observation',
              label: 'observation',
              size: data.positions.length / 2,
              indexSpace: '1d'
            },
            {
              id: 'contextVertex',
              label: 'community-area vertex',
              size: context.positions.length / 2,
              indexSpace: '1d'
            },
            {
              id: 'boundary',
              label: 'community-area boundary',
              size: context.ringOffsets.length,
              indexSpace: '1d'
            }
          ],
          choices: {gridSize: `${width} × ${height}`, binning: 'grid', statistic: 'count'},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected.positions, 'float32x2');
            const communityAreas = uploads.view('community-areas', context.positions, 'float32x2');
            const ringOffsets = uploads.view('ring-offsets', context.ringOffsets, 'uint32');
            const {counts} = await runExample(graph, {
              observations,
              spatialContext: frame.spatialContext,
              extent
            });
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              communityAreas: {
                view: communityAreas,
                role: 'context',
                keySpace: 'contextVertex',
                geometry: {
                  kind: 'polygons',
                  coordinateModel: 'projected',
                  unit: 'm',
                  offsetsCapture: 'ringOffsets'
                }
              },
              ringOffsets: {view: ringOffsets, role: 'input', keySpace: 'boundary'},
              counts: {
                view: counts,
                role: 'output',
                keySpace: 'cell',
                geometry: {
                  kind: 'raster',
                  shape: [width, height],
                  coordinateModel: 'projected',
                  cellSize: 125,
                  unit: 'm',
                  origin: [extent[0], extent[1]],
                  rowDirection: 'north'
                }
              }
            };
          }
        });
      }
    };
  }
});
