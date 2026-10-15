// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

registerLensScenario({
  id: 'gpu-spatial-analysis.points-in-polygons-choropleth-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addPointsInPolygonsChoroplethRecipe',
  title: 'Observations per tract, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, uploadS2JoinPolygons},
      {runExample},
      nature,
      tracts
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./points-in-polygons-choropleth-recipe.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, nature.positions).positions;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'observation', label: 'observation', size: projected.length / 2, indexSpace: '1d'},
          {id: 'class-break', label: 'class break', size: 6, indexSpace: '1d'},
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.points-in-polygons-choropleth-recipe',
          element: 'addPointsInPolygonsChoroplethRecipe',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-tracts'],
          retain: ['counts', 'classIndices', 'colors', 'breaks', 'overflow', 'uncertainCount'],
          keySpaces,
          choices: {backend: 'zonal', statistic: 'count', classMethod: 'quantile', classCount: 5},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected, 'float32x2');
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(...tractGeometry.keySpaces);
            const captures: Record<string, S2CaptureSpec> = {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              ...tractGeometry.captures
            };
            captures['tracts'].meaning = 'context';
            const result = await runExample(graph, {
              observations,
              tracts: tractGeometry.geometry,
              spatialContext: frame.spatialContext
            });
            result.parameters.forEach(parameters => uploads.own(parameters));
            captures['resultTracts'] = {
              view: tractGeometry.display,
              role: 'output',
              keySpace: 'tracts',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'tractsDisplayOffsets',
                valuesCapture: 'counts',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['counts'] = {view: result.counts, role: 'output', keySpace: 'tracts'};
            captures['classIndices'] = {
              view: result.classIndices,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['breaks'] = {view: result.breaks, role: 'output', keySpace: 'class-break'};
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            captures['uncertainCount'] = {
              view: result.uncertainCount,
              role: 'output',
              keySpace: 'status'
            };
            return captures;
          }
        });
      }
    };
  }
});
