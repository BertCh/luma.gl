// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.cluster-and-outline-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addClusterAndOutlineRecipe',
  title: 'Clusters and outlines of iNaturalist observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, projectS2Positions}, {runExample}, nature] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./cluster-and-outline-recipe.example'),
        import('../../data/s2-chicago-north-nature.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, nature.positions, 250);
        const count = projected.positions.length / 2;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.cluster-and-outline-recipe',
          element: 'addClusterAndOutlineRecipe',
          datasets: ['s2-chicago-north-nature'],
          retain: ['labels', 'clusterCount', 'hullPositions', 'hullOffsets', 'areas', 'perimeters'],
          keySpaces: [
            {id: 'observation', label: 'observation', size: count, indexSpace: '1d'},
            {id: 'cluster', label: 'cluster slot', size: 128, indexSpace: '1d'},
            {id: 'hull-offset', label: 'hull offset', size: 129, indexSpace: '1d'},
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {epsilon: 250, minimumPoints: 6},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected.positions, 'float32x2');
            const result = await runExample(graph, {
              observations,
              spatialContext: frame.spatialContext,
              bounds: projected.extent
            });
            result.parameters.forEach(parameters => uploads.own(parameters));
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              labels: {view: result.labels, role: 'output', keySpace: 'observation'},
              resultObservations: {
                view: observations,
                role: 'output',
                keySpace: 'observation',
                geometry: {
                  kind: 'points',
                  valuesCapture: 'labels',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              },
              hullOffsets: {view: result.hullOffsets, role: 'output', keySpace: 'hull-offset'},
              areas: {view: result.areas, role: 'output', keySpace: 'cluster'},
              perimeters: {view: result.perimeters, role: 'output', keySpace: 'cluster'},
              hullPositions: {
                view: result.hullPositions,
                role: 'output',
                keySpace: 'cluster',
                geometry: {
                  kind: 'polygons',
                  offsetsCapture: 'hullOffsets',
                  valuesCapture: 'areas',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              },
              clusterCount: {view: result.clusterCount, role: 'output', keySpace: 'status'}
            };
          }
        });
      }
    };
  }
});
