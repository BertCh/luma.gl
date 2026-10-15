// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.local-outlier-factor',
  entry: 'gpu-spatial-analysis',
  element: 'GPULocalOutlierFactor',
  title: 'Locally sparse iNaturalist observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runExample}, {runS2Lens}, support, spatial] = await Promise.all([
      import('./local-outlier-factor.example'),
      import('./s2-lens-run'),
      import('./s1-clustering-support'),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    const points = await support.loadNaturePoints();
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.local-outlier-factor',
          element: 'GPULocalOutlierFactor',
          datasets: ['s2-chicago-north-nature'],
          retain: ['kDistance', 'density', 'lof', 'outlier', 'outlierCount', 'neighborOverflow'],
          keySpaces: [
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'},
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {k: 10, threshold: 3, densityFloor: 10},
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const searchParameters = uploads.view(
              'searchParameters',
              spatial.getGPUNeighborSearchParameterValues({bounds: points.extent}),
              'float32'
            );
            // Observations repeat the same location (rounded to 1 m), so the floor is 10 m.
            const parameters = uploads.view(
              'parameters',
              spatial.getGPULocalOutlierFactorParameterValues({threshold: 3, densityFloor: 10}),
              'float32'
            );
            const result = await runExample(graph, {
              observations,
              spatialContext: points.frame.spatialContext,
              searchParameters,
              parameters
            });
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              kDistance: {
                view: result.kDistance,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Distance to the 10th nearest observation (m)'
              },
              localReachabilityDensity: {
                view: result.localReachabilityDensity,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Local reachability density (1/m): inverse mean reach distance plus 10 m'
              },
              outlier: {
                view: result.outlier,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: '1 where lof is above the chosen cut of 3, else 0'
              },
              outlierCount: {
                view: result.outlierCount,
                role: 'intermediate',
                keySpace: 'status',
                meaning: 'Observations flagged above the cut'
              },
              neighborOverflow: {
                view: result.neighborOverflow,
                role: 'intermediate',
                keySpace: 'status',
                meaning: 'Neighbour table capacity exceeded (1) or not (0)'
              },
              lof: {
                view: result.lof,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Local outlier factor; 1 is as dense as the neighbours'
              },
              resultObservations: {
                view: observations,
                role: 'output',
                keySpace: 'observation',
                geometry: {
                  kind: 'points',
                  valuesCapture: 'lof',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              }
            };
          }
        })
    };
  }
});
