// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.k-means',
  entry: 'gpu-spatial-analysis',
  element: 'GPUKMeans',
  title: 'Six k-means clusters of iNaturalist observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample, K_MEANS_CLUSTER_COUNT}, support, spatial] = await Promise.all([
      import('./s2-lens-run'),
      import('./k-means.example'),
      import('./s1-clustering-support'),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    const points = await support.loadNaturePoints();
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.k-means',
          element: 'GPUKMeans',
          datasets: ['s2-chicago-north-nature'],
          retain: [
            'labels',
            'centers',
            'sizes',
            'squaredDistances',
            'inertia',
            'iterationCount',
            'converged'
          ],
          seed: 3,
          keySpaces: [
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'},
            {id: 'cluster', label: 'cluster', size: K_MEANS_CLUSTER_COUNT, indexSpace: '1d'},
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {
            k: K_MEANS_CLUSTER_COUNT,
            maximumIterations: 20,
            initialization: 'kmeans++',
            restartCount: 1,
            seed: 3
          },
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const parameters = uploads.view(
              'parameters',
              spatial.getGPUKMeansParameterValues({seed: 3}),
              'uint32'
            );
            const result = await runExample(graph, {
              observations,
              spatialContext: points.frame.spatialContext,
              parameters
            });
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              squaredDistances: {
                view: result.squaredDistances,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Squared distance to the assigned center (m2); the inertia is their sum'
              },
              labels: {
                view: result.labels,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Cluster id (nominal)'
              },
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
              centers: {
                view: result.centers,
                role: 'intermediate',
                keySpace: 'cluster',
                meaning: 'Final cluster centers [x, y] (m)'
              },
              sizes: {view: result.sizes, role: 'intermediate', keySpace: 'cluster'},
              inertia: {
                view: result.inertia,
                role: 'intermediate',
                keySpace: 'status',
                meaning: 'Total within-cluster sum of squares (m2)'
              },
              iterationCount: {
                view: result.iterationCount,
                role: 'intermediate',
                keySpace: 'status',
                meaning: 'Lloyd iterations that moved centers'
              },
              converged: {
                view: result.converged,
                role: 'intermediate',
                keySpace: 'status',
                meaning: '1 when the last iteration moved no center'
              }
            };
          }
        })
    };
  }
});
