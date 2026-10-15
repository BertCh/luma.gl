// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-clustering',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialClustering',
  title: 'DBSCAN clusters of iNaturalist observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample}, support] = await Promise.all([
      import('./s2-lens-run'),
      import('./spatial-clustering.example'),
      import('./s1-clustering-support')
    ]);
    const points = await support.loadNaturePoints();
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.spatial-clustering',
          element: 'GPUSpatialClustering',
          datasets: ['s2-chicago-north-nature'],
          retain: ['labels', 'coreFlags', 'clusterCount'],
          keySpaces: [
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'},
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {epsilon: 250, minimumPoints: 6},
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const result = await runExample(graph, {
              observations,
              spatialContext: points.frame.spatialContext,
              bounds: points.extent
            });
            uploads.own(result.parameters);
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              coreFlags: {
                view: result.coreFlags,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Core point: at least 6 observations within 250 m, itself included'
              },
              labels: {
                view: result.labels,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Cluster id (nominal); 0xffffffff is noise'
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
              clusterCount: {
                view: result.clusterCount,
                role: 'intermediate',
                keySpace: 'status',
                meaning: 'Number of clusters found'
              }
            };
          }
        })
    };
  }
});
