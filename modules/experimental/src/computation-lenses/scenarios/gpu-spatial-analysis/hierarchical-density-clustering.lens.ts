// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.hierarchical-density-clustering',
  entry: 'gpu-spatial-analysis',
  element: 'GPUHierarchicalDensityClustering',
  title: 'HDBSCAN clusters of iNaturalist observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample}, support] = await Promise.all([
      import('./s2-lens-run'),
      import('./hierarchical-density-clustering.example'),
      import('./s1-clustering-support')
    ]);
    // Every third observation keeps the unrolled tree graph docs-sized (782 of 2,344 points).
    const points = await support.loadNaturePoints({step: 3});
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.hierarchical-density-clustering',
          element: 'GPUHierarchicalDensityClustering',
          datasets: ['s2-chicago-north-nature'],
          retain: ['labels', 'coreDistances', 'clusterCount'],
          keySpaces: [
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'},
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {minimumClusterSize: 12, minimumSamples: 5},
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const result = await runExample(graph, {
              observations,
              spatialContext: points.frame.spatialContext
            });
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                meaning: 'Every third observation (782 of 2,344), local metres',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              coreDistances: {
                view: result.coreDistances,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'Distance to the 5th nearest observation, itself counted (m)'
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
                meaning: 'Number of selected clusters'
              }
            };
          }
        })
    };
  }
});
