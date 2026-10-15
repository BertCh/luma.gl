// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.group-geometry',
  entry: 'gpu-spatial-analysis',
  element: 'GPUGroupGeometry',
  title: 'Centres and spread of DBSCAN clusters of iNaturalist observations',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample, GROUP_GEOMETRY_GROUP_COUNT}, support] = await Promise.all([
      import('./s2-lens-run'),
      import('./group-geometry.example'),
      import('./s1-clustering-support')
    ]);
    const points = await support.loadNaturePoints();
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.group-geometry',
          element: 'GPUGroupGeometry',
          datasets: ['s2-chicago-north-nature'],
          retain: [
            'labels',
            'counts',
            'groupBounds',
            'meanCenters',
            'medoidIndices',
            'standardDistances'
          ],
          keySpaces: [
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'},
            {id: 'group', label: 'cluster slot', size: GROUP_GEOMETRY_GROUP_COUNT, indexSpace: '1d'}
          ],
          choices: {groupCount: GROUP_GEOMETRY_GROUP_COUNT, medoidMaximumGroupSize: 4096},
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const result = await runExample(graph, {
              observations,
              spatialContext: points.frame.spatialContext,
              bounds: points.extent
            });
            result.parameters.forEach(parameters => uploads.own(parameters));
            return {
              observations: {
                view: observations,
                role: 'context',
                keySpace: 'observation',
                meaning: 'Observations, local metres',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              labels: {
                view: result.labels,
                role: 'intermediate',
                keySpace: 'observation',
                meaning: 'DBSCAN cluster id (nominal); 0xffffffff is noise'
              },
              groupBounds: {
                view: result.bounds,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'Bounding box [minX, minY, maxX, maxY] per cluster (m)'
              },
              medoidIndices: {
                view: result.medoidIndices,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'Observation row of the most central point of each cluster'
              },
              standardDistances: {
                view: result.standardDistances,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'One-sigma spread radius of each cluster (m)'
              },
              counts: {
                view: result.counts,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'Observations per cluster slot'
              },
              meanCenters: {
                view: result.meanCenters,
                role: 'output',
                keySpace: 'group',
                meaning: 'Mean centre of each cluster, sized by its count (m)',
                geometry: {
                  kind: 'points',
                  valuesCapture: 'counts',
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
