// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.group-convex-hull',
  entry: 'gpu-spatial-analysis',
  element: 'GPUGroupConvexHull',
  title: 'Convex outlines of DBSCAN clusters of iNaturalist observations',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [
      {runS2Lens},
      {runExample, GROUP_CONVEX_HULL_GROUP_COUNT, GROUP_CONVEX_HULL_VERTEX_CAPACITY},
      support
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./group-convex-hull.example'),
      import('./s1-clustering-support')
    ]);
    const points = await support.loadNaturePoints();
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.group-convex-hull',
          element: 'GPUGroupConvexHull',
          datasets: ['s2-chicago-north-nature'],
          retain: ['labels', 'vertexPositions', 'offsets', 'counts', 'sizes', 'overflow'],
          keySpaces: [
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'},
            {
              id: 'group',
              label: 'cluster slot',
              size: GROUP_CONVEX_HULL_GROUP_COUNT,
              indexSpace: '1d'
            },
            {
              id: 'hull-offset',
              label: 'hull offset',
              size: GROUP_CONVEX_HULL_GROUP_COUNT + 1,
              indexSpace: '1d'
            },
            {
              id: 'hull-vertex',
              label: 'hull vertex slot',
              size: GROUP_CONVEX_HULL_VERTEX_CAPACITY,
              indexSpace: '1d'
            },
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {
            groupCount: GROUP_CONVEX_HULL_GROUP_COUNT,
            maximumVerticesPerGroup: 64,
            extremePointFilter: true
          },
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            // The cluster slot is its own nominal id: it colours each outline, not an analysis result.
            const groupIds = uploads.view(
              'groupIds',
              Uint32Array.from({length: GROUP_CONVEX_HULL_GROUP_COUNT}, (_, group) => group),
              'uint32'
            );
            const result = await runExample(graph, {
              observations,
              spatialContext: points.frame.spatialContext,
              bounds: points.extent
            });
            uploads.own(result.parameters);
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
              groupIds: {
                view: groupIds,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'Cluster slot index (nominal), the colour of each outline'
              },
              sizes: {
                view: result.sizes,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'True hull vertex count per cluster'
              },
              counts: {
                view: result.counts,
                role: 'intermediate',
                keySpace: 'group',
                meaning: 'Emitted hull vertex count per cluster'
              },
              overflow: {
                view: result.overflow,
                role: 'intermediate',
                keySpace: 'status',
                meaning:
                  'Overflow bits: 1 when a hull was too large, 2 when the vertex list was full'
              },
              offsets: {
                view: result.offsets,
                role: 'intermediate',
                keySpace: 'hull-offset',
                meaning: 'First vertex slot of each cluster hull'
              },
              vertexPositions: {
                view: result.vertexPositions,
                role: 'output',
                keySpace: 'hull-vertex',
                meaning: 'Convex hull vertices of each cluster, counter-clockwise (m)',
                geometry: {
                  kind: 'polygons',
                  offsetsCapture: 'offsets',
                  valuesCapture: 'groupIds',
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
