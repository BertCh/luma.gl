// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.region-statistics',
  entry: 'gpu-spatial-analysis',
  element: 'GPURegionStatistics',
  title: 'Region statistics: poverty counts inside a lasso',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's1-chicago-tracts/1',
    source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022)',
    licence:
      'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain; CDC/ATSDR SVI 2022, public domain'
  },
  async load() {
    const [
      {runExample, REGION_STATISTICS_BIN_COUNT},
      {loadChicagoTracts},
      utilities,
      support,
      {STATISTICS_OPERATORS}
    ] = await Promise.all([
      import('./region-statistics.example'),
      import('./spatial-lens-data'),
      import('./spatial-scenario-utils'),
      import('./s1-summaries-support'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await loadChicagoTracts();
    const centroids = await support.projectTractCentroids(tracts);
    // A five-vertex lasso over the middle of the city, in local metres.
    const [minX, minY, maxX, maxY] = support.getPositionBounds(centroids);
    const x0 = minX + 0.3 * (maxX - minX);
    const x1 = minX + 0.65 * (maxX - minX);
    const y0 = minY + 0.3 * (maxY - minY);
    const y1 = minY + 0.75 * (maxY - minY);
    const lasso = Float32Array.from([
      x0,
      y0,
      x1,
      y0 + 0.15 * (y1 - y0),
      x1,
      y1,
      x0 + 0.2 * (x1 - x0),
      y1,
      x0,
      y0 + 0.5 * (y1 - y0)
    ]);
    return {
      run: async device => {
        const extra: {destroy(): void}[] = [];
        try {
          return await utilities.runTractStatisticsScenario({
            device,
            id: 'gpu-spatial-analysis.region-statistics',
            tracts,
            entry: STATISTICS_OPERATORS['GPURegionStatistics'],
            columns: {values: 'poverty150'},
            choices: {regionKind: 'polygon', binCount: REGION_STATISTICS_BIN_COUNT},
            metadata: {
              // Centroids and the lasso are metres in a local azimuthal-equidistant frame.
              coordinateModel: 'projected',
              valueColumn: 'poverty150'
            },
            async execute({graph, columns}) {
              const positions = utilities.uploadSpatialView(
                graph,
                'positions',
                centroids,
                'float32x2'
              );
              const lassoVertices = utilities.uploadSpatialView(
                graph,
                'lassoVertices',
                lasso,
                'float32x2'
              );
              const lassoVertexCount = utilities.uploadSpatialView(
                graph,
                'lassoVertexCount',
                Uint32Array.of(lasso.length / 2),
                'uint32'
              );
              extra.push(positions, lassoVertices, lassoVertexCount);
              const execution = await runExample(device, {
                graph,
                positions: positions.view,
                values: columns['values'],
                lassoVertices: lassoVertices.view,
                lassoVertexCount: lassoVertexCount.view
              });
              return {
                ...execution,
                keySpaces: [
                  {
                    id: 'lasso',
                    label: 'Lasso vertex',
                    size: lasso.length / 2,
                    indexSpace: '1d' as const
                  },
                  {
                    id: 'lassoCount',
                    label: 'Active lasso vertices',
                    size: 1,
                    indexSpace: '1d' as const
                  },
                  {
                    id: 'summary',
                    label: 'Selected, valued, sum, mean, minimum, maximum, outside, flags, bins',
                    size: 8 + REGION_STATISTICS_BIN_COUNT,
                    indexSpace: '1d' as const
                  }
                ],
                tags: {
                  positions: {
                    role: 'input',
                    meaning: 'Tract area centroids in local metres (azimuthal equidistant frame)'
                  },
                  values: {role: 'input', meaning: 'People below 150% of the poverty line (count)'},
                  lassoVertices: {
                    role: 'input',
                    keySpace: 'lasso',
                    meaning: 'Lasso polygon in local metres, even-odd fill'
                  },
                  lassoVertexCount: {role: 'input', keySpace: 'lassoCount'},
                  mask: {
                    role: 'output',
                    polygons: true,
                    meaning: 'Tracts whose centroid is inside the lasso (1)'
                  },
                  summary: {
                    role: 'intermediate',
                    keySpace: 'summary',
                    meaning:
                      'u32 words; sum, mean, minimum and maximum are f32 bit patterns; then the histogram bins'
                  }
                }
              };
            }
          });
        } finally {
          extra.forEach(resource => resource.destroy());
        }
      }
    };
  }
});
