// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.region-mask',
  entry: 'gpu-spatial-analysis',
  element: 'GPURegionMask',
  title: 'Region mask: tracts inside a rectangle',
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
    const [{runExample}, {loadChicagoTracts}, utilities, support, {STATISTICS_OPERATORS}] =
      await Promise.all([
        import('./region-mask.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('./s1-summaries-support'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    const centroids = await support.projectTractCentroids(tracts);
    const [minX, minY, maxX, maxY] = support.getPositionBounds(centroids);
    const bounds = Float32Array.of(
      minX + 0.35 * (maxX - minX),
      minY + 0.25 * (maxY - minY),
      minX + 0.7 * (maxX - minX),
      minY + 0.6 * (maxY - minY)
    );
    return {
      run: async device => {
        const extra: {destroy(): void}[] = [];
        try {
          return await utilities.runTractStatisticsScenario({
            device,
            id: 'gpu-spatial-analysis.region-mask',
            tracts,
            entry: STATISTICS_OPERATORS['GPURegionMask'],
            columns: {},
            choices: {regionKind: 'rectangle'},
            metadata: {
              // Centroids and bounds are metres in a local azimuthal-equidistant frame.
              coordinateModel: 'projected'
            },
            async execute({graph}) {
              const positions = utilities.uploadSpatialView(
                graph,
                'positions',
                centroids,
                'float32x2'
              );
              const rectangle = utilities.uploadSpatialView(graph, 'bounds', bounds, 'float32');
              extra.push(positions, rectangle);
              const execution = await runExample(device, {
                graph,
                positions: positions.view,
                bounds: rectangle.view
              });
              return {
                ...execution,
                keySpaces: [
                  {
                    id: 'bounds',
                    label: 'minX, minY, maxX, maxY',
                    size: 4,
                    indexSpace: '1d' as const
                  },
                  {id: 'status', label: 'Truncation flag', size: 1, indexSpace: '1d' as const}
                ],
                tags: {
                  positions: {
                    role: 'input',
                    meaning: 'Tract area centroids in local metres (azimuthal equidistant frame)'
                  },
                  bounds: {
                    role: 'input',
                    keySpace: 'bounds',
                    meaning: 'Inclusive rectangle in local metres'
                  },
                  mask: {
                    role: 'output',
                    polygons: true,
                    meaning: 'Tracts whose centroid is inside the rectangle (1)'
                  },
                  overflow: {role: 'intermediate', keySpace: 'status'}
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
