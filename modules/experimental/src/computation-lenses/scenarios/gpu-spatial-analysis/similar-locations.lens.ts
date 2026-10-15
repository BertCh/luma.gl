// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

/** Study tract whose neighbourhood profile the other tracts are compared to. */
const REFERENCE_TRACT = 40;

registerLensScenario({
  id: 'gpu-spatial-analysis.similar-locations',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSimilarLocations',
  title: 'Tracts most like one reference tract in five social indicators',
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
      {runExample, SIMILAR_LOCATIONS_COLUMNS},
      support,
      utilities,
      spatial,
      {STATISTICS_OPERATORS}
    ] = await Promise.all([
      import('./similar-locations.example'),
      import('./s1-regions-support'),
      import('./spatial-scenario-utils'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await support.loadRegionTracts();
    const matrix = new Float32Array(tracts.count * SIMILAR_LOCATIONS_COLUMNS.length);
    for (const [column, name] of SIMILAR_LOCATIONS_COLUMNS.entries()) {
      for (let tract = 0; tract < tracts.count; tract++) {
        matrix[tract * SIMILAR_LOCATIONS_COLUMNS.length + column] = tracts.columns[name][tract];
      }
    }
    const reference = new Uint32Array(tracts.count);
    reference[REFERENCE_TRACT] = 1;
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.similar-locations',
          tracts,
          entry: STATISTICS_OPERATORS['GPUSimilarLocations'],
          columns: {},
          choices: {
            standardization: 'zscore',
            resultCount: 10,
            direction: 'most',
            excludeReference: true
          },
          async execute({graph, upload}) {
            const execution = await runExample(device, {
              graph,
              values: upload('values', matrix, 'float32'),
              referenceMask: upload('referenceMask', reference, 'uint32'),
              parameters: upload(
                'parameters',
                spatial.getGPUSimilarLocationsParameterValues(
                  {resultCount: 10},
                  SIMILAR_LOCATIONS_COLUMNS.length
                ),
                'float32'
              )
            });
            return {
              ...execution,
              keySpaces: [
                {
                  id: 'attribute',
                  label: 'Tract attribute value',
                  size: matrix.length,
                  indexSpace: '1d' as const
                },
                {id: 'top', label: 'Most similar rank', size: 32, indexSpace: '1d' as const},
                {id: 'status', label: 'Result count', size: 1, indexSpace: '1d' as const}
              ],
              tags: {
                values: {
                  role: 'intermediate',
                  keySpace: 'attribute',
                  meaning:
                    'Poverty %, median income, uninsured %, no-vehicle %, unemployed %, row-major per tract'
                },
                referenceMask: {
                  role: 'intermediate',
                  meaning: 'The one reference tract (1); every other tract is compared to it'
                },
                ranks: {
                  role: 'intermediate',
                  meaning: 'Similarity rank, 0 is the most similar; the reference has none'
                },
                distances: {
                  role: 'output',
                  polygons: true,
                  meaning:
                    'Distance to the reference in standardized units; small is similar, the reference is blank'
                },
                topIds: {
                  role: 'intermediate',
                  keySpace: 'top',
                  meaning: 'Tract rows of the 10 most similar tracts in rank order'
                },
                topCount: {
                  role: 'intermediate',
                  keySpace: 'status',
                  meaning: 'Valid entries of topIds'
                }
              }
            };
          }
        })
    };
  }
});
