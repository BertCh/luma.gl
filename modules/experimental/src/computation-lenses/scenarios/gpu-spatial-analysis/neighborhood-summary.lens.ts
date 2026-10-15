// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.neighborhood-summary',
  entry: 'gpu-spatial-analysis',
  element: 'GPUNeighborhoodSummary',
  title: 'Neighbourhood summary: Chicago poverty around each tract',
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
    const [{runExample}, {loadChicagoTracts}, utilities, {STATISTICS_OPERATORS}] =
      await Promise.all([
        import('./neighborhood-summary.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.neighborhood-summary',
          tracts,
          entry: STATISTICS_OPERATORS['GPUNeighborhoodSummary'],
          columns: {values: 'poverty150Pct'},
          choices: {transform: 'row', includeFocal: false, statistics: 'mean'},
          metadata: {
            weightsStandardisation: 'row',
            valueColumn: 'poverty150Pct',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, polygons, columns, upload}) {
            const categories = upload(
              'categories',
              Uint32Array.from(tracts.columns['communityArea']),
              'uint32'
            );
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values'],
              categories
            });
            return {
              ...execution,
              tags: {
                values: {role: 'input', meaning: 'Population below 150% poverty line (%)'},
                categories: {
                  role: 'input',
                  meaning: 'Community area label of each tract (nominal)'
                },
                table: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Row-standardised mean poverty of the queen neighbours (%)'
                },
                entropy: {
                  role: 'intermediate',
                  polygons: true,
                  meaning: 'Entropy of the neighbours’ community-area labels (nats)'
                }
              }
            };
          }
        })
    };
  }
});
