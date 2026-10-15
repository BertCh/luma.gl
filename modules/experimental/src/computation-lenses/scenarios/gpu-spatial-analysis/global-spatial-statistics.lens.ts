// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.global-spatial-statistics',
  entry: 'gpu-spatial-analysis',
  element: 'GPUGlobalSpatialStatistics',
  title: 'Global Moran, Geary and G: Chicago poverty percentages',
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
        import('./global-spatial-statistics.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.global-spatial-statistics',
          tracts,
          entry: STATISTICS_OPERATORS['GPUGlobalSpatialStatistics'],
          columns: {values: 'poverty150Pct'},
          choices: {transform: 'row', islands: 'exclude', alternative: 'two-sided'},
          metadata: {
            // Analytic normal p-values of one test each; no multiple-testing family exists.
            pCorrection: 'none',
            significanceAvailable: true,
            weightsStandardisation: 'row',
            valueColumn: 'poverty150Pct',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, polygons, columns}) {
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values']
            });
            return {
              ...execution,
              keySpaces: [
                {
                  id: 'summary',
                  label: 'Summary moments, then Moran, Geary and G blocks',
                  size: 52,
                  indexSpace: '1d' as const
                }
              ],
              tags: {
                values: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Population below 150% poverty line (%)'
                },
                summary: {
                  role: 'output',
                  keySpace: 'summary',
                  meaning:
                    'n, S0, S1, S2, mean, variance, black, islands; then statistic, E, VN, zN, pN, VR, zR, pR for Moran (8), Geary (16), G (24). G uses row weights (weighted generalization); its two inference triples are identical analytical moments. NaN blocks not requested'
                }
              }
            };
          }
        })
    };
  }
});
