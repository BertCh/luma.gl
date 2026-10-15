// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.assuncao-reis-standardization',
  entry: 'gpu-spatial-analysis',
  element: 'GPUAssuncaoReisStandardization',
  title: 'Assuncao-Reis standardization: Chicago poverty rates',
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
        import('./assuncao-reis-standardization.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.assuncao-reis-standardization',
          tracts,
          entry: STATISTICS_OPERATORS['GPUAssuncaoReisStandardization'],
          columns: {events: 'poverty150', populations: 'population'},
          choices: {},
          metadata: {
            // b = sum e / sum n over the valid rows, the pooled rate (not a mean of tract rates).
            ebReferenceRate: 'pooled',
            denominator: 'population',
            valueColumn: 'poverty150',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, columns}) {
            const execution = await runExample(device, {
              graph,
              events: columns['events'],
              populations: columns['populations']
            });
            return {
              ...execution,
              keySpaces: [
                {
                  id: 'moments',
                  label: 'Valid rows n, pooled rate b, variance s2, prior variance a, population',
                  size: 5,
                  indexSpace: '1d' as const
                },
                {id: 'status', label: 'Rejected rows', size: 1, indexSpace: '1d' as const}
              ],
              tags: {
                events: {
                  role: 'input',
                  polygons: true,
                  meaning: 'People below 150% of the poverty line (count)'
                },
                populations: {
                  role: 'input',
                  meaning: 'Population at risk per tract; zero marks a missing row'
                },
                standardizedRates: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Empirical-Bayes standardized poverty rate z (NaN for missing rows)'
                },
                statistics: {
                  role: 'intermediate',
                  keySpace: 'moments',
                  meaning: 'n, b, s2, a (unclamped), total population'
                },
                invalidCount: {
                  role: 'intermediate',
                  keySpace: 'status',
                  meaning: 'Unmasked rows rejected as missing'
                }
              }
            };
          }
        })
    };
  }
});
