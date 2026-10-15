// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.empirical-bayes-rates',
  entry: 'gpu-spatial-analysis',
  element: 'GPUEmpiricalBayesRates',
  title: 'Empirical-Bayes smoothing: Chicago poverty rates',
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
        import('./empirical-bayes-rates.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.empirical-bayes-rates',
          tracts,
          entry: STATISTICS_OPERATORS['GPUEmpiricalBayesRates'],
          columns: {events: 'poverty150', populations: 'population'},
          choices: {clampPriorVariance: true},
          metadata: {
            // m = sum e / sum b: the kernel pools events and populations, never averages tract rates.
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
                  label: 'n, sum e, sum b, m, s2, a, clamped a, unused',
                  size: 8,
                  indexSpace: '1d' as const
                }
              ],
              tags: {
                events: {role: 'input', meaning: 'People below 150% of the poverty line (count)'},
                populations: {
                  role: 'input',
                  meaning: 'Population at risk per tract; zero marks an excluded row'
                },
                rawRates: {
                  role: 'intermediate',
                  polygons: true,
                  meaning: 'Crude poverty rate e / b before smoothing'
                },
                smoothedRates: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Empirical-Bayes smoothed poverty rate (NaN for excluded rows)'
                },
                summary: {
                  role: 'intermediate',
                  keySpace: 'moments',
                  meaning: 'Pooled rate m, weighted rate variance, prior variance a and its clamp'
                }
              }
            };
          }
        })
    };
  }
});
