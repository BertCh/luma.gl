// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-empirical-bayes-rates',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialEmpiricalBayesRates',
  title: 'Spatial empirical-Bayes smoothing: Chicago poverty rates',
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
        import('./spatial-empirical-bayes-rates.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-empirical-bayes-rates',
          tracts,
          entry: STATISTICS_OPERATORS['GPUSpatialEmpiricalBayesRates'],
          columns: {events: 'poverty150', populations: 'population'},
          choices: {criterion: 'queen'},
          metadata: {
            // m_i = E_i / B_i pools events and populations over the neighbourhood.
            ebReferenceRate: 'pooled',
            denominator: 'population',
            valueColumn: 'poverty150',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, polygons, columns}) {
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              events: columns['events'],
              populations: columns['populations']
            });
            return {
              ...execution,
              tags: {
                events: {role: 'input', meaning: 'People below 150% of the poverty line (count)'},
                populations: {
                  role: 'input',
                  meaning: 'Population at risk per tract; zero marks an excluded row'
                },
                spatialRates: {
                  role: 'intermediate',
                  polygons: true,
                  meaning: 'Spatial rate: pooled neighbourhood poverty rate E / B'
                },
                smoothedRates: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Spatial empirical-Bayes smoothed poverty rate'
                },
                priorMeans: {
                  role: 'intermediate',
                  meaning: 'Local prior mean of the neighbourhood'
                },
                priorVariances: {
                  role: 'intermediate',
                  meaning: 'Local prior variance a, clamped at zero'
                }
              }
            };
          }
        })
    };
  }
});
