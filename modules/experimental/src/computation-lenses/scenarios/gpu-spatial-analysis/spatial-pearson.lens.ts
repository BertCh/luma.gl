// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-pearson',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialPearson',
  title: 'Spatial Pearson: poverty against neighbouring uninsured rates',
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
        import('./spatial-pearson.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-pearson',
          tracts,
          entry: STATISTICS_OPERATORS['GPUSpatialPearson'],
          columns: {values: 'poverty150Pct', secondValues: 'uninsuredPct'},
          choices: {transform: 'row'},
          metadata: {
            weightsStandardisation: 'row',
            valueColumn: 'poverty150Pct',
            secondValueColumn: 'uninsuredPct',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, polygons, columns}) {
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values'],
              secondValues: columns['secondValues']
            });
            return {
              ...execution,
              keySpaces: [
                {id: 'statistic', label: 'Global statistic', size: 1, indexSpace: '1d' as const}
              ],
              tags: {
                values: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Population below 150% poverty line (%)'
                },
                secondValues: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Uninsured population (%)'
                },
                statistic: {
                  role: 'output',
                  keySpace: 'statistic',
                  meaning: 'Edge-weighted Pearson r of x at the tract and y at its neighbour'
                }
              }
            };
          }
        })
    };
  }
});
