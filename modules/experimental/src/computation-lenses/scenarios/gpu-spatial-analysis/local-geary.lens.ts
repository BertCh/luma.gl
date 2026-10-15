// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.local-geary',
  entry: 'gpu-spatial-analysis',
  element: 'GPULocalGeary',
  title: 'Local Geary: Chicago poverty percentages',
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
    const [{runExample}, {loadChicagoTracts}, utilities, spatial, {STATISTICS_OPERATORS}] =
      await Promise.all([
        import('./local-geary.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('@luma.gl/experimental/gpu-spatial-analysis'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.local-geary',
          tracts,
          entry: STATISTICS_OPERATORS['GPULocalGeary'],
          columns: {values: 'poverty150Pct'},
          choices: {transform: 'row', islands: 'exclude'},
          metadata: {
            // Local Geary has no analytic p-value: a cluster class needs permutation inference.
            significanceAvailable: false,
            weightsStandardisation: 'row',
            valueColumn: 'poverty150Pct',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, polygons, columns, upload}) {
            const parameters = upload(
              'parameters',
              spatial.getGPUSpatialAutocorrelationParameterValues({}),
              'float32'
            );
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values'],
              parameters
            });
            return {
              ...execution,
              tags: {
                values: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Population below 150% poverty line (%)'
                },
                statistic: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Local Geary c_i (small = similar neighbours; not significance)'
                }
              }
            };
          }
        })
    };
  }
});
