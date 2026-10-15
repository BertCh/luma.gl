// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.multiple-testing-adjustment',
  entry: 'gpu-spatial-analysis',
  element: 'GPUMultipleTestingAdjustment',
  title: 'False discovery rate: Chicago hot spot p-values',
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
        import('./multiple-testing-adjustment.example'),
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
          id: 'gpu-spatial-analysis.multiple-testing-adjustment',
          tracts,
          entry: STATISTICS_OPERATORS['GPUMultipleTestingAdjustment'],
          columns: {values: 'poverty150Pct'},
          choices: {method: 'benjamini-hochberg', minimumFamilySize: 0, significanceLevel: 0.05},
          metadata: {
            pCorrection: 'benjamini-hochberg',
            significanceAvailable: true,
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
              keySpaces: [
                {id: 'family', label: 'Tested family size', size: 1, indexSpace: '1d' as const}
              ],
              tags: {
                values: {
                  role: 'intermediate',
                  meaning: 'Population below 150% poverty line (%)'
                },
                pValues: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Unadjusted two-sided normal p-value of the hot spot z-score'
                },
                adjustedPValues: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Benjamini-Hochberg adjusted p-value (significant when at most alpha)'
                },
                zScores: {role: 'intermediate', meaning: 'Gi* z-score'},
                testedCount: {
                  role: 'intermediate',
                  keySpace: 'family',
                  meaning: 'Family size m actually used'
                }
              }
            };
          }
        })
    };
  }
});
