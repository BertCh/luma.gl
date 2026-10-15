// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const PERMUTATIONS = 99;

registerLensScenario({
  id: 'gpu-spatial-analysis.global-permutation-test',
  entry: 'gpu-spatial-analysis',
  element: 'GPUGlobalPermutationTest',
  title: 'Global Moran permutation test: Chicago poverty percentages',
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
        import('./global-permutation-test.example'),
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
          id: 'gpu-spatial-analysis.global-permutation-test',
          tracts,
          entry: STATISTICS_OPERATORS['GPUGlobalPermutationTest'],
          columns: {values: 'poverty150Pct'},
          choices: {
            transform: 'row',
            islands: 'exclude',
            statistic: 'moran',
            alternative: 'two-sided',
            permutations: PERMUTATIONS
          },
          metadata: {
            // A single pseudo p-value: there is no multiple-testing family.
            pCorrection: 'none',
            significanceAvailable: true,
            weightsStandardisation: 'row',
            valueColumn: 'poverty150Pct',
            mappedVariableExtent: 'intensive',
            seed: 1
          },
          async execute({graph, polygons, columns, upload}) {
            const parameters = upload(
              'parameters',
              spatial.getGPUPermutationParameterValues(
                {seed: 1, permutationCount: PERMUTATIONS},
                {maximumPermutations: PERMUTATIONS}
              ),
              'uint32'
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
                {id: 'summary', label: 'Permutation summary', size: 12, indexSpace: '1d' as const},
                {
                  id: 'permutation',
                  label: 'Permutation',
                  size: PERMUTATIONS,
                  indexSpace: '1d' as const
                },
                {id: 'bin', label: 'Histogram bin', size: 20, indexSpace: '1d' as const}
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
                    'observed I, pseudo p, exceedances M, permutations P, simulated mean, sd, z, normal p, minimum, maximum, islands, n'
                },
                referenceDistribution: {
                  role: 'output',
                  keySpace: 'permutation',
                  meaning: 'Moran I of each of the 99 relabelings (readback, not a replay)'
                },
                histogram: {role: 'intermediate', keySpace: 'bin'}
              }
            };
          }
        })
    };
  }
});
