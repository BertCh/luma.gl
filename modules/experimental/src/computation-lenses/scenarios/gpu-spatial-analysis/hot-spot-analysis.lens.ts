// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.hot-spot-analysis',
  entry: 'gpu-spatial-analysis',
  element: 'GPUHotSpotAnalysis',
  title: 'Hot spots: Chicago poverty percentages',
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
        import('./hot-spot-analysis.example'),
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
          id: 'gpu-spatial-analysis.hot-spot-analysis',
          tracts,
          entry: STATISTICS_OPERATORS['GPUHotSpotAnalysis'],
          columns: {values: 'poverty150Pct'},
          choices: {
            transform: 'binary',
            selfWeight: 1,
            islands: 'exclude',
            minimumNeighbors: 4,
            multipleTesting: 'benjamini-hochberg'
          },
          metadata: {
            pCorrection: 'benjamini-hochberg',
            significanceAvailable: true,
            weightsStandardisation: 'binary',
            valueColumn: 'poverty150Pct',
            mappedVariableExtent: 'intensive'
          },
          async execute({graph, polygons, columns, upload}) {
            // The bins use fixed 0.10, 0.05 and 0.01 levels; the level in the parameters is unused.
            const parameters = upload(
              'parameters',
              spatial.getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05}),
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
                zScores: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Gi* z-score (not significance)'
                },
                bins: {
                  role: 'intermediate',
                  meaning: 'Confidence bin: sign of z times the 90, 95 or 99% level'
                },
                pValues: {role: 'intermediate', meaning: 'Two-sided normal p-value of z'},
                adjustedPValues: {
                  role: 'intermediate',
                  meaning: 'Benjamini-Hochberg adjusted p-value'
                }
              }
            };
          }
        })
    };
  }
});
