// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-weights-summary',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialWeightsSummary',
  title: 'Moments and isolates of queen weights',
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
    const [{runExample}, {loadChicagoTracts}, {runTractWeightsScenario}, {WEIGHTS_OPERATORS}] =
      await Promise.all([
        import('./spatial-weights-summary.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/weights')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        runTractWeightsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-weights-summary',
          tracts,
          entry: WEIGHTS_OPERATORS['GPUSpatialWeightsSummary'],
          presentation: 'summary',
          choices: {
            symmetryTolerance: 0
          },
          metadata: {weightsStandardisation: 'binary'},
          runExample
        })
    };
  }
});
