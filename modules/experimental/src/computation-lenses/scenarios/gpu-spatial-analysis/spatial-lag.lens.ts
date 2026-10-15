// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-lag',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialLag',
  title: 'Neighbouring poverty percentage across Chicago tracts',
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
        import('./spatial-lag.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/weights')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        runTractWeightsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-lag',
          tracts,
          entry: WEIGHTS_OPERATORS['GPUSpatialLag'],
          presentation: 'lag',
          choices: {
            normalize: false,
            columnCount: 1
          },
          metadata: {weightsStandardisation: 'row'},
          runExample
        })
    };
  }
});
