// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-weights-transform',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialWeightsTransform',
  title: 'Row-standardised queen weights on Chicago tracts',
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
        import('./spatial-weights-transform.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/weights')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        runTractWeightsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-weights-transform',
          tracts,
          entry: WEIGHTS_OPERATORS['GPUSpatialWeightsTransform'],
          presentation: 'graph',
          choices: {
            operation: 'row'
          },
          metadata: {weightsStandardisation: 'row'},
          runExample
        })
    };
  }
});
