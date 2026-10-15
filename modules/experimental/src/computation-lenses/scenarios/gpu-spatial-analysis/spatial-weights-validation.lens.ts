// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-weights-validation',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialWeightsValidation',
  title: 'Validity of queen weights on Chicago tracts',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
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
        import('./spatial-weights-validation.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/weights')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        runTractWeightsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-weights-validation',
          tracts,
          entry: WEIGHTS_OPERATORS['GPUSpatialWeightsValidation'],
          presentation: 'validation',
          choices: {},
          metadata: {weightsStandardisation: 'binary'},
          runExample
        })
    };
  }
});
