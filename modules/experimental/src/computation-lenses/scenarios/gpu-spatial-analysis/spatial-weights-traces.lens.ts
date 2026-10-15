// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerRegressionLens} from './s1-regression-lens';

registerRegressionLens({
  slug: 'spatial-weights-traces',
  element: 'GPUSpatialWeightsTraces',
  title: 'Powers of the row-standardised queen weights of Chicago tracts',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  withMap: false,
  choices: {
    weightsCriterion: 'queen',
    weightsTransform: 'row',
    mode: 'exact',
    maximumPower: 12
  },
  metadata: {weightsStandardisation: 'row'},
  loadExample: () => import('./spatial-weights-traces.example'),
  prepare(execution) {
    const tag = (name: string, meaning: string) => ({
      role: 'output' as const,
      keySpace: 'power',
      meaning: `${name}; rows: matrix power q = 1 to 12; ${meaning}`
    });
    return {
      views: {
        powerTraces: execution.powerTraces,
        powerSums: execution.powerSums,
        infinityNorm: execution.output.infinityNorm,
        weightsOverflow: execution.weightsOverflow
      },
      tags: {
        powerTraces: tag('powerTraces', 'tr((W/s)^q)'),
        powerSums: tag('powerSums', "1'(W/s)^q 1")
      },
      keySpaces: [{id: 'power', label: 'Matrix power q', size: 12, indexSpace: '1d'}]
    };
  }
});
