// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.key-join',
  entry: 'gpu-dataframe',
  element: 'GPUKeyJoin',
  title: 'Tract poverty joined onto Chicago iNaturalist observations by tract id',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-chicago-tracts',
    source: 'U.S. Census ACS and CDC/ATSDR SVI; Chicago showcase tracts',
    licence:
      'Census and CDC public domain; included nature-count data: iNaturalist contributors CC0 / CC BY 4.0 / CC BY-NC 4.0, non-commercial'
  },
  load: async () => import('./key-join.run')
});
