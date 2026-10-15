// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.data-frame-global-sort-query',
  entry: 'gpu-dataframe',
  element: 'GPUDataFrameGlobalSortQuery',
  title: 'Highest poverty rates across two tract batches',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-chicago-tracts',
    source: 'U.S. Census ACS and CDC/ATSDR SVI; Chicago showcase tracts',
    licence:
      'Census and CDC public domain; included nature-count data: iNaturalist contributors CC0 / CC BY 4.0 / CC BY-NC 4.0, non-commercial'
  },
  load: async () => import('./data-frame-global-sort-query.run')
});
