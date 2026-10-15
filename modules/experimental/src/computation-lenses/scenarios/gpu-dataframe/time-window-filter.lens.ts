// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.time-window-filter',
  entry: 'gpu-dataframe',
  element: 'GPUTimeWindowFilter',
  title: 'Chicago nature observations in a summer window',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-chicago-nature',
    source: 'iNaturalist observations, Chicago, 2023; showcase chicago-nature v1',
    licence:
      'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial'
  },
  load: async () => import('./time-window-filter.run')
});
