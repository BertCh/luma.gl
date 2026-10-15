// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.ripley',
  entry: 'gpu-dataframe',
  element: 'GPURipley',
  title: "Ripley's K and L of Chicago iNaturalist observations",
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-chicago-nature',
    source: 'iNaturalist observations, Chicago 2023 (api.inaturalist.org)',
    licence:
      'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial'
  },
  load: async () => import('./ripley.run')
});
