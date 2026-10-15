// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.temporal-reduction',
  entry: 'gpu-dataframe',
  element: 'GPUTemporalReduction',
  title: 'Chicago nature observations reduced to weekly counts per taxon group',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-chicago-nature',
    source: 'iNaturalist observations, Chicago, 2023; showcase chicago-nature v1',
    licence:
      'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial'
  },
  load: async () => import('./temporal-reduction.run')
});
