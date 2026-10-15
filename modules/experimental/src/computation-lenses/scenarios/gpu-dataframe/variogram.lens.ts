// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.variogram',
  entry: 'gpu-dataframe',
  element: 'GPUVariogram',
  title: 'Empirical semivariogram of Hurricane Helene rainfall at 1,200 stations',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-ghcn-stations',
    source: 'NOAA NCEI GHCN-Daily via AWS Open Data (noaa-ghcn-pds)',
    licence: 'NOAA NCEI GHCN-Daily, public domain'
  },
  load: async () => import('./variogram.run')
});
