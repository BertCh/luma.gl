// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-dataframe.spatial-correlogram',
  entry: 'gpu-dataframe',
  element: 'GPUSpatialCorrelogram',
  title: "Moran's I of Hurricane Helene rainfall by distance band",
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 'f-ghcn-stations',
    source: 'NOAA NCEI GHCN-Daily via AWS Open Data (noaa-ghcn-pds)',
    licence: 'NOAA NCEI GHCN-Daily, public domain'
  },
  load: async () => import('./spatial-correlogram.run')
});
