// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-graph.graph-degree',
  entry: 'gpu-graph',
  element: 'GPUGraphDegree',
  title: 'Outgoing connections per Dutch rail station',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula'],
  defaultRung: 'result',
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-degree.run')
});
