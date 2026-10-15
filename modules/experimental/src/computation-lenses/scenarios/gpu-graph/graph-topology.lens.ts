// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-graph.graph-topology',
  entry: 'gpu-graph',
  element: 'GPUGraphTopology',
  title: 'Reusable adjacency for the Dutch rail graph',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula'],
  defaultRung: 'result',
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-topology.run')
});
