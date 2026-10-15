// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-graph.graph-breadth-first-search',
  entry: 'gpu-graph',
  element: 'GPUGraphBreadthFirstSearch',
  title: 'Rail hops from Utrecht Centraal',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'iterations'],
  defaultRung: 'result',
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-breadth-first-search.run')
});
