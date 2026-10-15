// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-graph.graph-label-propagation',
  entry: 'gpu-graph',
  element: 'GPUGraphLabelPropagation',
  title: 'Majority-vote communities on the Dutch rail graph',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'iterations'],
  defaultRung: 'result',
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-label-propagation.run')
});
