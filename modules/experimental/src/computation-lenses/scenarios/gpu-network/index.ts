// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network lens scenarios (side-effect imports): the network pack first, then one
// import line per scenario registration file.

import '../../../gpu-network/computation-pack/index';
import {registerLensScenario} from '../../registry';
import './register-search';
import './register-structure';
import './register-construction';

const dataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};
registerLensScenario({
  id: 'gpu-network.network-reachability',
  entry: 'gpu-network',
  element: 'GPUNetworkReachability',
  title: 'Chicago Loop reachability wavefront',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'iterations', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-reachability.lens')
});
registerLensScenario({
  id: 'gpu-network.network-reachability.chained',
  entry: 'gpu-network',
  element: 'GPUNetworkReachability',
  title: 'Chicago Loop chained frontier rounds',
  audience: 'analyst',
  rungs: ['result', 'iterations', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  example: {path: 'scenarios/gpu-network/network-reachability.example.ts'},
  load: () => import('./network-reachability.chained.lens')
});
