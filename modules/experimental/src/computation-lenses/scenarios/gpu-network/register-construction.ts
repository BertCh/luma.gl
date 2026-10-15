// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network construction-family lens scenarios (one registerLensScenario call per scenario).

import {registerLensScenario} from '../../registry';

const roadsDataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};
const polylinesDataset = {
  id: 'n-chicago-loop-polylines',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract (raw centrelines)'
};
const tracesDataset = {
  id: 'n-chicago-loop-traces',
  licence: 'Synthetic, derived from OpenStreetMap data (ODbL 1.0)',
  source: 'SIMULATED GPS traces on OpenStreetMap roads (not real vehicles)'
};
const crashesDataset = {
  id: 'n-chicago-loop-crashes',
  licence: 'City of Chicago Data Portal Terms of Use',
  source: 'City of Chicago traffic crashes 2023 (sampled) on OpenStreetMap roads'
};
const flowsDataset = {
  id: 'n-chicago-od-flows',
  licence: 'City of Chicago Data Portal Terms of Use',
  source: 'City of Chicago taxi trips 2023 between community areas'
};

registerLensScenario({
  id: 'gpu-network.network-cost-table',
  entry: 'gpu-network',
  element: 'GPUNetworkCostTable',
  title: 'Chicago Loop 30-second travel-cost table',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: roadsDataset,
  load: () => import('./network-cost-table.lens')
});

registerLensScenario({
  id: 'gpu-network.network-cost-weights',
  entry: 'gpu-network',
  element: 'GPUNetworkCostWeights',
  title: 'Chicago Loop facility catchment weights',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: roadsDataset,
  load: () => import('./network-cost-weights.lens')
});

registerLensScenario({
  id: 'gpu-network.network-noding',
  entry: 'gpu-network',
  element: 'GPUNetworkNoding',
  title: 'Chicago Loop centrelines noded into a network',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: polylinesDataset,
  load: () => import('./network-noding.lens')
});

registerLensScenario({
  id: 'gpu-network.map-matching',
  entry: 'gpu-network',
  element: 'GPUMapMatching',
  title: 'Simulated GPS tracks matched to Loop roads',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: tracesDataset,
  load: () => import('./map-matching.lens')
});

registerLensScenario({
  id: 'gpu-network.network-k-function',
  entry: 'gpu-network',
  element: 'GPUNetworkKFunction',
  title: 'Network K function of Chicago Loop crashes',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: crashesDataset,
  load: () => import('./network-k-function.lens')
});

registerLensScenario({
  id: 'gpu-network.edge-bundling',
  entry: 'gpu-network',
  element: 'GPUEdgeBundling',
  title: 'Chicago taxi flows bundled by density',
  audience: 'analyst',
  rungs: ['result', 'datum', 'iterations', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: flowsDataset,
  load: () => import('./edge-bundling.lens')
});

registerLensScenario({
  id: 'gpu-network.flow-aggregation',
  entry: 'gpu-network',
  element: 'GPUFlowAggregation',
  title: 'Chicago taxi flows ranked by zone pair',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: flowsDataset,
  load: () => import('./flow-aggregation.lens')
});
