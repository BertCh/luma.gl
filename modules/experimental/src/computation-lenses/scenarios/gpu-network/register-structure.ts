// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network structure-family lens scenarios (one registerLensScenario call per scenario).

import {registerLensScenario} from '../../registry';

const dataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};

registerLensScenario({
  id: 'gpu-network.network-analytics-columns',
  entry: 'gpu-network',
  element: 'GPUNetworkAnalyticsColumns',
  title: 'Chicago Loop intersection degrees',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-analytics-columns.lens')
});

registerLensScenario({
  id: 'gpu-network.network-centrality',
  entry: 'gpu-network',
  element: 'GPUNetworkCentrality',
  title: 'Chicago Loop closeness of eight sampled pivots',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-centrality.lens')
});

registerLensScenario({
  id: 'gpu-network.network-statistics',
  entry: 'gpu-network',
  element: 'GPUNetworkStatistics',
  title: 'Chicago Loop network summary statistics',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-statistics.lens')
});

registerLensScenario({
  id: 'gpu-network.network-subgraph-filter',
  entry: 'gpu-network',
  element: 'GPUNetworkSubgraphFilter',
  title: 'Chicago Loop short-block subgraph',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-subgraph-filter.lens')
});

registerLensScenario({
  id: 'gpu-network.network-coarsening',
  entry: 'gpu-network',
  element: 'GPUNetworkCoarsening',
  title: 'Chicago Loop coarsened to four zones',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-coarsening.lens')
});

registerLensScenario({
  id: 'gpu-network.adjacency-matrix',
  entry: 'gpu-network',
  element: 'GPUAdjacencyMatrix',
  title: 'Chicago Loop adjacency matrix in zone order',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./adjacency-matrix.lens')
});

registerLensScenario({
  id: 'gpu-network.adjacency-matrix-order',
  entry: 'gpu-network',
  element: 'GPUAdjacencyMatrixOrder',
  title: 'Chicago Loop intersections ordered by zone',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./adjacency-matrix-order.lens')
});

registerLensScenario({
  id: 'gpu-network.network-line-graph',
  entry: 'gpu-network',
  element: 'GPUNetworkLineGraph',
  title: 'Chicago Loop turn-cost line graph',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-line-graph.lens')
});
