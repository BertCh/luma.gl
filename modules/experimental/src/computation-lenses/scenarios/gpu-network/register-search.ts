// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-network search-family lens scenarios (one registerLensScenario call per scenario).

import {registerLensScenario} from '../../registry';

const dataset = {
  id: 'n-chicago-loop-roads',
  licence: 'ODbL 1.0',
  source: 'OpenStreetMap via BBBike Chicago extract'
};

registerLensScenario({
  id: 'gpu-network.network-service-areas',
  entry: 'gpu-network',
  element: 'GPUNetworkServiceAreas',
  title: 'Chicago Loop service areas of four facilities',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-service-areas.lens')
});

registerLensScenario({
  id: 'gpu-network.network-path-extraction',
  entry: 'gpu-network',
  element: 'GPUNetworkPathExtraction',
  title: 'Chicago Loop shortest path to the West Loop library',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-path-extraction.lens')
});

registerLensScenario({
  id: 'gpu-network.network-neighborhood',
  entry: 'gpu-network',
  element: 'GPUNetworkNeighborhood',
  title: 'Chicago Loop six-hop neighborhood',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-neighborhood.lens')
});

registerLensScenario({
  id: 'gpu-network.network-isochrones',
  entry: 'gpu-network',
  element: 'GPUNetworkIsochrones',
  title: 'Chicago Loop travel-time bands',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-isochrones.lens')
});

registerLensScenario({
  id: 'gpu-network.network-snapping',
  entry: 'gpu-network',
  element: 'GPUNetworkSnapping',
  title: 'Chicago Loop facilities snapped to the nearest road edge',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-snapping.lens')
});

registerLensScenario({
  id: 'gpu-network.network-cost-matrix',
  entry: 'gpu-network',
  element: 'GPUNetworkCostMatrix',
  title: 'Chicago Loop travel times from four facilities',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-cost-matrix.lens')
});

registerLensScenario({
  id: 'gpu-network.network-accessibility',
  entry: 'gpu-network',
  element: 'GPUNetworkAccessibility',
  title: 'Chicago Loop gravity access to four facilities',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./network-accessibility.lens')
});

registerLensScenario({
  id: 'gpu-network.drive-time-catchment-recipe',
  entry: 'gpu-network',
  element: 'addDriveTimeCatchmentRecipe',
  title: 'Chicago Loop drive-time catchment of four facilities',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset,
  load: () => import('./drive-time-catchment-recipe.lens')
});
