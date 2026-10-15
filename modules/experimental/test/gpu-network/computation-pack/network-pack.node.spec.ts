// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as pack from '../../../src/gpu-network/computation-pack/index';
import {createDefaultViewRegistry} from '../../../src/computation-views/registry/default-registry';

const elements = [
  'GPUNetworkReachability',
  'GPUNetworkServiceAreas',
  'GPUNetworkPathExtraction',
  'GPUNetworkNeighborhood',
  'GPUNetworkIsochrones',
  'addDriveTimeCatchmentRecipe',
  'GPUNetworkCostTable',
  'GPUNetworkSnapping',
  'GPUNetworkCostMatrix',
  'GPUNetworkAccessibility',
  'GPUNetworkCostWeights',
  'GPUNetworkCentrality',
  'GPUNetworkAnalyticsColumns',
  'GPUNetworkStatistics',
  'GPUNetworkSubgraphFilter',
  'GPUNetworkCoarsening',
  'GPUAdjacencyMatrix',
  'GPUAdjacencyMatrixOrder',
  'GPUNetworkNoding',
  'GPUNetworkLineGraph',
  'GPUMapMatching',
  'GPUNetworkKFunction',
  'GPUEdgeBundling',
  'GPUFlowAggregation'
];

test('all 24 network elements have a lawful scope and reduction', () => {
  expect(Object.keys(pack.NETWORK_OPERATORS).sort()).toEqual(elements.sort());
  const scopes = ['local', 'focal', 'block', 'zonal', 'global', 'frontier', 'pairwise', 'solver'];
  const reductions = [
    'none',
    'count',
    'sum',
    'weighted-mean',
    'argmin',
    'histogram',
    'mode',
    'select',
    'scan-sort',
    'solve'
  ];
  for (const operator of Object.values(pack.NETWORK_OPERATORS)) {
    expect(scopes).toContain(operator.scope);
    if (operator.scopeSecondary) expect(scopes).toContain(operator.scopeSecondary);
    expect(reductions).toContain(operator.reduction);
  }
});

test('pack imports without a device or exported GPU contributor and registers the wavefront', () => {
  expect(
    Object.values(pack).some(value => typeof value === 'function' && /^GPUNetwork/.test(value.name))
  ).toBe(false);
  expect(
    createDefaultViewRegistry()
      .viewsForRung('iterations')
      .map(view => view.id)
  ).toContain(pack.NETWORK_WAVEFRONT_VIEW.id);
});
