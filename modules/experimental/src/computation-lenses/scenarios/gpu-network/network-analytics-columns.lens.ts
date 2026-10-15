// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-analytics-columns.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {createNodeValuePoints} from './search-lens-support';
import {getMaximumKey, getStructureKeySpaces} from './structure-lens-support';

const scenarioId = 'gpu-network.network-analytics-columns';
const roads = loadChicagoLoopRoads();

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['degree', 'normalized', 'extent'],
  formulas: ['gpu-network.analytics.normalize'],
  // The nested gpu-graph degree kernels name their rows `vertex`.
  keySpaces: getStructureKeySpaces(roads, [
    {id: 'vertex', label: 'graph vertex', indexSpace: '1d', size: roads.nodeCount}
  ]),
  entityKeySpace: 'node',
  choices: [
    {
      stage: 'representation',
      name: 'metric',
      value: 'outgoing degree',
      alternatives: ['in-degree', 'PageRank', 'core number', 'components', 'communities'],
      sweepable: false
    },
    {
      stage: 'summary',
      name: 'normalization',
      value: 'min-max over all nodes',
      alternatives: ['raw column'],
      sweepable: false
    }
  ],
  async build(context) {
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology
    });
    return {
      views: {degree: example.degree, normalized: example.normalized, extent: example.extent}
    };
  },
  capture(final) {
    const positions = final('nodePositions');
    const degree = {
      ...final('degree'),
      keySpace: 'node',
      meaning: 'GPU outgoing degree of every intersection (CSR row length)'
    };
    return [
      degree,
      {
        ...final('normalized'),
        keySpace: 'node',
        meaning: 'GPU min-max scaled degree, (degree - lowest) / (highest - lowest)'
      },
      {...final('extent'), meaning: 'GPU extent of the degree column: [lowest, highest]'},
      createNodeValuePoints(positions, {
        id: 'degree-nodes',
        valuesCapture: 'degree',
        meaning: 'output: every intersection coloured by outgoing degree; the highest is marked',
        highlight: {key: getMaximumKey(degree.values), label: 'highest degree'}
      })
    ];
  }
};

/** Executes degree and normalization on the Chicago Loop roads and reads them back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
