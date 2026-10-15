// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-neighborhood.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  createNodeValuePoints,
  gatherNodePoints,
  getSearchKeySpaces,
  maskNoneSentinel
} from './search-lens-support';

const scenarioId = 'gpu-network.network-neighborhood';
const roads = loadChicagoLoopRoads();
/** Hop radius of the ego network. */
const HOPS = 6;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: roads.startNode,
  retain: ['hopDistances'],
  formulas: ['gpu-network.neighborhood.hops'],
  keySpaces: getSearchKeySpaces(roads, 0),
  entityKeySpace: 'node',
  queuesResource: 'network-neighborhood-reachability-queues',
  choices: [{stage: 'search', name: 'hops', value: HOPS, alternatives: [3, 8], sweepable: true}],
  async build(context) {
    const startNodes = context.upload('startNodes', Uint32Array.of(roads.startNode), 'uint32');
    const hops = context.upload('hops', Uint32Array.of(HOPS), 'uint32');
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      startNodes,
      hops
    });
    return {views: {hopDistances: example.hopDistances}};
  },
  capture(final) {
    const positions = final('nodePositions');
    return [
      gatherNodePoints(positions, [roads.startNode], {
        id: 'start-intersection',
        meaning: 'input: start intersection (position gathered from the nodePositions readback)'
      }),
      {
        ...maskNoneSentinel(
          final('hopDistances'),
          'GPU hopDistances readback; the NONE sentinel (0xffffffff, outside the ego network) is shown as no data'
        ),
        keySpace: 'node'
      },
      createNodeValuePoints(positions, {
        id: 'ego-network-nodes',
        valuesCapture: 'hopDistances',
        meaning: `output: intersections within ${HOPS} hops of the start, coloured by hop distance; others are no data`
      })
    ];
  }
};

/** Executes the hop-radius neighborhood on the Chicago Loop roads and reads it back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
