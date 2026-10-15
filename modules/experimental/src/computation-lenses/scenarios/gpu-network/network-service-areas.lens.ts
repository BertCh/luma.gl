// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-service-areas.example';
import {
  createNodeValuePoints,
  gatherNodePoints,
  getSearchFacilities,
  getSearchKeySpaces
} from './search-lens-support';
import {loadChicagoLoopRoads} from './network-lens-support';

const scenarioId = 'gpu-network.network-service-areas';

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['assignments', 'costs', 'converged'],
  formulas: ['gpu-network.service-areas.label'],
  keySpaces: getSearchKeySpaces(loadChicagoLoopRoads(), 0),
  entityKeySpace: 'node',
  queuesResource: 'network-service-areas-reachability-queues',
  choices: [
    {
      stage: 'assignment',
      name: 'facilities',
      value: getSearchFacilities(loadChicagoLoopRoads()).length,
      alternatives: [loadChicagoLoopRoads().facilities.length],
      sweepable: false
    }
  ],
  async build(context) {
    const facilityNodes = getSearchFacilities(context.data).map(facility => facility.nearestNode);
    const uploaded = context.upload('facilityNodes', Uint32Array.from(facilityNodes), 'uint32');
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      facilityNodes: uploaded
    });
    return {
      views: {
        assignments: example.assignments,
        costs: example.costs,
        converged: example.converged,
        facilityNodes: uploaded
      }
    };
  },
  capture(final) {
    // Cost and label phases must both have reached a fixpoint within their round bounds.
    if (final('converged').values[0] === 0)
      throw new Error('Network service areas did not converge within maximumIterations = 24');
    const positions = final('nodePositions');
    return [
      gatherNodePoints(positions, Array.from(final('facilityNodes').values), {
        id: 'facility-nodes',
        meaning:
          'input: facility intersections (prepared nearest nodes from the data file; positions gathered from the nodePositions readback)'
      }),
      {...final('assignments'), keySpace: 'node'},
      {...final('costs'), keySpace: 'node'},
      createNodeValuePoints(positions, {
        id: 'served-nodes',
        valuesCapture: 'assignments',
        meaning: 'output: every intersection coloured by the facility row that serves it'
      })
    ];
  }
};

/** Executes the service-area assignment on the Chicago Loop roads and reads it back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
