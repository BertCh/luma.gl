// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-cost-matrix.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  createNodeValuePoints,
  getFacilityPositions,
  getSearchFacilities,
  getSearchKeySpaces
} from './search-lens-support';

const scenarioId = 'gpu-network.network-cost-matrix';
const roads = loadChicagoLoopRoads();
const facilities = getSearchFacilities(roads);

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['matrixCosts', 'converged', 'snappedEdges', 'startNodes', 'startCosts'],
  formulas: ['gpu-network.snapping.nearest'],
  keySpaces: getSearchKeySpaces(roads, facilities.length),
  entityKeySpace: 'node',
  queuesResource: 'network-cost-matrix-batch-0-queues',
  choices: [
    {
      stage: 'search',
      name: 'startsPerRow',
      value: 2,
      alternatives: [1],
      sweepable: false
    }
  ],
  async build(context) {
    const facilityPositions = context.upload(
      'facilityPositions',
      getFacilityPositions(context.data),
      'float32x2'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      facilityPositions
    });
    return {
      views: {
        facilityPositions,
        matrixCosts: example.costs,
        converged: example.converged,
        startNodes: example.startNodes,
        startCosts: example.startCosts
      }
    };
  },
  capture(final) {
    if (final('converged').values[0] === 0)
      throw new Error('Network cost matrix did not converge within maximumIterations = 24');
    const positions = final('nodePositions');
    const matrix = final('matrixCosts');
    const nodeCount = positions.values.length / 2;
    return [
      {
        ...final('facilityPositions'),
        keySpace: undefined,
        role: 'input',
        meaning: 'input: facility longitude/latitude as uploaded (row i of the matrix)',
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      },
      {
        ...matrix,
        id: 'matrixCosts',
        keySpace: 'node',
        values: Float32Array.from(Array.from(matrix.values).slice(0, nodeCount)),
        meaning: `row 0 of the ${facilities.length} x ${nodeCount} facility-by-node matrix (${facilities[0].name}); travel seconds from the snapped facility, +Infinity unreached`
      },
      createNodeValuePoints(positions, {
        id: 'matrix-row-nodes',
        valuesCapture: 'matrixCosts',
        meaning: `output: row 0 of the cost matrix (${facilities[0].name}) on every intersection, coloured by travel seconds`
      })
    ];
  }
};

/** Executes the snapping and cost matrix on the Chicago Loop roads and reads them back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
