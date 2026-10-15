// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getGPUNetworkAccessibilityParameterValues} from '@luma.gl/experimental/gpu-network';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample as runCostMatrixExample} from './network-cost-matrix.example';
import {runExample} from './network-accessibility.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  createNodeValuePoints,
  getFacilityPositions,
  getSearchFacilities,
  getSearchKeySpaces,
  reverseRoads
} from './search-lens-support';

const scenarioId = 'gpu-network.network-accessibility';
const roads = loadChicagoLoopRoads();
const facilities = getSearchFacilities(roads);
/** Cost cutoff in seconds and decay rate per second of the exponential gravity score. */
const THRESHOLD_SECONDS = 600;
const BETA = 0.005;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  // Rows search the reversed roads, so a matrix entry is the travel time from a node to a facility.
  prepareRoads: reverseRoads,
  seed: 0,
  retain: ['matrixCosts', 'converged', 'snappedEdges', 'startNodes', 'startCosts', 'gravity'],
  formulas: ['gpu-network.accessibility.gravity', 'gpu-network.snapping.nearest'],
  keySpaces: getSearchKeySpaces(roads, facilities.length),
  entityKeySpace: 'node',
  queuesResource: 'network-cost-matrix-batch-0-queues',
  choices: [
    {
      stage: 'summary',
      name: 'decay',
      value: 'exponential',
      alternatives: ['none', 'power'],
      sweepable: true
    }
  ],
  async build(context) {
    const facilityPositions = context.upload(
      'facilityPositions',
      getFacilityPositions(context.data),
      'float32x2'
    );
    const matrix = await runCostMatrixExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      facilityPositions
    });
    // Unit opportunities: a demonstrative count of facilities, not their capacity.
    const opportunityWeights = context.upload(
      'opportunityWeights',
      Float32Array.from(facilities, () => 1),
      'float32'
    );
    const parameters = context.upload(
      'parameters',
      getGPUNetworkAccessibilityParameterValues({
        threshold: THRESHOLD_SECONDS,
        decay: 'exponential',
        beta: BETA
      }),
      'float32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      matrixCosts: matrix.costs,
      opportunityWeights,
      parameters,
      nodeCount: context.data.nodeCount
    });
    return {
      views: {
        matrixCosts: matrix.costs,
        converged: matrix.converged,
        gravity: example.gravity
      }
    };
  },
  capture(final) {
    if (final('converged').values[0] === 0)
      throw new Error('Network accessibility: the cost matrix did not converge');
    const positions = final('nodePositions');
    const gravity = final('gravity');
    const scores = Array.from(gravity.values);
    const highest = scores.indexOf(Math.max(...scores));
    const matrix = final('matrixCosts');
    const nodeCount = positions.values.length / 2;
    return [
      {
        ...positions,
        id: 'facility-positions',
        name: 'facility-positions',
        keySpace: undefined,
        values: Float32Array.from(facilities.flatMap(facility => [facility.lon, facility.lat])),
        role: 'input',
        meaning: 'input: facility longitude/latitude (opportunities, unit weight each)',
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      },
      {
        ...matrix,
        id: 'matrixCosts',
        keySpace: 'node',
        values: Float32Array.from(Array.from(matrix.values).slice(0, nodeCount)),
        meaning: `row 0 of the ${facilities.length} x ${nodeCount} matrix: travel seconds from each node to ${facilities[0].name}`
      },
      {...gravity, keySpace: 'node'},
      createNodeValuePoints(positions, {
        id: 'gravity-nodes',
        valuesCapture: 'gravity',
        highlight: {key: highest, label: 'highest access'},
        meaning: `output: gravity accessibility per intersection, sum of exp(-${BETA} cost) over facilities within ${THRESHOLD_SECONDS} s`
      })
    ];
  }
};

/** Executes snapping, the cost matrix and the gravity score on the Chicago Loop roads. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
