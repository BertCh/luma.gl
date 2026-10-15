// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getGPUNetworkCostWeightsParameterValues} from '@luma.gl/experimental/gpu-network';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample as runCostMatrixExample} from './network-cost-matrix.example';
import {runExample} from './network-cost-weights.example';
import {gatherRows, sliceCapture} from './construction-lens-support';
import {loadChicagoLoopRoads} from './network-lens-support';
import {getFacilityPositions, getSearchFacilities, getSearchKeySpaces} from './search-lens-support';

const scenarioId = 'gpu-network.network-cost-weights';
const roads = loadChicagoLoopRoads();
const facilities = getSearchFacilities(roads);
/** Catchment: nodes within 240 s of a facility keep a weight exp(-0.01 * seconds). */
const COST_LIMIT_SECONDS = 240;
const DECAY_BETA = 0.01;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: [
    'matrixCosts',
    'converged',
    'snappedEdges',
    'startNodes',
    'startCosts',
    'network-cost-weights-row-counts',
    'weightOffsets',
    'weightNeighbors',
    'weightValues',
    'weightDistances',
    'weightOverflow'
  ],
  formulas: ['gpu-network.cost-weights.decay', 'gpu-network.snapping.nearest'],
  keySpaces: [...getSearchKeySpaces(roads, facilities.length)],
  entityKeySpace: 'facility',
  queuesResource: 'network-cost-matrix-batch-0-queues',
  choices: [
    {
      stage: 'impedance',
      name: 'costLimitSeconds',
      value: COST_LIMIT_SECONDS,
      alternatives: [120, 600],
      sweepable: true
    },
    {
      stage: 'impedance',
      name: 'decay',
      value: 'exponential',
      alternatives: ['none', 'power', 'gaussian'],
      sweepable: false
    }
  ],
  async build(context) {
    const facilityPositions = context.upload(
      'facilityPositions',
      getFacilityPositions(context.data),
      'float32x2'
    );
    // Prerequisite: the facility-by-node matrix of travel seconds from snapping and a cost matrix.
    const matrix = await runCostMatrixExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      facilityPositions
    });
    const parameters = context.upload(
      'weightParameters',
      getGPUNetworkCostWeightsParameterValues({
        costLimit: COST_LIMIT_SECONDS,
        decay: 'exponential',
        beta: DECAY_BETA
      }),
      'float32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      matrixCosts: matrix.costs,
      parameters,
      facilityCount: facilities.length,
      nodeCount: context.data.nodeCount
    });
    return {
      views: {
        facilityPositions,
        matrixCosts: matrix.costs,
        converged: matrix.converged,
        weightOffsets: example.offsets,
        weightNeighbors: example.neighbors,
        weightValues: example.weights,
        weightDistances: example.distances,
        weightOverflow: example.status.overflow!,
        'network-cost-weights-row-counts': context.findTransient('network-cost-weights-row-counts')
      }
    };
  },
  capture(final) {
    if (final('converged').values[0] === 0)
      throw new Error('Network cost matrix did not converge within maximumIterations = 24');
    if (final('weightOverflow').values[0] !== 0)
      throw new Error('Network cost weights overflowed the facility-by-node capacity');
    const positions = final('nodePositions');
    const offsets = final('weightOffsets');
    const facilityCount = facilities.length;
    const rowStart = offsets.values[0];
    const rowEnd = offsets.values[1];
    const rowNeighbors = sliceCapture(final('weightNeighbors'), rowStart, rowEnd, {
      id: 'rowNeighbors',
      meaning: `weights row 0 (${facilities[0].name}): retained node ids (entries ${rowStart} to ${rowEnd}; the other rows are not recorded)`
    });
    return [
      {
        ...final('facilityPositions'),
        values: Float32Array.from(Array.from(final('facilityPositions').values).slice(0, 2)),
        keySpace: undefined,
        role: 'input',
        meaning: `input: facility row 0 (${facilities[0].name}), longitude/latitude as uploaded`,
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      },
      sliceCapture(offsets, 0, facilityCount, {
        id: 'weightRowStart',
        keySpace: 'facility',
        meaning: 'first retained entry of each facility row (GPU row offsets)'
      }),
      sliceCapture(final('network-cost-weights-row-counts'), 0, facilityCount, {
        id: 'weightRowCounts',
        keySpace: 'facility',
        meaning: 'retained entries per facility row (the GPU count pass before the scan)'
      }),
      final('weightOverflow'),
      rowNeighbors,
      sliceCapture(final('weightValues'), rowStart, rowEnd, {
        id: 'rowWeights',
        meaning: 'weights row 0: exp(-beta * seconds) of each retained node'
      }),
      sliceCapture(final('weightDistances'), rowStart, rowEnd, {
        id: 'rowSeconds',
        meaning: 'weights row 0: travel seconds of each retained node'
      }),
      {
        ...positions,
        id: 'retained-nodes',
        name: 'retained-nodes',
        keySpace: undefined,
        values: gatherRows(positions, rowNeighbors.values),
        role: 'output',
        meaning: `output: intersections within ${COST_LIMIT_SECONDS} s of facility row 0, coloured by decay weight (positions gathered by the GPU neighbor ids)`,
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'rowWeights'}
      }
    ];
  }
};

/** Executes snapping, a cost matrix and the distance-decay weights, and reads one row back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
