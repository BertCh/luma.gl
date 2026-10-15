// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './drive-time-catchment-recipe.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {getFacilityPositions, getSearchKeySpaces, maskNoneSentinel} from './search-lens-support';

const scenarioId = 'gpu-network.drive-time-catchment-recipe';
const roads = loadChicagoLoopRoads();
/** Every 12th intersection stands in for demand: a demonstrative sample, not resident counts. */
const DEMAND_STRIDE = 12;
const demandNodes = Array.from(
  {length: Math.ceil(roads.nodeCount / DEMAND_STRIDE)},
  (_, index) => index * DEMAND_STRIDE
);
/** Ascending drive-time upper bounds in seconds. */
const COST_BREAKS = [90, 180, 270, 360];
const prefix = 'drive-time-catchment';

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: [
    `${prefix}-demand-times`,
    `${prefix}-demand-bands`,
    `${prefix}-band-keys`,
    `${prefix}-band-counts`,
    `${prefix}-service-area-converged`
  ],
  formulas: ['gpu-network.catchment.band'],
  keySpaces: getSearchKeySpaces(roads, demandNodes.length),
  entityKeySpace: 'point',
  queuesResource: `${prefix}-service-areas-reachability-queues`,
  choices: [
    {
      stage: 'assignment',
      name: 'costBreaks',
      value: COST_BREAKS.join(','),
      alternatives: ['60,120,180,240'],
      sweepable: true
    }
  ],
  async build(context) {
    const facilityPositions = context.upload(
      'facilityPositions',
      getFacilityPositions(context.data),
      'float32x2'
    );
    const demandPositions = context.upload(
      'demandPositions',
      Float32Array.from(
        demandNodes.flatMap(node => [
          context.data.nodes[node * 2],
          context.data.nodes[node * 2 + 1]
        ])
      ),
      'float32x2'
    );
    const costBreaks = context.upload('costBreaks', Float32Array.from(COST_BREAKS), 'float32');
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      facilityPositions,
      demandPositions,
      costBreaks
    });
    return {
      views: {
        facilityPositions,
        demandPositions,
        demandTimes: example.demandTimes,
        demandBands: example.demandBands,
        converged: example.catchment.status.serviceAreaConverged
      }
    };
  },
  capture(final) {
    if (final('converged').values[0] === 0)
      throw new Error('Drive-time catchment: the service-area search did not converge');
    const demand = final('demandPositions');
    return [
      {
        ...final('facilityPositions'),
        keySpace: undefined,
        role: 'input',
        meaning: 'input: facility longitude/latitude as uploaded',
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      },
      {...demand, keySpace: 'point', meaning: 'demand points: every 12th road intersection'},
      {...final('demandTimes'), keySpace: 'point'},
      {
        ...maskNoneSentinel(
          final('demandBands'),
          'GPU demandBands readback; the no-band sentinel (0xffffffff) is shown as no data'
        ),
        keySpace: 'point'
      },
      {
        ...demand,
        id: 'catchment-demand',
        name: 'catchment-demand',
        keySpace: 'point',
        role: 'output',
        meaning: 'output: demand points coloured by drive time to the nearest facility',
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'demandTimes'}
      }
    ];
  }
};

/** Executes the drive-time catchment recipe on the Chicago Loop roads and reads it back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
