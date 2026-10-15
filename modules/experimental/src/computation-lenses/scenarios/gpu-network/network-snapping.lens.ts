// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-snapping.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {getFacilityPositions, getSearchKeySpaces} from './search-lens-support';

const scenarioId = 'gpu-network.network-snapping';
const roads = loadChicagoLoopRoads();
/** Every facility of the file is snapped; snapping needs no nominal palette. */
const FACILITY_ROWS = roads.facilities.map((_, row) => row);

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['snappedEdges', 'snapDistances', 'snappedPositions'],
  formulas: ['gpu-network.snapping.nearest'],
  keySpaces: getSearchKeySpaces(roads, FACILITY_ROWS.length),
  entityKeySpace: 'point',
  choices: [
    {
      stage: 'assignment',
      name: 'snapMetric',
      value: 'great-circle',
      alternatives: ['planar'],
      sweepable: false
    }
  ],
  async build(context) {
    const facilityPositions = context.upload(
      'facilityPositions',
      getFacilityPositions(context.data, FACILITY_ROWS),
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
        snappedEdges: example.snappedEdges,
        snapDistances: example.snapDistances,
        snappedPositions: example.snappedPositions
      }
    };
  },
  capture(final) {
    const distances = final('snapDistances');
    // Without a maximum snap distance every point snaps; -1 would mean no edge was found.
    if (Array.from(distances.values).some(distance => !(distance >= 0)))
      throw new Error('Network snapping: a facility snapped to no edge');
    const values = Array.from(distances.values);
    const farthest = values.indexOf(Math.max(...values));
    const snapped: GPUComputationCapture = final('snappedPositions');
    return [
      {
        ...final('facilityPositions'),
        keySpace: 'point',
        role: 'input',
        meaning: 'input: facility longitude/latitude as uploaded',
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      },
      {...final('snappedEdges'), keySpace: 'point'},
      {...distances, keySpace: 'point', meaning: 'snap distance in meters (great-circle)'},
      {
        ...snapped,
        keySpace: 'point',
        role: 'output',
        highlight: {key: farthest, label: 'farthest snap'},
        meaning:
          'output: GPU snapped positions on the nearest road edge, coloured by snap distance',
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'snapDistances'}
      }
    ];
  }
};

/** Executes the snapping on the Chicago Loop roads and reads it back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
