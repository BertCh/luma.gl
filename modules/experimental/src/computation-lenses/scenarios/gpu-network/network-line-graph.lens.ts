// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getGPUNetworkLineGraphParameterValues} from '../../../gpu-network/network-line-graph/index';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-line-graph.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  createGatheredCapture,
  getSlotSources,
  getStructureKeySpaces,
  roundCoordinate,
  takeFirstRows
} from './structure-lens-support';

const scenarioId = 'gpu-network.network-line-graph';
const roads = loadChicagoLoopRoads();
/** Cost per radian of absolute turn angle, in seconds. */
const ANGLE_COST = 2;
/** Extra cost of a left turn, in seconds. */
const LEFT_TURN_COST = 5;
/** Upper bound of the arcs: for every edge u -> v, the out-degree of v (U-turns are then banned). */
const arcCapacity = roads.neighbors.reduce(
  (sum, head) => sum + roads.offsets[head + 1] - roads.offsets[head],
  0
);

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['lineOffsets', 'lineNeighbors', 'lineWeights', 'arcCount', 'overflow'],
  formulas: ['gpu-network.line-graph.arc-weight'],
  keySpaces: getStructureKeySpaces(roads, [
    {id: 'arc', label: 'turn (line-graph arc)', indexSpace: '1d', size: arcCapacity}
  ]),
  entityKeySpace: 'arc',
  choices: [
    {
      stage: 'impedance',
      name: 'angleCost',
      value: ANGLE_COST,
      alternatives: [0, 5],
      sweepable: true,
      unit: 's/rad'
    },
    {
      stage: 'impedance',
      name: 'leftTurnCost',
      value: LEFT_TURN_COST,
      alternatives: [0, 10],
      sweepable: true,
      unit: 's'
    },
    {
      stage: 'impedance',
      name: 'uTurns',
      value: 'banned',
      alternatives: ['priced'],
      sweepable: false
    },
    {
      stage: 'representation',
      name: 'arcCapacity',
      value: arcCapacity,
      sweepable: false,
      label: 'upper bound: sum over edges u -> v of the out-degree of v'
    }
  ],
  async build(context) {
    const parameters = context.upload(
      'parameters',
      getGPUNetworkLineGraphParameterValues({
        angleCost: ANGLE_COST,
        leftTurnCost: LEFT_TURN_COST
      }),
      'float32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      parameters,
      arcCapacity
    });
    return {
      views: {
        lineOffsets: example.lineOffsets,
        lineNeighbors: example.lineNeighbors,
        lineWeights: example.lineWeights,
        arcCount: example.arcCount,
        overflow: example.overflow
      }
    };
  },
  capture(final) {
    // Arcs past the capacity are dropped; a shown line graph must not be cut.
    if (final('overflow').values[0] !== 0)
      throw new Error('Network line graph overflowed its arc capacity');
    const positions = final('nodePositions');
    const offsets = final('offsets');
    const neighbors = final('neighbors');
    const lineOffsets = final('lineOffsets');
    const count = final('arcCount').values[0];
    const sources = getSlotSources(offsets.values);
    // Each line node is a base edge, drawn at the midpoint of its two endpoint positions.
    const midpoints = sources.flatMap((source, edge) => {
      const target = neighbors.values[edge];
      return [
        roundCoordinate((positions.values[source * 2] + positions.values[target * 2]) / 2),
        roundCoordinate((positions.values[source * 2 + 1] + positions.values[target * 2 + 1]) / 2)
      ];
    });
    const legalTurns = sources.map(
      (_, edge) => lineOffsets.values[edge + 1] - lineOffsets.values[edge]
    );
    return [
      {
        ...lineOffsets,
        keySpace: undefined,
        meaning: 'GPU line-graph CSR row offsets (one row per edge)'
      },
      takeFirstRows(final('lineNeighbors'), count, {
        keySpace: 'arc',
        meaning: 'GPU next edge of each legal turn (first arcCount rows)'
      }),
      takeFirstRows(final('lineWeights'), count, {
        keySpace: 'arc',
        meaning: 'GPU turn cost in seconds: travel time of the next edge plus turn cost'
      }),
      createGatheredCapture({
        id: 'edgeMidpoints',
        elementType: 'f32',
        components: 2,
        keySpace: 'edge',
        values: midpoints,
        meaning:
          'derived layout: midpoint of each base edge (gathered from the offsets, neighbors and nodePositions readbacks); not a force layout'
      }),
      createGatheredCapture({
        id: 'legalTurns',
        source: lineOffsets,
        elementType: 'u32',
        keySpace: 'edge',
        values: legalTurns,
        role: 'output',
        meaning:
          'output: line graph; nodes are base edges at edge midpoints, arcs are legal turns, coloured by legal turns out of the edge (differences of the GPU lineOffsets)',
        geometry: {
          kind: 'graph',
          offsetsCapture: 'lineOffsets',
          targetsCapture: 'lineNeighbors',
          layoutCapture: 'edgeMidpoints',
          valuesCapture: 'legalTurns',
          coordinateModel: 'geographic'
        }
      })
    ];
  }
};

/** Executes the turn-cost line graph on the Chicago Loop roads and reads it back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
