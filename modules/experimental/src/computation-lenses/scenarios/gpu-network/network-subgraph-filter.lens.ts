// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getGPUNetworkSubgraphFilterParameterValues} from '../../../gpu-network/network-subgraph-filter/index';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-subgraph-filter.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  createGatheredCapture,
  getSlotSources,
  getStructureKeySpaces,
  roundCoordinate
} from './structure-lens-support';

const scenarioId = 'gpu-network.network-subgraph-filter';
const roads = loadChicagoLoopRoads();
/** Edges are kept when their travel time is in [0, MAXIMUM_SECONDS): the short blocks. */
const MAXIMUM_SECONDS = 6;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['nodeMask', 'edgeMask', 'liveCounts'],
  formulas: ['gpu-network.subgraph-filter.range'],
  keySpaces: getStructureKeySpaces(roads),
  entityKeySpace: 'edge',
  choices: [
    {
      stage: 'representation',
      name: 'edgeRange',
      value: `travel time in [0, ${MAXIMUM_SECONDS}) s`,
      alternatives: ['[0, 10) s', '[0, 30) s'],
      sweepable: true,
      unit: 's'
    },
    {
      stage: 'representation',
      name: 'dropIsolated',
      value: true,
      alternatives: [false],
      sweepable: false
    }
  ],
  async build(context) {
    const parameters = context.upload(
      'parameters',
      getGPUNetworkSubgraphFilterParameterValues(
        {nodeColumnCount: 0, edgeColumnCount: 1, hasEdgeTimes: false},
        {edgeRanges: [[0, MAXIMUM_SECONDS]]}
      ),
      'float32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      parameters
    });
    return {
      views: {
        nodeMask: example.nodeMask,
        edgeMask: example.edgeMask,
        liveCounts: example.counts,
        weights: context.topology.weights
      }
    };
  },
  capture(final) {
    const positions = final('nodePositions');
    const offsets = final('offsets');
    const neighbors = final('neighbors');
    const edgeMask = final('edgeMask');
    const counts = final('liveCounts');
    const sources = getSlotSources(offsets.values);
    // Kept slots become two-vertex segments between their endpoint positions.
    const coordinates: number[] = [];
    let kept = 0;
    for (let slot = 0; slot < edgeMask.values.length; slot++) {
      if (edgeMask.values[slot] === 0) continue;
      const source = sources[slot];
      const target = neighbors.values[slot];
      coordinates.push(
        roundCoordinate(positions.values[source * 2]),
        roundCoordinate(positions.values[source * 2 + 1]),
        roundCoordinate(positions.values[target * 2]),
        roundCoordinate(positions.values[target * 2 + 1])
      );
      kept++;
    }
    // liveSlotCount (word 1) is counted by the GPU from the same mask.
    if (kept !== counts.values[1])
      throw new Error('Network subgraph filter: edgeMask and counts disagree');
    return [
      {
        ...final('weights'),
        keySpace: 'edge',
        meaning: 'travel time in seconds of every directed edge slot (uploaded weights read back)'
      },
      {
        ...edgeMask,
        keySpace: 'edge',
        meaning: 'GPU edge mask: 1 when the slot is kept (in range, both endpoints live)'
      },
      {
        ...final('nodeMask'),
        keySpace: 'node',
        meaning: 'GPU node mask: 1 when the intersection keeps at least one live slot'
      },
      {
        ...counts,
        meaning:
          'GPU counts: liveNodeCount, liveSlotCount, selfLoopSlotCount, liveEdgeCount (directed edge slots)'
      },
      createGatheredCapture({
        id: 'keptEdgeOffsets',
        elementType: 'u32',
        values: Array.from({length: kept + 1}, (_, edge) => 2 * edge),
        meaning: 'index bookkeeping: two vertices per kept edge'
      }),
      createGatheredCapture({
        id: 'kept-edges',
        elementType: 'f32',
        components: 2,
        values: coordinates,
        role: 'output',
        meaning:
          'output: edges kept by the GPU edgeMask (endpoint positions gathered from the offsets, neighbors and nodePositions readbacks)',
        geometry: {kind: 'lines', coordinateModel: 'geographic', offsetsCapture: 'keptEdgeOffsets'}
      })
    ];
  }
};

/** Executes the travel-time edge filter on the Chicago Loop roads and reads the masks back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
