// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-centrality.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {gatherNodePoints, reverseRoads} from './search-lens-support';
import {getMaximumKey, getStructureKeySpaces} from './structure-lens-support';

const scenarioId = 'gpu-network.network-centrality';
const roads = loadChicagoLoopRoads();
/** Number of sampled pivot intersections, searched together as one batch of lanes. */
const PIVOT_COUNT = 8;
/** Pivots are the evenly spaced node ids floor(i * N / 8): a fixed sample, not a random one. */
const pivots = Array.from({length: PIVOT_COUNT}, (_, row) =>
  Math.floor((row * roads.nodeCount) / PIVOT_COUNT)
);

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: PIVOT_COUNT,
  retain: ['closeness', 'reachedCounts', 'converged'],
  formulas: ['gpu-network.centrality.closeness'],
  keySpaces: getStructureKeySpaces(roads, [
    {id: 'pivot', label: 'pivot intersection', indexSpace: '1d', size: PIVOT_COUNT},
    {
      id: 'frontier-slot',
      label: 'frontier slot',
      indexSpace: '1d',
      size: 2 * PIVOT_COUNT * roads.nodeCount
    }
  ]),
  entityKeySpace: 'pivot',
  queuesResource: 'network-centrality-batch-0-reachability-queues',
  choices: [
    {
      stage: 'search',
      name: 'pivots',
      value: PIVOT_COUNT,
      alternatives: [1, roads.nodeCount],
      sweepable: false,
      label: 'sampled sources: node ids floor(i * N / 8), a fixed sample (seed = pivot count)'
    },
    {
      stage: 'summary',
      name: 'centrality',
      value: 'outward closeness',
      alternatives: ['harmonic closeness', 'betweenness'],
      sweepable: false
    }
  ],
  async build(context) {
    // Closeness still needs the exact transpose: the same edges reversed, weights carried over.
    const reverse = reverseRoads(context.data);
    const reverseTopology = {
      offsets: context.upload('reverseOffsets', Uint32Array.from(reverse.offsets), 'uint32'),
      neighbors: context.upload('reverseNeighbors', Uint32Array.from(reverse.neighbors), 'uint32'),
      weights: context.upload(
        'reverseWeights',
        Float32Array.from(reverse.travelTimeSeconds),
        'float32'
      )
    };
    const pivotNodes = context.upload('pivotNodes', Uint32Array.from(pivots), 'uint32');
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      reverseRoads: reverseTopology,
      pivotNodes
    });
    return {
      views: {
        closeness: example.closeness,
        reachedCounts: example.reachedCounts,
        converged: example.converged,
        pivotNodes
      }
    };
  },
  capture(final) {
    // The pivot searches must have reached a fixpoint within the unrolled rounds.
    if (final('converged').values[0] === 0)
      throw new Error('Network centrality did not converge within maximumIterations = 32');
    const positions = final('nodePositions');
    const closeness = {
      ...final('closeness'),
      keySpace: 'pivot',
      meaning: 'GPU outward closeness per pivot, (reached - 1) / total cost, in 1/s'
    };
    return [
      {
        ...final('pivotNodes'),
        keySpace: 'pivot',
        meaning: 'input: network node id of each pivot (a fixed even sample)'
      },
      gatherNodePoints(positions, pivots, {
        id: 'pivot-nodes',
        meaning: 'input: pivot intersections (positions gathered from the nodePositions readback)'
      }),
      closeness,
      {
        ...final('reachedCounts'),
        keySpace: 'pivot',
        meaning: 'GPU count of nodes reached from each pivot, the pivot included'
      },
      {
        ...gatherNodePoints(positions, pivots, {
          id: 'closeness-pivots',
          meaning:
            'output: pivot intersections coloured by outward closeness; the highest is marked (positions gathered from the nodePositions readback)'
        }),
        role: 'output',
        highlight: {key: getMaximumKey(closeness.values), label: 'highest closeness'},
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'closeness'}
      }
    ];
  }
};

/** Executes pivot closeness on the Chicago Loop roads and reads it back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
