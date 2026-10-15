// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-path-extraction.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {gatherNodePoints, getSearchKeySpaces, SEARCH_FACILITY_ROWS} from './search-lens-support';

const scenarioId = 'gpu-network.network-path-extraction';
const roads = loadChicagoLoopRoads();
// From the file's centre intersection to the West Loop library's prepared nearest intersection.
const startNode = roads.startNode;
const targetNode = roads.facilities[SEARCH_FACILITY_ROWS[1]].nearestNode;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: startNode,
  retain: [
    'costs',
    'predecessors',
    'converged',
    'ids',
    'lineOffsets',
    'pathFound',
    'count',
    'overflow'
  ],
  formulas: [],
  keySpaces: getSearchKeySpaces(roads, 0),
  entityKeySpace: 'node',
  queuesResource: 'network-reachability-queues',
  choices: [
    {
      stage: 'search',
      name: 'targets',
      value: 1,
      alternatives: [4],
      sweepable: false
    }
  ],
  async build(context) {
    const startNodes = context.upload('startNodes', Uint32Array.of(startNode), 'uint32');
    const targets = context.upload('targets', Uint32Array.of(targetNode), 'uint32');
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      startNodes,
      targets
    });
    return {
      views: {
        costs: example.costs,
        predecessors: example.predecessors,
        converged: example.converged,
        ids: example.ids,
        lineOffsets: example.lineOffsets,
        pathFound: example.pathFound,
        count: example.paths.props.output.status.count,
        overflow: example.paths.props.output.status.overflow
      }
    };
  },
  capture(final) {
    // The search must be complete, the walk untruncated and the target found, or the path is partial.
    if (final('converged').values[0] === 0)
      throw new Error('Network path extraction: the reachability search did not converge');
    if (final('overflow').values[0] !== 0)
      throw new Error('Network path extraction: path output overflowed or a walk was truncated');
    if (final('pathFound').values[0] !== 1)
      throw new Error('Network path extraction: the target was not found');
    const positions = final('nodePositions');
    const count = final('count').values[0];
    const ids = final('ids');
    const nodes = Array.from(ids.values).slice(0, count);
    return [
      gatherNodePoints(positions, [startNode, targetNode], {
        id: 'start-and-target',
        meaning:
          'input: start and target intersections (positions gathered from the nodePositions readback)'
      }),
      {...final('costs'), keySpace: 'node'},
      {...final('predecessors'), keySpace: 'node'},
      {
        ...ids,
        values: Uint32Array.from(nodes),
        meaning: 'first count entries of the GPU ids output'
      },
      final('lineOffsets'),
      final('pathFound'),
      {
        ...positions,
        id: 'extracted-path',
        name: 'extracted-path',
        keySpace: undefined,
        values: Float32Array.from(
          nodes.flatMap(node => [positions.values[node * 2], positions.values[node * 2 + 1]])
        ),
        role: 'output',
        highlight: {key: 0, label: 'path'},
        meaning:
          'output: extracted path (node ids from the GPU ids and lineOffsets outputs; positions gathered from the nodePositions readback)',
        geometry: {kind: 'lines', coordinateModel: 'geographic', offsetsCapture: 'lineOffsets'}
      }
    ];
  }
};

/** Executes reachability and path extraction on the Chicago Loop roads and reads the path back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
