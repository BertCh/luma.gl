// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-cost-table.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {gatherRows, sliceCapture} from './construction-lens-support';
import {gatherNodePoints, getSearchKeySpaces} from './search-lens-support';

const scenarioId = 'gpu-network.network-cost-table';
const roads = loadChicagoLoopRoads();
/** Cost limit in seconds: every node within 30 s of a source is stored in its table row. */
const COST_LIMIT_SECONDS = 30;
/** Entry capacity: 32 entries per row on average; the run fails loudly if the table overflows. */
const TABLE_CAPACITY = roads.nodeCount * 32;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: roads.startNode,
  retain: ['tableOffsets', 'tableTargets', 'tableCosts', 'overflow', 'requiredCount', 'converged'],
  formulas: ['gpu-network.cost-table.admission'],
  keySpaces: getSearchKeySpaces(roads, 0),
  entityKeySpace: 'node',
  queuesResource: 'network-cost-table-batch-0-queues',
  choices: [
    {
      stage: 'search',
      name: 'costLimitSeconds',
      value: COST_LIMIT_SECONDS,
      alternatives: [15, 60],
      sweepable: true
    }
  ],
  async build(context) {
    const costLimit = context.upload('costLimit', Float32Array.of(COST_LIMIT_SECONDS), 'float32');
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      costLimit,
      tableCapacity: TABLE_CAPACITY
    });
    return {
      views: {
        tableOffsets: example.offsets,
        tableTargets: example.targets,
        tableCosts: example.costs,
        overflow: example.status.overflow!,
        requiredCount: example.status.requiredCount!,
        converged: example.status.converged!
      }
    };
  },
  capture(final) {
    if (final('converged').values[0] === 0)
      throw new Error('Network cost table did not converge within maximumIterations = 12');
    if (final('overflow').values[0] !== 0)
      throw new Error(`Network cost table overflowed its ${TABLE_CAPACITY}-entry capacity`);
    const positions = final('nodePositions');
    const offsets = final('tableOffsets');
    const nodeCount = positions.values.length / 2;
    const rowStart = offsets.values[roads.startNode];
    const rowEnd = offsets.values[roads.startNode + 1];
    const rowTargets = sliceCapture(final('tableTargets'), rowStart, rowEnd, {
      id: 'rowTargets',
      meaning: `table row of the start intersection: target node ids (entries ${rowStart} to ${rowEnd} of the GPU table; the other rows are not recorded)`
    });
    const rowCosts = sliceCapture(final('tableCosts'), rowStart, rowEnd, {
      id: 'rowCosts',
      meaning: 'table row of the start intersection: travel seconds from the start to each target'
    });
    return [
      gatherNodePoints(positions, [roads.startNode], {
        id: 'start-intersection',
        meaning:
          'input: start intersection (the table row shown; position gathered from the nodePositions readback)'
      }),
      sliceCapture(offsets, 0, nodeCount, {
        id: 'tableRowStart',
        keySpace: 'node',
        meaning: 'first table entry of each row (the GPU row offsets without the final total)'
      }),
      sliceCapture(offsets, 1, nodeCount + 1, {
        id: 'tableRowEnd',
        keySpace: 'node',
        meaning: 'one past the last table entry of each row (the GPU row offsets without the first)'
      }),
      final('requiredCount'),
      final('overflow'),
      final('converged'),
      rowTargets,
      rowCosts,
      {
        ...positions,
        id: 'table-row-targets',
        name: 'table-row-targets',
        keySpace: undefined,
        values: gatherRows(positions, rowTargets.values),
        role: 'output',
        meaning:
          'output: the intersections stored in the start row, coloured by travel seconds (positions gathered by the GPU target ids)',
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'rowCosts'}
      }
    ];
  }
};

/** Executes the bounded cost table on the Chicago Loop roads and reads one row back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
