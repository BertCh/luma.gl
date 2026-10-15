// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './adjacency-matrix-order.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  createGatheredCapture,
  getGridZoneLabels,
  getStructureKeySpaces,
  MATRIX_ZONE_COLUMNS,
  MATRIX_ZONE_ROWS
} from './structure-lens-support';

const scenarioId = 'gpu-network.adjacency-matrix-order';
const roads = loadChicagoLoopRoads();
/** Evenly spaced intersections shown in the table: node ids i * floor(N / 12). */
const SAMPLE_COUNT = 12;
const sampleNodes = Array.from(
  {length: SAMPLE_COUNT},
  (_, row) => row * Math.floor(roads.nodeCount / SAMPLE_COUNT)
);

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: MATRIX_ZONE_COLUMNS * MATRIX_ZONE_ROWS,
  retain: ['order'],
  formulas: [],
  keySpaces: getStructureKeySpaces(roads, [
    {id: 'sample', label: 'shown intersection', indexSpace: '1d', size: SAMPLE_COUNT}
  ]),
  entityKeySpace: 'node',
  // The result is a table of node-to-position rows, not a map drawing.
  roadContext: false,
  choices: [
    {
      stage: 'assignment',
      name: 'groups',
      value: `${MATRIX_ZONE_COLUMNS} x ${MATRIX_ZONE_ROWS} grid over the data bounding box`,
      alternatives: ['communities', 'service areas'],
      sweepable: false,
      label: 'input groups computed in the script (data preparation), not a GPU result'
    },
    {
      stage: 'presentation',
      name: 'shown nodes',
      value: SAMPLE_COUNT,
      alternatives: [roads.nodeCount],
      sweepable: false,
      label: 'the table shows evenly spaced node ids; the datum rung reads every node'
    }
  ],
  async build(context) {
    const groups = context.upload(
      'groups',
      getGridZoneLabels(context.data, MATRIX_ZONE_COLUMNS, MATRIX_ZONE_ROWS),
      'uint32'
    );
    const example = await runExample(context.device, {graph: context.graph, groups});
    return {views: {groups, order: example.order}};
  },
  capture(final) {
    const groups = final('groups');
    const order = final('order');
    // The permutation must be a bijection onto the node positions.
    if (new Set(Array.from(order.values)).size !== roads.nodeCount)
      throw new Error('Adjacency matrix order is not a permutation');
    return [
      {...groups, keySpace: 'node', meaning: 'input: group label of every intersection'},
      {
        ...order,
        keySpace: 'node',
        meaning: 'GPU matrix position of every intersection (order[node] = position)'
      },
      createGatheredCapture({
        id: 'sampleNodes',
        elementType: 'u32',
        keySpace: 'sample',
        values: sampleNodes,
        role: 'input',
        meaning: 'input: node id of each shown intersection (evenly spaced)'
      }),
      createGatheredCapture({
        id: 'groupsSample',
        source: groups,
        elementType: 'u32',
        keySpace: 'sample',
        values: sampleNodes.map(node => groups.values[node]),
        role: 'input',
        meaning: 'input: group label of each shown intersection (from the uploaded groups)'
      }),
      createGatheredCapture({
        id: 'orderSample',
        source: order,
        elementType: 'u32',
        keySpace: 'sample',
        values: sampleNodes.map(node => order.values[node]),
        role: 'output',
        meaning:
          'output: GPU matrix position of each shown intersection; positions sort by group, then node id'
      })
    ];
  }
};

/** Executes the group ordering on the Chicago Loop roads and reads the permutation back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
