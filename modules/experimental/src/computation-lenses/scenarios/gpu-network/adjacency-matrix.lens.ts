// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample as runOrderExample} from './adjacency-matrix-order.example';
import {runExample} from './adjacency-matrix.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {
  getGridZoneLabels,
  getMaximumKey,
  getStructureKeySpaces,
  MATRIX_ZONE_COLUMNS,
  MATRIX_ZONE_ROWS
} from './structure-lens-support';

const scenarioId = 'gpu-network.adjacency-matrix';
const roads = loadChicagoLoopRoads();
const RESOLUTION = 8;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: RESOLUTION,
  retain: ['order', 'matrixCounts', 'maximumCount'],
  formulas: ['gpu-network.adjacency-matrix.bin'],
  keySpaces: getStructureKeySpaces(roads, [
    {
      id: 'cell',
      label: 'matrix cell',
      indexSpace: '2d',
      size: RESOLUTION * RESOLUTION,
      shape: [RESOLUTION, RESOLUTION]
    }
  ]),
  entityKeySpace: 'cell',
  // The matrix is drawn in its own cell space, not on the map.
  roadContext: false,
  choices: [
    {
      stage: 'representation',
      name: 'resolution',
      value: RESOLUTION,
      alternatives: [4, 16, 64],
      sweepable: true
    },
    {
      stage: 'representation',
      name: 'order',
      value: `intersections sorted by ${MATRIX_ZONE_COLUMNS} x ${MATRIX_ZONE_ROWS} grid zone, then node id`,
      alternatives: ['node id order'],
      sweepable: false,
      label:
        'zones are an input partition computed in the script; the sort is GPUAdjacencyMatrixOrder'
    }
  ],
  async build(context) {
    const groups = context.upload(
      'groups',
      getGridZoneLabels(context.data, MATRIX_ZONE_COLUMNS, MATRIX_ZONE_ROWS),
      'uint32'
    );
    const ordering = await runOrderExample(context.device, {graph: context.graph, groups});
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      order: ordering.order
    });
    return {
      views: {
        order: ordering.order,
        matrixCounts: example.counts,
        maximumCount: example.maximumCount
      }
    };
  },
  capture(final) {
    const counts = final('matrixCounts');
    const total = Array.from(counts.values).reduce((sum, count) => sum + count, 0);
    // Every directed edge slot lands in exactly one cell of the full window.
    if (total !== roads.edgeCount)
      throw new Error('Adjacency matrix counts do not add up to the edge count');
    return [
      {
        ...final('order'),
        keySpace: 'node',
        meaning: 'GPU matrix position of every intersection (order[node] = position)'
      },
      {
        ...final('maximumCount'),
        meaning: 'GPU maximum of the cell counts'
      },
      {
        ...counts,
        keySpace: 'cell',
        role: 'output',
        highlight: {key: getMaximumKey(counts.values), label: 'most edges'},
        meaning: `output: ${RESOLUTION} x ${RESOLUTION} matrix of directed edge counts, row = source position bin, column = target position bin (about ${Math.round(roads.nodeCount / RESOLUTION)} positions per bin)`,
        geometry: {kind: 'raster', shape: [RESOLUTION, RESOLUTION]}
      }
    ];
  }
};

/** Executes the adjacency matrix on the Chicago Loop roads and reads the cell counts back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
