// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-coarsening.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {createNodeValuePoints} from './search-lens-support';
import {
  createGatheredCapture,
  getGridZoneLabels,
  getStructureKeySpaces,
  roundCoordinate,
  takeFirstRows
} from './structure-lens-support';

const scenarioId = 'gpu-network.network-coarsening';
const roads = loadChicagoLoopRoads();
/** The input partition is a 2 x 2 grid over the data bounding box. */
const ZONE_COLUMNS = 2;
const ZONE_ROWS = 2;
const GROUP_CAPACITY = ZONE_COLUMNS * ZONE_ROWS;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: GROUP_CAPACITY,
  retain: [
    'groupNodeCount',
    'groupCentroid',
    'sourceGroups',
    'targetGroups',
    'edgeCounts',
    'superedgeCount',
    'overflow'
  ],
  formulas: [],
  keySpaces: getStructureKeySpaces(roads, [
    {id: 'group', label: 'zone (supernode)', indexSpace: '1d', size: GROUP_CAPACITY},
    {
      id: 'superedge',
      label: 'superedge row',
      indexSpace: '1d',
      size: GROUP_CAPACITY * GROUP_CAPACITY
    }
  ]),
  entityKeySpace: 'group',
  choices: [
    {
      stage: 'assignment',
      name: 'zones',
      value: `${ZONE_COLUMNS} x ${ZONE_ROWS} grid over the data bounding box`,
      alternatives: ['service areas of four facilities'],
      sweepable: false,
      label: 'input partition computed in the script (data preparation), not a GPU result'
    },
    {
      stage: 'representation',
      name: 'directed',
      value: true,
      alternatives: [false],
      sweepable: false
    }
  ],
  async build(context) {
    const zones = context.upload(
      'zones',
      getGridZoneLabels(context.data, ZONE_COLUMNS, ZONE_ROWS),
      'uint32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      zones,
      groupCapacity: GROUP_CAPACITY
    });
    return {
      views: {
        zones,
        groupNodeCount: example.groupNodeCount,
        groupCentroid: example.groupCentroid,
        sourceGroups: example.sourceGroups,
        targetGroups: example.targetGroups,
        edgeCounts: example.edgeCounts,
        superedgeCount: example.superedgeCount,
        overflow: example.overflow
      }
    };
  },
  capture(final) {
    // Overflow means a label reached groupCapacity or the superedge list was cut.
    if (final('overflow').values[0] !== 0)
      throw new Error('Network coarsening overflowed its group or superedge capacity');
    const positions = final('nodePositions');
    const count = final('superedgeCount').values[0];
    const groupNodeCount = {
      ...final('groupNodeCount'),
      keySpace: 'group',
      meaning: 'GPU intersections per zone'
    };
    if (Array.from(groupNodeCount.values).some(nodes => nodes === 0))
      throw new Error('Network coarsening: a zone is empty');
    const centroids = final('groupCentroid');
    const sources = takeFirstRows(final('sourceGroups'), count, {
      keySpace: 'superedge',
      meaning: 'GPU source zone of each superedge (first superedgeCount rows)'
    });
    const targets = takeFirstRows(final('targetGroups'), count, {
      keySpace: 'superedge',
      meaning: 'GPU target zone of each superedge (first superedgeCount rows)'
    });
    const edgeCounts = takeFirstRows(final('edgeCounts'), count, {
      keySpace: 'superedge',
      meaning: 'GPU directed edge count of each superedge'
    });
    const centroidAt = (group: number) => [
      roundCoordinate(centroids.values[group * 2]),
      roundCoordinate(centroids.values[group * 2 + 1])
    ];
    return [
      {
        ...final('zones'),
        keySpace: 'node',
        meaning:
          'input: zone of every intersection (2 x 2 grid prepared in the script and uploaded)'
      },
      {
        ...createNodeValuePoints(positions, {
          id: 'zone-nodes',
          valuesCapture: 'zones',
          meaning: 'input: intersections coloured by their input zone'
        }),
        role: 'input'
      },
      groupNodeCount,
      {
        ...centroids,
        keySpace: 'group',
        meaning:
          'GPU zone centroid, the arithmetic mean of member longitude and latitude (a drawing anchor)'
      },
      sources,
      targets,
      edgeCounts,
      {
        ...centroids,
        id: 'supernodes',
        name: 'supernodes',
        keySpace: undefined,
        role: 'output',
        meaning:
          'output: one supernode per zone at its GPU centroid, coloured by GPU intersection count',
        geometry: {
          kind: 'points',
          coordinateModel: 'geographic',
          valuesCapture: 'groupNodeCount'
        }
      },
      createGatheredCapture({
        id: 'superedgeOffsets',
        elementType: 'u32',
        values: Array.from({length: count + 1}, (_, row) => 2 * row),
        meaning: 'index bookkeeping: two vertices per superedge'
      }),
      createGatheredCapture({
        id: 'superedges',
        elementType: 'f32',
        components: 2,
        values: Array.from({length: count}, (_, row) => [
          ...centroidAt(sources.values[row]),
          ...centroidAt(targets.values[row])
        ]).flat(),
        role: 'output',
        meaning:
          'output: superedges drawn between the GPU zone centroids (endpoints gathered from the GPU source and target zone rows), coloured by edge count',
        geometry: {
          kind: 'lines',
          coordinateModel: 'geographic',
          offsetsCapture: 'superedgeOffsets',
          valuesCapture: 'edgeCounts'
        }
      })
    ];
  }
};

/** Executes the zone coarsening on the Chicago Loop roads and reads the summary graph back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
