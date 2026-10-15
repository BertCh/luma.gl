// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './flow-aggregation.example';
import {
  CHICAGO_OD_FLOWS,
  CONSTRUCTION_DATASETS,
  gatherRows,
  getKeySpaces,
  sliceCapture
} from './construction-lens-support';

const scenarioId = 'gpu-network.flow-aggregation';
const records = CHICAGO_OD_FLOWS.hourly;
const TOP_FLOW_COUNT = 32;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'lengthMeters',
  withoutRoads: true,
  seed: 0,
  datasetVersion: `${CONSTRUCTION_DATASETS.odFlows.id}-v1`,
  retain: [
    'flowKeys',
    'flowOriginZones',
    'flowDestinationZones',
    'flowCounts',
    'flowWeights',
    'flowCount',
    'flowOverflow',
    'flowRequiredCount',
    'flowPairOverflow'
  ],
  formulas: ['gpu-network.flow-aggregation.pair-sum'],
  keySpaces: getKeySpaces([
    {id: 'record', label: 'taxi OD record', size: records.trips.length},
    {id: 'flow', label: 'ranked zone-pair flow', size: TOP_FLOW_COUNT},
    {id: 'zone', label: 'community-area centroid', size: CHICAGO_OD_FLOWS.zoneCount}
  ]),
  entityKeySpace: 'flow',
  choices: [
    {
      stage: 'summary',
      name: 'sumOrder',
      value: 'sorted',
      alternatives: ['atomic'],
      sweepable: false
    },
    {
      stage: 'summary',
      name: 'records',
      value: 'weekday 07:00-09:59 rows of the 150 busiest zone pairs',
      alternatives: ['annual rows'],
      sweepable: false
    }
  ],
  async build(context) {
    const zonePositions = context.upload(
      'zonePositions',
      Float32Array.from(CHICAGO_OD_FLOWS.zonePositions),
      'float32x2'
    );
    const originZoneIds = context.upload(
      'originZoneIds',
      Uint32Array.from(records.originZones),
      'uint32'
    );
    const destinationZoneIds = context.upload(
      'destinationZoneIds',
      Uint32Array.from(records.destinationZones),
      'uint32'
    );
    const tripCounts = context.upload('tripCounts', Float32Array.from(records.trips), 'float32');
    const example = await runExample(context.device, {
      graph: context.graph,
      originZoneIds,
      destinationZoneIds,
      tripCounts,
      zoneCount: CHICAGO_OD_FLOWS.zoneCount
    });
    return {
      views: {
        zonePositions,
        originZoneIds,
        destinationZoneIds,
        flowOriginZones: example.flowOriginZoneIds,
        flowDestinationZones: example.flowDestinationZoneIds,
        flowCounts: example.flowCounts,
        flowWeights: example.flowWeights,
        flowCount: example.status.count,
        flowOverflow: example.status.overflow,
        flowRequiredCount: example.status.requiredCount!,
        flowPairOverflow: example.status.pairOverflow!
      }
    };
  },
  capture(final) {
    if (final('flowPairOverflow').values[0] !== 0)
      throw new Error('Flow aggregation: more distinct zone pairs than pairCapacity = 512');
    const zones = final('zonePositions');
    const origins = final('originZoneIds');
    const destinations = final('destinationZoneIds');
    const recordCount = records.trips.length;
    const flowCount = final('flowCount').values[0];
    const flowOrigins = sliceCapture(final('flowOriginZones'), 0, flowCount, {
      id: 'flowOriginZones',
      keySpace: 'flow',
      meaning: 'origin zone of each ranked flow (GPU output)'
    });
    const flowDestinations = sliceCapture(final('flowDestinationZones'), 0, flowCount, {
      id: 'flowDestinationZones',
      keySpace: 'flow',
      meaning: 'destination zone of each ranked flow (GPU output)'
    });
    const flowWeights = sliceCapture(final('flowWeights'), 0, flowCount, {
      id: 'flowWeights',
      keySpace: 'flow',
      meaning: 'summed trips of each ranked zone pair, largest first (GPU output)'
    });
    return [
      {
        ...zones,
        keySpace: undefined,
        role: 'context',
        meaning: 'context: community-area centroids as uploaded (longitude/latitude)',
        geometry: {kind: 'points', coordinateModel: 'geographic'}
      },
      {
        ...zones,
        id: 'recordLines',
        name: 'recordLines',
        keySpace: undefined,
        values: Float32Array.from(
          Array.from({length: recordCount}, (_, record) => [
            ...gatherRows(zones, [origins.values[record]]),
            ...gatherRows(zones, [destinations.values[record]])
          ]).flat()
        ),
        role: 'input',
        meaning:
          'input: one straight line per taxi OD record (weekday 07:00-09:59 rows, several records per zone pair), positions gathered from the zonePositions readback',
        geometry: {kind: 'lines', coordinateModel: 'geographic', offsetsCapture: 'recordOffsets'}
      },
      {
        ...final('flowCount'),
        id: 'recordOffsets',
        name: 'recordOffsets',
        keySpace: undefined,
        values: Uint32Array.from({length: recordCount + 1}, (_, record) => record * 2),
        meaning: 'derived: record k spans vertices 2k to 2k + 2 of the input lines'
      },
      final('flowCount'),
      final('flowRequiredCount'),
      final('flowOverflow'),
      final('flowPairOverflow'),
      flowOrigins,
      flowDestinations,
      flowWeights,
      sliceCapture(final('flowCounts'), 0, flowCount, {
        id: 'flowCounts',
        keySpace: 'flow',
        meaning: 'records in each ranked zone pair (GPU output)'
      }),
      {
        ...zones,
        id: 'rankedFlowLines',
        name: 'rankedFlowLines',
        keySpace: undefined,
        values: Float32Array.from(
          Array.from({length: flowCount}, (_, flow) => [
            ...gatherRows(zones, [flowOrigins.values[flow]]),
            ...gatherRows(zones, [flowDestinations.values[flow]])
          ]).flat()
        ),
        role: 'output',
        highlight: {key: 0, label: 'busiest pair'},
        meaning:
          'output: the top ranked zone-pair flows, coloured by summed trips (positions gathered from the zonePositions readback by the GPU zone ids)',
        geometry: {
          kind: 'lines',
          coordinateModel: 'geographic',
          offsetsCapture: 'rankedFlowOffsets',
          valuesCapture: 'flowWeights'
        }
      },
      {
        ...final('flowCount'),
        id: 'rankedFlowOffsets',
        name: 'rankedFlowOffsets',
        keySpace: undefined,
        values: Uint32Array.from({length: flowCount + 1}, (_, flow) => flow * 2),
        meaning: 'derived: ranked flow k spans vertices 2k to 2k + 2 of the output lines'
      }
    ];
  }
};

/** Executes the zone-pair flow aggregation on taxi records and reads the ranked flows back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
