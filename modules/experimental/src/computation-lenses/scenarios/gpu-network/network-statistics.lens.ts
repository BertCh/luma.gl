// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-statistics.example';
import {loadChicagoLoopRoads} from './network-lens-support';
import {createGatheredCapture, getStructureKeySpaces} from './structure-lens-support';

const scenarioId = 'gpu-network.network-statistics';
const roads = loadChicagoLoopRoads();
/** Degree histogram bins; the last bin collects every larger degree. */
const BIN_COUNT = 8;
/** Header words shown: liveNodeCount .. selfLoopSlotCount (words 14 and 15 are reserved zeros). */
const HEADER_WORDS = 14;
/** Words before the three degree histograms. */
const HEADER_LENGTH = 16;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'travelTimeSeconds',
  seed: 0,
  retain: ['summary'],
  formulas: ['gpu-network.statistics.degree-histogram'],
  // The nested gpu-graph component kernels name their rows `vertex`.
  keySpaces: getStructureKeySpaces(roads, [
    {id: 'vertex', label: 'graph vertex', indexSpace: '1d', size: roads.nodeCount},
    {id: 'degree-bin', label: 'out-degree bin', indexSpace: '1d', size: BIN_COUNT},
    {id: 'statistic', label: 'summary header word', indexSpace: '1d', size: HEADER_WORDS}
  ]),
  entityKeySpace: 'degree-bin',
  // The summary is a table, not a map drawing.
  roadContext: false,
  choices: [
    {
      stage: 'summary',
      name: 'degreeBinCount',
      value: BIN_COUNT,
      alternatives: [4, 16],
      sweepable: true
    },
    {
      stage: 'summary',
      name: 'degreeBinning',
      value: 'linear',
      alternatives: ['log2'],
      sweepable: false
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
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology
    });
    return {views: {summary: example.summary}};
  },
  capture(final) {
    const summary = final('summary');
    const words = Array.from(summary.values);
    // Word 5 is the GPU's own report that component labelling reached a fixed point.
    if (words[5] === 0)
      throw new Error('Network statistics: components did not converge within 32 iterations');
    return [
      createGatheredCapture({
        id: 'networkStatistics',
        source: summary,
        elementType: 'u32',
        keySpace: 'statistic',
        values: words.slice(0, HEADER_WORDS),
        role: 'output',
        meaning:
          'output: GPU summary words 0-13 as written: 0 liveNodeCount, 1 liveEdgeCount, 2 liveSlotCount, 3 componentCount, 4 largestComponentSize, 5 componentsConverged, 6 isolatedNodeCount, 7 maximumOutDegree, 8 maximumInDegree, 9 maximumTotalDegree, 10 modularity (float32 bits, 0 without communities), 11 modularityValid, 12 intraCommunitySlotCount, 13 selfLoopSlotCount'
      }),
      createGatheredCapture({
        id: 'outDegreeHistogram',
        source: summary,
        elementType: 'u32',
        keySpace: 'degree-bin',
        values: words.slice(HEADER_LENGTH, HEADER_LENGTH + BIN_COUNT),
        role: 'output',
        meaning:
          'output: GPU summary words 16-23, live nodes per out-degree bin; bin b holds out-degree b (linear width 1) and the last bin holds 7 and above'
      })
    ];
  }
};

/** Executes the network statistics on the Chicago Loop roads and reads the summary back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
