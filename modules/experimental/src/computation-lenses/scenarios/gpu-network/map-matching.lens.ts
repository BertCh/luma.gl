// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues
} from '@luma.gl/experimental/gpu-network';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './map-matching.example';
import {
  CONSTRUCTION_DATASETS,
  LOOP_TRACES,
  createLoopFrame,
  projectLoopRoads
} from './construction-lens-support';
import {loadChicagoLoopRoads} from './network-lens-support';
import {getSearchKeySpaces, maskNoneSentinel} from './search-lens-support';

const scenarioId = 'gpu-network.map-matching';
const roads = loadChicagoLoopRoads();
/** Emission sigma, transition beta and candidate search radius, in metres of the local frame. */
const SIGMA_METERS = 20;
const BETA_METERS = 20;
const SEARCH_RADIUS_METERS = 75;

// Grid bounds in the local metre frame: the projected road and fix extent plus a margin.
function getBounds() {
  const frame = createLoopFrame();
  const planar = [...frame.projectArray(roads.nodes), ...frame.projectArray(LOOP_TRACES.positions)];
  const xs = planar.filter((_, index) => index % 2 === 0);
  const ys = planar.filter((_, index) => index % 2 === 1);
  return {
    minimum: [Math.min(...xs) - 100, Math.min(...ys) - 100] as [number, number],
    maximum: [Math.max(...xs) + 100, Math.max(...ys) + 100] as [number, number]
  };
}

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'lengthMeters',
  prepareRoads: projectLoopRoads,
  coordinateModel: 'projected',
  seed: 0,
  datasetVersion: `${CONSTRUCTION_DATASETS.traces.id}-v1`,
  retain: [
    'matchedEdges',
    'snappedPositions',
    'matchSnapDistances',
    'matchBreaks',
    'matchedCount',
    'matchOverflow'
  ],
  formulas: [],
  keySpaces: [
    ...getSearchKeySpaces(roads, 0),
    {id: 'fix', label: 'GPS fix', indexSpace: '1d', size: LOOP_TRACES.sampleCount},
    {id: 'track', label: 'GPS track', indexSpace: '1d', size: LOOP_TRACES.trackCount}
  ],
  entityKeySpace: 'fix',
  choices: [
    {
      stage: 'assignment',
      name: 'sigmaMeters',
      value: SIGMA_METERS,
      alternatives: [10, 40],
      sweepable: true
    },
    {
      stage: 'assignment',
      name: 'frame',
      value: 'local azimuthal-equidistant metres at the Loop centre',
      alternatives: [],
      sweepable: false
    },
    {
      stage: 'representation',
      name: 'traces',
      value: 'SIMULATED GPS traces on OpenStreetMap roads, not real vehicles',
      alternatives: [],
      sweepable: false
    }
  ],
  async build(context) {
    const fixes = context.upload(
      'fixes',
      createLoopFrame().projectArray(LOOP_TRACES.positions),
      'float32x2'
    );
    const trackOffsets = context.upload(
      'trackOffsets',
      Uint32Array.from(LOOP_TRACES.trackOffsets),
      'uint32'
    );
    const parameters = context.upload(
      'matchParameters',
      getGPUMapMatchingParameterValues({
        sigma: SIGMA_METERS,
        beta: BETA_METERS,
        searchRadius: SEARCH_RADIUS_METERS
      }),
      'float32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      fixes,
      trackOffsets,
      parameters,
      bounds: getBounds()
    });
    return {
      views: {
        fixes,
        trackOffsets,
        matchedEdges: example.matchedEdges,
        snappedPositions: example.snappedPositions,
        matchSnapDistances: example.snapDistances,
        matchBreaks: example.breaks,
        matchedCount: example.matchedCount,
        matchOverflow: example.overflow
      }
    };
  },
  capture(final) {
    if (final('matchOverflow').values[0] !== 0)
      throw new Error('Map matching dropped table entries (edge grid or route budget overflow)');
    if (final('matchedCount').values[0] === 0) throw new Error('Map matching matched no fix');
    const edges = final('matchedEdges');
    const breaks = final('matchBreaks');
    const snapped = final('snappedPositions');
    const trackOffsets = final('trackOffsets');
    // Derived geometry: runs of consecutive matched fixes inside a track, cut at GPU breaks.
    const runOffsets = [0];
    const runVertices: number[] = [];
    for (let track = 0; track < LOOP_TRACES.trackCount; track++) {
      const [first, end] = [trackOffsets.values[track], trackOffsets.values[track + 1]];
      let runLength = 0;
      const closeRun = () => {
        if (runLength >= 2) runOffsets.push(runOffsets.at(-1)! + runLength);
        else runVertices.length -= 2 * runLength;
        runLength = 0;
      };
      for (let fix = first; fix < end; fix++) {
        const isMatched = edges.values[fix] !== GPU_MAP_MATCHING_NONE;
        if (!isMatched || (breaks.values[fix] !== 0 && runLength > 0)) closeRun();
        if (isMatched) {
          runVertices.push(snapped.values[2 * fix], snapped.values[2 * fix + 1]);
          runLength++;
        }
      }
      closeRun();
    }
    const edgesWithNone = maskNoneSentinel(
      {...edges, keySpace: 'fix'},
      'directed road edge matched to each fix (no data: unmatched)'
    );
    const captures: GPUComputationCapture[] = [
      {
        ...final('fixes'),
        keySpace: undefined,
        role: 'input',
        meaning:
          'input: SIMULATED noisy GPS fixes of 3 tracks in local metres (synthetic traces on OpenStreetMap roads, not real vehicles); fix order preserved',
        geometry: {
          kind: 'lines',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'trackOffsets'
        }
      },
      trackOffsets,
      {
        ...snapped,
        id: 'matchedRuns',
        name: 'matchedRuns',
        keySpace: undefined,
        values: Float32Array.from(runVertices),
        role: 'output',
        meaning:
          'output: snapped positions of consecutive matched fixes, cut at GPU breaks and unmatched fixes (derived runs; local metres)',
        geometry: {
          kind: 'lines',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'matchedRunOffsets'
        }
      },
      {
        ...trackOffsets,
        id: 'matchedRunOffsets',
        name: 'matchedRunOffsets',
        values: Uint32Array.from(runOffsets),
        meaning: 'derived: first vertex of each matched run, plus the end'
      },
      edgesWithNone,
      {
        ...final('matchSnapDistances'),
        keySpace: 'fix',
        meaning: 'distance from each fix to its matched edge in metres (-1: unmatched)'
      },
      {
        ...breaks,
        keySpace: 'fix',
        meaning: 'break flag per fix: 0 none, 1 ordinary break, 2 budget break (GPU output)'
      },
      final('matchedCount')
    ];
    return captures;
  }
};

/** Executes the HMM map matching of simulated GPS tracks on the Chicago Loop roads. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
