// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {getGPUNetworkKFunctionParameterValues} from '@luma.gl/experimental/gpu-network';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkFinalReader,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-k-function.example';
import {
  CONSTRUCTION_DATASETS,
  LOOP_CRASHES,
  getUndirectedLength,
  symmetrizeRoads
} from './construction-lens-support';
import {loadChicagoLoopRoads} from './network-lens-support';
import {getSearchKeySpaces} from './search-lens-support';

const scenarioId = 'gpu-network.network-k-function';
const roads = loadChicagoLoopRoads();
/** Largest distance of the K bands, in metres. */
const MAXIMUM_DISTANCE_METERS = 1000;
/** Events farther than this from every road are excluded, in metres. */
const MAXIMUM_SNAP_METERS = 30;
const BAND_COUNT = 8;
/** Small, reproducible pointwise min/mean/max envelope; this is not a confidence interval. */
const SIMULATION_COUNT = 19;
const SIMULATION_SEED = 17;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'lengthMeters',
  prepareRoads: symmetrizeRoads,
  seed: SIMULATION_SEED,
  datasetVersion: `${CONSTRUCTION_DATASETS.crashes.id}-v1`,
  retain: ['kValues', 'pairCounts', 'envelope', 'snappedEventCount', 'converged'],
  roadContext: false,
  formulas: ['gpu-network.k-function.k', 'gpu-network.snapping.nearest'],
  keySpaces: [
    ...getSearchKeySpaces(roads, LOOP_CRASHES.eventCount),
    {id: 'band', label: 'distance band', indexSpace: '1d', size: BAND_COUNT},
    {id: 'event', label: 'crash event', indexSpace: '1d', size: LOOP_CRASHES.eventCount}
  ],
  entityKeySpace: 'band',
  queuesResource: 'network-k-function-block-0-batch-0-queues',
  choices: [
    {
      stage: 'representation',
      name: 'network',
      value: 'symmetric simple network, weights in metres',
      alternatives: ['directed travel-time roads'],
      sweepable: false
    },
    {
      stage: 'summary',
      name: 'maximumDistanceMeters',
      value: MAXIMUM_DISTANCE_METERS,
      alternatives: [500, 2000],
      sweepable: true
    },
    {
      stage: 'summary',
      name: 'simulations',
      value: SIMULATION_COUNT,
      label:
        '19 Philox patterns, seed 17; pointwise minimum, mean and maximum (not a confidence interval)',
      alternatives: [0, 99],
      sweepable: false
    }
  ],
  async build(context) {
    const crashPositions = context.upload(
      'crashPositions',
      Float32Array.from(LOOP_CRASHES.positions),
      'float32x2'
    );
    const maximumDistance = context.upload(
      'maximumDistance',
      Float32Array.of(MAXIMUM_DISTANCE_METERS),
      'float32'
    );
    const maximumSnapDistance = context.upload(
      'maximumSnapDistance',
      Float32Array.of(MAXIMUM_SNAP_METERS),
      'float32'
    );
    // Total length with every road once: half of the symmetric weight sum (data preparation).
    const networkLength = context.upload(
      'networkLength',
      Float32Array.of(getUndirectedLength(context.data)),
      'float32'
    );
    const parameters = context.upload(
      'kParameters',
      getGPUNetworkKFunctionParameterValues({
        seed: SIMULATION_SEED,
        activeSimulations: SIMULATION_COUNT
      }),
      'uint32'
    );
    const example = await runExample(context.device, {
      graph: context.graph,
      roads: context.topology,
      crashPositions,
      maximumDistance,
      maximumSnapDistance,
      networkLength,
      parameters,
      simulationCount: SIMULATION_COUNT
    });
    return {
      views: {
        crashPositions,
        kValues: example.kValues,
        pairCounts: example.pairCounts,
        envelope: example.envelope,
        snappedEventCount: example.snappedEventCount,
        converged: example.converged
      }
    };
  },
  capture: captureNetworkKFunction
};

/** Arranges GPU readbacks as band-aligned observed values and envelope columns. @internal */
export function captureNetworkKFunction(final: NetworkFinalReader): GPUComputationCapture[] {
  if (final('converged').values[0] === 0)
    throw new Error('Network K function did not converge within maximumIterations = 24');
  const snapped = final('snappedEventCount').values[0];
  if (snapped < 2) throw new Error('Network K function: fewer than two events snapped');
  return [
    {
      ...final('crashPositions'),
      keySpace: undefined,
      role: 'input',
      meaning: `input: ${LOOP_CRASHES.eventCount} sampled 2023 traffic crashes, longitude/latitude as uploaded (every ${LOOP_CRASHES.sampleStep}th crash within 25 m of the roads)`,
      geometry: {kind: 'points', coordinateModel: 'geographic'}
    },
    {
      ...final('pairCounts'),
      values: Uint32Array.from(final('pairCounts').values).slice(0, BAND_COUNT),
      keySpace: 'band',
      role: 'output',
      meaning: 'output: unordered crash pairs with network distance below each band threshold'
    },
    {
      ...final('kValues'),
      values: Float32Array.from(final('kValues').values).slice(0, BAND_COUNT),
      keySpace: 'band',
      role: 'output',
      meaning: 'output: network K of the observed crashes at each band threshold, in metres'
    },
    {
      ...final('kValues'),
      id: 'bandThreshold',
      name: 'bandThreshold',
      keySpace: 'band',
      values: Float32Array.from(
        {length: BAND_COUNT},
        (_, band) => (MAXIMUM_DISTANCE_METERS * band) / (BAND_COUNT - 1)
      ),
      role: 'input',
      meaning:
        'input: distance threshold of each band in metres, maximumDistance * k / (bandCount - 1) from the uploaded maximum distance'
    },
    ...['minimum', 'mean', 'maximum'].map((statistic, row) => ({
      ...final('envelope'),
      id: `envelope-${statistic}`,
      name: `envelope-${statistic}`,
      keySpace: 'band',
      values: Float32Array.from(final('envelope').values).slice(
        row * BAND_COUNT,
        (row + 1) * BAND_COUNT
      ),
      role: 'output' as const,
      meaning: `output: pointwise ${statistic} K in metres across ${SIMULATION_COUNT} Philox patterns (seed ${SIMULATION_SEED}); not a confidence interval`
    })),
    final('snappedEventCount')
  ];
}

/** Executes the network K function of Chicago Loop crashes and reads the threshold table back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
