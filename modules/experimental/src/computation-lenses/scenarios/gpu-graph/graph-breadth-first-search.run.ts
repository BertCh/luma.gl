// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';
import {
  validateGPUComputationSidecars,
  type GPUComputationCapture,
  type GPUComputationRecordWithSidecars
} from '@luma.gl/gpgpu';
import {
  buildGPUGraphScenario,
  describeGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario,
  type GPUGraphScenarioRecordOptions
} from './gpu-graph-scenario';
import {runExample} from './graph-breadth-first-search.example';

const UNREACHABLE_VERTEX = 0xffffffff;

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-breadth-first-search',
    directed: true,
    example: runExample
  });
  try {
    return {
      scenarioId: 'gpu-graph.graph-breadth-first-search',
      element: 'GPUGraphBreadthFirstSearch',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures: {
        'hop-distance': {
          view: getGPUGraphCaptureView(fixture, 'hop-distance', outputs.hops),
          keySpace: 'vertex',
          stationValues: true
        }
      },
      // The kernels bind seed rows and compact frontier slots besides stations.
      extraKeySpaces: [
        {id: 'seed', label: 'seed', size: 1},
        {id: 'frontier', label: 'frontier slot', size: fixture.stations.vertexCount}
      ],
      iterationFrames: [{name: 'hop-distance', loopId: 'gpu-graph-breadth-first-search-depths'}],
      choices: [
        {stage: 'parameters', name: 'maxDepth', value: 16, sweepable: true},
        {
          stage: 'parameters',
          name: 'direction',
          value: 'outgoing',
          alternatives: ['incoming', 'both'],
          sweepable: false
        }
      ],
      highlight: {
        capture: 'hop-distance',
        select: (_values, railFixture) => {
          const key = railFixture.stations.names.indexOf('Utrecht Centraal');
          return {key, label: 'Utrecht Centraal: source'};
        }
      }
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/**
 * Marks the unreached sentinel as no data without changing any reached hop count; the GPU wrote
 * 0xffffffff, which would otherwise dominate every colour scale.
 */
export function markUnreachableAsNoData(
  record: GPUComputationRecordWithSidecars,
  captureName: string
): GPUComputationRecordWithSidecars {
  const convert = (capture: GPUComputationCapture): GPUComputationCapture =>
    capture.name === captureName
      ? {
          ...capture,
          elementType: 'f32',
          values: Float32Array.from(capture.values, value =>
            value >= UNREACHABLE_VERTEX ? Number.NaN : value
          )
        }
      : capture;
  // recordGPUGraphScenario always attaches both sidecars.
  assert(record.captures);
  const captures = {...record.captures, snapshots: record.captures.snapshots.map(convert)};
  const issues = validateGPUComputationSidecars(record, {metrics: record.metrics, captures});
  if (!issues.valid) {
    throw new Error(`Invalid gpu-graph lens record: ${JSON.stringify(issues.issues)}`);
  }
  return {...record, captures};
}

/** Executes the public search example and captures hop distances for the selected depths. */
export async function run(device: Device) {
  return markUnreachableAsNoData(
    await recordGPUGraphScenario(await buildScenario(device)),
    'hop-distance'
  );
}

/** Builds the same compiled scenario on NullDevice, with no submission or fabricated values. */
export async function describeStructure(device: Device) {
  const options = await buildScenario(device);
  try {
    return describeGPUGraphScenario(options);
  } finally {
    try {
      options.compiled.destroy();
    } finally {
      options.fixture.destroy();
    }
  }
}
