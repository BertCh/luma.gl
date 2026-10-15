// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  buildGPUGraphScenario,
  describeGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario,
  type GPUGraphScenarioRecordOptions
} from './gpu-graph-scenario';
import {runExample} from './graph-core-number.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-core-number',
    directed: false,
    example: runExample
  });
  try {
    return {
      scenarioId: 'gpu-graph.graph-core-number',
      element: 'GPUGraphCoreNumber',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures: {
        'core-number': {
          view: getGPUGraphCaptureView(fixture, 'core-number', outputs.coreNumbers),
          keySpace: 'vertex',
          stationValues: true
        }
      },
      iterationFrames: [{name: 'core-number', loopId: 'gpu-graph-core-number-iterations'}],
      choices: [{stage: 'parameters', name: 'iterations', value: 12, sweepable: true}]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public core-number example and captures the estimate after selected rounds. */
export async function run(device: Device) {
  return recordGPUGraphScenario(await buildScenario(device));
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
