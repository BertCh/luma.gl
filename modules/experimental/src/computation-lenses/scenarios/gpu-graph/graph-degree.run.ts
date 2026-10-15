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
import {runExample} from './graph-degree.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-degree',
    directed: true,
    example: runExample
  });
  try {
    return {
      scenarioId: 'gpu-graph.graph-degree',
      element: 'GPUGraphDegree',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures: {
        'out-degree': {
          view: getGPUGraphCaptureView(fixture, 'out-degree', outputs.outDegrees),
          keySpace: 'vertex',
          stationValues: true
        }
      },
      choices: [
        {
          stage: 'parameters',
          name: 'direction',
          value: 'outgoing',
          alternatives: ['incoming'],
          sweepable: false
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public degree example and captures its real per-station counts. */
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
