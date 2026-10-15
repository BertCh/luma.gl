// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';
import {
  buildGPUGraphScenario,
  describeGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario,
  type GPUGraphScenarioRecordOptions
} from './gpu-graph-scenario';
import {runExample} from './graph-label-propagation.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-label-propagation',
    directed: false,
    example: runExample
  });
  try {
    // The public contributor returns the caller-supplied convergence scalar in this example.
    assert(outputs.converged);
    const outputCaptures = {
      'label-propagation': {
        view: getGPUGraphCaptureView(fixture, 'label-propagation', outputs.communities),
        keySpace: 'vertex',
        stationValues: true
      },
      'label-propagation-converged': {
        view: getGPUGraphCaptureView(fixture, 'label-propagation-converged', outputs.converged)
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-label-propagation',
      element: 'GPUGraphLabelPropagation',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      iterationFrames: [
        {name: 'label-propagation', loopId: 'gpu-graph-label-propagation-iterations'}
      ],
      choices: [{stage: 'parameters', name: 'iterations', value: 16, sweepable: true}],
      series: captures => [
        {
          name: 'converged (1 = fixed point)',
          axis: 'publication',
          values: [captures['label-propagation-converged'].values[0]]
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public label-propagation example and captures its real buffers and rounds. */
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
