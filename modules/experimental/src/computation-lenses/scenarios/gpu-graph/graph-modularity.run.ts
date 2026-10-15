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
import {runExample} from './graph-modularity.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  // Unweighted: the contributor treats missing edge weights as one, so tracks count equally.
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-modularity',
    directed: false,
    example: runExample
  });
  try {
    // The example asked the contributor to publish contributions and returns them.
    assert(outputs.contributions);
    const outputCaptures = {
      'modularity-partition': {
        view: getGPUGraphCaptureView(fixture, 'modularity-partition', outputs.communities),
        keySpace: 'vertex',
        stationValues: true
      },
      'modularity-score': {
        view: getGPUGraphCaptureView(fixture, 'modularity-score', outputs.score)
      },
      // Indexed by community label, not by station: deliberately no vertex key space.
      'modularity-contributions': {
        view: getGPUGraphCaptureView(fixture, 'modularity-contributions', outputs.contributions)
      },
      'modularity-valid': {
        view: getGPUGraphCaptureView(fixture, 'modularity-valid', outputs.valid)
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-modularity',
      element: 'GPUGraphModularity',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      choices: [
        {stage: 'parameters', name: 'partitionIterations', value: 16, sweepable: true},
        {stage: 'parameters', name: 'resolution', value: 1, sweepable: true}
      ],
      series: captures => [
        {
          name: 'modularity score',
          axis: 'publication',
          values: [captures['modularity-score'].values[0]]
        },
        {
          name: 'score valid (1 = yes)',
          axis: 'publication',
          values: [captures['modularity-valid'].values[0]]
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public modularity example on a GPU label-propagation partition. */
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
