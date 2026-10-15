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
import {runExample} from './graph-modularity-optimization.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  // Unweighted: the contributor treats missing edge weights as one, so tracks count equally.
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-modularity-optimization',
    directed: false,
    example: runExample
  });
  try {
    const outputCaptures = {
      'modularity-seed': {
        view: getGPUGraphCaptureView(fixture, 'modularity-seed', outputs.seedCommunities),
        keySpace: 'vertex'
      },
      'modularity-optimization': {
        view: getGPUGraphCaptureView(fixture, 'modularity-optimization', outputs.communities),
        keySpace: 'vertex',
        stationValues: true
      },
      'modularity-optimization-score': {
        view: getGPUGraphCaptureView(fixture, 'modularity-optimization-score', outputs.modularity)
      },
      'modularity-optimization-converged': {
        view: getGPUGraphCaptureView(
          fixture,
          'modularity-optimization-converged',
          outputs.converged
        )
      },
      'modularity-optimization-valid': {
        view: getGPUGraphCaptureView(fixture, 'modularity-optimization-valid', outputs.valid)
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-modularity-optimization',
      element: 'GPUGraphModularityOptimization',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      iterationFrames: [
        {name: 'modularity-optimization', loopId: 'gpu-graph-modularity-optimization-iterations'}
      ],
      choices: [
        {stage: 'parameters', name: 'seedIterations', value: 16, sweepable: true},
        {stage: 'parameters', name: 'iterations', value: 64, sweepable: true},
        {stage: 'parameters', name: 'resolution', value: 1, sweepable: true},
        {stage: 'parameters', name: 'minimumGain', value: 0, sweepable: true}
      ],
      series: captures => [
        {
          name: 'final modularity',
          axis: 'publication',
          values: [captures['modularity-optimization-score'].values[0]]
        },
        {
          name: 'local optimum reached (1 = yes)',
          axis: 'publication',
          values: [captures['modularity-optimization-converged'].values[0]]
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public modularity-optimization example and captures its buffers and moves. */
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
