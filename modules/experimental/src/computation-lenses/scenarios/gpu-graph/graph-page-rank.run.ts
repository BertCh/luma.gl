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
import {runExample} from './graph-page-rank.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-page-rank',
    directed: true,
    reverse: true,
    example: runExample
  });
  try {
    // The public contributor always returns the caller-supplied residual in this example.
    assert(outputs.residual);
    const outputCaptures = {
      'page-rank': {
        view: getGPUGraphCaptureView(fixture, 'page-rank', outputs.ranks),
        keySpace: 'vertex',
        stationValues: true
      },
      'page-rank-residual': {
        view: getGPUGraphCaptureView(fixture, 'page-rank-residual', outputs.residual)
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-page-rank',
      element: 'GPUGraphPageRank',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      iterationFrames: [{name: 'page-rank', loopId: 'gpu-graph-page-rank-iterations'}],
      choices: [
        {stage: 'parameters', name: 'damping', value: 0.85, sweepable: true},
        {stage: 'parameters', name: 'iterations', value: 40, sweepable: true}
      ],
      highlight: {
        capture: 'page-rank',
        select: (values, railFixture) => {
          let key = 0;
          for (let index = 1; index < values.length; index++) {
            if (values[index] > values[key]) key = index;
          }
          return {key, label: `${railFixture.stations.names[key]}: rank 1`};
        }
      },
      series: captures => [
        {
          name: 'final L1 residual',
          axis: 'publication',
          values: [captures['page-rank-residual'].values[0]]
        }
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public PageRank example and captures its real buffers and selected rounds. */
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
