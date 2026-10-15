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
import {runExample} from './graph-single-source-shortest-path.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-single-source-shortest-path',
    directed: true,
    weights: true,
    example: runExample
  });
  try {
    return {
      scenarioId: 'gpu-graph.graph-single-source-shortest-path',
      element: 'GPUGraphSingleSourceShortestPath',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures: {
        'travel-time': {
          view: getGPUGraphCaptureView(fixture, 'travel-time', outputs.seconds),
          keySpace: 'vertex',
          stationValues: true
        }
      },
      iterationFrames: [
        {name: 'travel-time', loopId: 'gpu-graph-single-source-shortest-path-iterations'}
      ],
      choices: [
        {stage: 'parameters', name: 'maxIterations', value: 24, sweepable: true},
        {
          stage: 'parameters',
          name: 'direction',
          value: 'outgoing',
          alternatives: ['incoming', 'both'],
          sweepable: false
        }
      ],
      highlight: {
        capture: 'travel-time',
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

/** Executes the public shortest-path example and captures travel times for selected rounds. */
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
