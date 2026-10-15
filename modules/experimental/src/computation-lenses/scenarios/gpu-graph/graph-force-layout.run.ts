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
import {runExample} from './graph-force-layout.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  // Undirected: attraction and repulsion read the symmetric forward CSR; weights are ignored.
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-force-layout',
    directed: false,
    example: runExample
  });
  try {
    const outputCaptures = {
      // Planar km start positions, uploaded once; the layout moves a separate copy of them.
      'force-layout-start': {
        view: getGPUGraphCaptureView(fixture, 'force-layout-start', outputs.startPositions),
        keySpace: 'vertex'
      },
      'force-layout': {
        view: getGPUGraphCaptureView(fixture, 'force-layout', outputs.positions),
        keySpace: 'vertex',
        geometry: {
          kind: 'graph' as const,
          offsetsCapture: 'csr-offsets',
          targetsCapture: 'csr-neighbors',
          layoutCapture: 'force-layout',
          coordinateModel: 'projected' as const
        }
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-force-layout',
      element: 'GPUGraphForceLayout',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      iterationFrames: [{name: 'force-layout', loopId: 'gpu-graph-force-layout-iterations'}],
      choices: [
        {stage: 'parameters', name: 'iterationsPerFrame', value: 100, sweepable: true},
        {stage: 'parameters', name: 'repulsion', value: 5, sweepable: true},
        {stage: 'parameters', name: 'attraction', value: 0.02, sweepable: true},
        {stage: 'parameters', name: 'gravity', value: 0.05, sweepable: true},
        {stage: 'parameters', name: 'maxVelocity', value: 2, sweepable: true}
      ]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public force-layout example and captures its real buffers and selected steps. */
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
