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
import {runExample} from './graph-spatial-force-layout.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  // Undirected: attraction reads the symmetric forward CSR; the grid replaces all-pairs repulsion.
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-spatial-force-layout',
    directed: false,
    example: runExample
  });
  try {
    const outputCaptures = {
      // Planar km start positions, uploaded once; the layout moves a separate copy of them.
      'spatial-force-layout-start': {
        view: getGPUGraphCaptureView(fixture, 'spatial-force-layout-start', outputs.startPositions),
        keySpace: 'vertex'
      },
      'spatial-force-layout': {
        view: getGPUGraphCaptureView(fixture, 'spatial-force-layout', outputs.positions),
        keySpace: 'vertex',
        geometry: {
          kind: 'graph' as const,
          offsetsCapture: 'csr-offsets',
          targetsCapture: 'csr-neighbors',
          layoutCapture: 'spatial-force-layout',
          coordinateModel: 'projected' as const
        }
      }
    };
    return {
      scenarioId: 'gpu-graph.graph-spatial-force-layout',
      element: 'GPUGraphSpatialForceLayout',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures,
      // The cell-center pass binds one row per grid cell (16 x 16), a key space beyond stations.
      extraKeySpaces: [{id: 'cell', label: 'grid cell', size: 256}],
      iterationFrames: [
        {name: 'spatial-force-layout', loopId: 'gpu-graph-spatial-force-layout-iterations'}
      ],
      choices: [
        {stage: 'parameters', name: 'iterationsPerFrame', value: 64, sweepable: true},
        {stage: 'parameters', name: 'theta', value: 0.6, sweepable: true},
        {stage: 'parameters', name: 'nearCellRadius', value: 1, sweepable: true},
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

/** Executes the public spatial force-layout example and captures its real buffers and selected steps. */
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
