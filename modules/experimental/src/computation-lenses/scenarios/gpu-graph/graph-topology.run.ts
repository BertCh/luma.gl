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
import {runExample} from './graph-topology.example';

async function buildScenario(device: Device): Promise<GPUGraphScenarioRecordOptions> {
  const {fixture, outputs} = await buildGPUGraphScenario(device, {
    id: 'gpu-graph.graph-topology',
    directed: true,
    example: runExample
  });
  try {
    // The example's own rebuild owns these buffers; the fixture's rebuild only reserves inputs.
    const offsets = getGPUGraphCaptureView(fixture, 'csr-row-offsets', outputs.offsets);
    // A second view of the same physical offsets keeps the first N rows: one start offset per
    // station. The final row is the total adjacency count and is drawn by the input graph.
    const stationOffsets = fixture.commandGraph.createDataView(offsets.buffer, {
      format: 'uint32',
      length: fixture.stations.vertexCount,
      byteOffset: offsets.byteOffset
    });
    return {
      scenarioId: 'gpu-graph.graph-topology',
      element: 'GPUGraphTopology',
      fixture,
      compiled: fixture.commandGraph.compile(),
      outputCaptures: {
        'csr-row-offsets': {view: stationOffsets, keySpace: 'vertex', stationValues: true}
      },
      choices: [{stage: 'topology', name: 'directed', value: true, sweepable: false}]
    };
  } catch (error) {
    fixture.destroy();
    throw error;
  }
}

/** Executes the public topology example and captures its real CSR offsets and neighbors. */
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
