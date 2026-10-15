// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkAccessibility,
  getGPUNetworkAccessibilityParameterValues
} from '../../../src/gpu-network/network-accessibility';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity} from '../parity-wp3-comparison';
import {
  ACCESSIBILITY_FIXTURES,
  type AccessibilityInput,
  type AccessibilityCase,
  type OutputName
} from './parity-wp3-network-accessibility-fixture';

async function compareScoring(
  device: Device,
  fixture: AccessibilityInput,
  cases: AccessibilityCase[],
  orientation: 'opportunity-rows' | 'origin-rows'
): Promise<void> {
  const graph = new GPUCommandGraph(device, {
    id: `parity-wp3-accessibility-${fixture.case}-${orientation}`
  });
  const buffers: Buffer[] = [];
  const outputs: {name: OutputName; buffer: Buffer; length: number}[] = [];
  const parameters = new GPUParameterBuffer(device, {
    id: 'parity-wp3-accessibility-parameters',
    format: 'float32',
    length: 4
  });
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  const input = (name: string, values: Float32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32', values.length);
  };
  const output = (name: OutputName, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.push({name, buffer, length});
    return importGraphBuffer(graph, name, buffer, 'float32', length);
  };
  try {
    const costs = Float32Array.from(fixture.costs);
    const facilityCount = fixture.supply.length;
    const matrix =
      orientation === 'opportunity-rows'
        ? costs
        : Float32Array.from(
            {length: costs.length},
            (_, index) =>
              costs[
                (index % facilityCount) * fixture.originCount + Math.floor(index / facilityCount)
              ]
          );
    graph.add(
      new GPUNetworkAccessibility({
        costs: input('costs', matrix),
        orientation,
        opportunityWeights: input('supply', Float32Array.from(fixture.supply)),
        parameters: parameters.importToGraph(graph),
        cumulative: output('cumulative', fixture.originCount),
        gravity: output('gravity', fixture.originCount),
        catchment:
          orientation === 'opportunity-rows'
            ? {
                demand: input('demand', Float32Array.from(fixture.demand)),
                output: output('catchment', fixture.originCount),
                ratios: output('ratios', facilityCount)
              }
            : undefined
      })
    );
    compiled = graph.compile();
    for (const reference of cases) {
      parameters.write(getGPUNetworkAccessibilityParameterValues(reference.settings));
      submitGraph(device, compiled, undefined);
      for (const {name, buffer, length} of outputs) {
        const actual = await readFloat32(buffer, length);
        compareParity(
          'GPUNetworkAccessibility',
          reference,
          name,
          actual,
          reference.expected[name],
          reference.tolerances[name],
          `${reference.case}/${orientation}`
        );
      }
    }
  } finally {
    compiled?.destroy();
    parameters.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const {input, cases} of ACCESSIBILITY_FIXTURES) {
  for (const orientation of ['opportunity-rows', 'origin-rows'] as const) {
    it(`GPUNetworkAccessibility NetworkX/access parity: ${input.case}/${orientation}`, async context => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device, context);
      await compareScoring(device, input, cases, orientation);
    });
  }
}
