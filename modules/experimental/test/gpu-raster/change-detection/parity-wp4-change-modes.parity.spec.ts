// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUChangeDetection,
  getGPUChangeDetectionParameterValues,
  type GPUChangeDetectionOutput,
  type GPUChangeDetectionSettings
} from '../../../src/gpu-raster/change-detection';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-change-modes.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const FLOAT_OUTPUTS = [
  'difference',
  'logRatio',
  'percentChange',
  'tStatistic',
  'tDegreesOfFreedom',
  'tPValue',
  'senSlope',
  'mannKendallZ',
  'mannKendallP'
] as const;
type OutputName = (typeof FLOAT_OUTPUTS)[number] | 'mannKendallS' | 'significance';
type ChangeCase = {
  name: string;
  slices: number[];
  cellCount: number;
  sliceCount: number;
  mask: number[];
  settings: GPUChangeDetectionSettings;
  expected: Record<OutputName | 'welchSignificance', number[]>;
};
const fixture = decodeOracleFixture<{reference: string; cases: ChangeCase[]}>(fixtureJSON);

for (const scene of fixture.cases) {
  for (const significanceSource of ['mann-kendall', 't-test'] as const) {
    for (const omitMask of [false, true]) {
      if (omitMask && scene.mask.some(value => value === 0)) continue;
      it(`GPUChangeDetection SciPy parity: ${scene.name}/${significanceSource}/mask-free-${omitMask}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const graph = new GPUCommandGraph(device);
        const buffers: Buffer[] = [];
        const input = (values: Float32Array | Uint32Array) => {
          const buffer = createInputBuffer(device, values);
          buffers.push(buffer);
          return buffer;
        };
        const outputBuffers = new Map<OutputName, Buffer>();
        const output: GPUChangeDetectionOutput = {};
        for (const name of FLOAT_OUTPUTS) {
          const buffer = createOutputBuffer(device, scene.cellCount);
          buffers.push(buffer);
          outputBuffers.set(name, buffer);
          output[name] = importGraphBuffer(graph, name, buffer, 'float32', scene.cellCount);
        }
        const score = createOutputBuffer(device, scene.cellCount);
        const significance = createOutputBuffer(device, scene.cellCount);
        buffers.push(score, significance);
        outputBuffers.set('mannKendallS', score);
        outputBuffers.set('significance', significance);
        output.mannKendallS = importGraphBuffer(
          graph,
          'mannKendallS',
          score,
          'sint32',
          scene.cellCount
        );
        output.significance = importGraphBuffer(
          graph,
          'significance',
          significance,
          'uint32',
          scene.cellCount
        );
        const parameters = new GPUParameterBuffer(device, {
          id: 'parameters',
          format: 'float32',
          length: 5
        });
        parameters.write(getGPUChangeDetectionParameterValues(scene.settings));
        graph.add(
          new GPUChangeDetection({
            slices: importGraphBuffer(
              graph,
              'slices',
              input(new Float32Array(scene.slices)),
              'float32'
            ),
            mask: omitMask
              ? undefined
              : importGraphBuffer(graph, 'mask', input(new Uint32Array(scene.mask)), 'uint32'),
            parameters: parameters.importToGraph(graph),
            cellCount: scene.cellCount,
            sliceCount: scene.sliceCount,
            significanceSource,
            output:
              significanceSource === 'mann-kendall' ? {significance: output.significance} : output
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          for (const [name, buffer] of outputBuffers) {
            if (significanceSource === 'mann-kendall' && name !== 'significance') continue;
            const bytes = await buffer.readAsync();
            const ArrayType =
              name === 'mannKendallS'
                ? Int32Array
                : name === 'significance'
                  ? Uint32Array
                  : Float32Array;
            const actual = new ArrayType(
              bytes.buffer as ArrayBuffer,
              bytes.byteOffset,
              scene.cellCount
            );
            const expected =
              name === 'significance' && significanceSource === 't-test'
                ? scene.expected.welchSignificance
                : scene.expected[name];
            compareRasterParity(
              'GPUChangeDetection',
              `${scene.name}/${significanceSource}/mask-free-${omitMask}`,
              fixture.reference,
              name,
              actual,
              expected,
              name === 'mannKendallS' || name === 'significance' ? 'E' : 'S'
            );
          }
        } finally {
          compiled.destroy();
          parameters.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      });
    }
  }
}

type VectorCase = {
  name: string;
  cellCount: number;
  sliceCount: number;
  bandCount: number;
  slices: number[];
  mask: number[];
  settings: GPUChangeDetectionSettings;
  expected: Record<
    'difference' | 'logRatio' | 'percentChange' | 'changeMagnitude' | 'changeDirection',
    number[]
  >;
};
const vectors = decodeOracleFixture<{vectorCases: VectorCase[]}>(fixtureJSON).vectorCases;
for (const scene of vectors) {
  it(`GPUChangeDetection independent multiband parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const output: GPUChangeDetectionOutput = {};
    const outputs = new Map<string, Buffer>();
    for (const name of ['difference', 'logRatio', 'percentChange', 'changeMagnitude'] as const) {
      const buffer = createOutputBuffer(device, scene.expected[name].length);
      buffers.push(buffer);
      outputs.set(name, buffer);
      output[name] = importGraphBuffer(graph, name, buffer, 'float32');
    }
    const direction = createOutputBuffer(device, scene.cellCount);
    buffers.push(direction);
    outputs.set('changeDirection', direction);
    output.changeDirection =
      scene.bandCount === 2
        ? importGraphBuffer(graph, 'direction', direction, 'float32')
        : importGraphBuffer(graph, 'direction', direction, 'uint32');
    const slices = createInputBuffer(device, new Float32Array(scene.slices));
    const mask = createInputBuffer(device, new Uint32Array(scene.mask));
    const parameters = createInputBuffer(
      device,
      getGPUChangeDetectionParameterValues(scene.settings)
    );
    buffers.push(slices, mask, parameters);
    graph.add(
      new GPUChangeDetection({
        cellCount: scene.cellCount,
        sliceCount: scene.sliceCount,
        bandCount: scene.bandCount,
        slices: importGraphBuffer(graph, 'slices', slices, 'float32'),
        mask: importGraphBuffer(graph, 'mask', mask, 'uint32'),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32'),
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const name of [
        'difference',
        'logRatio',
        'percentChange',
        'changeMagnitude',
        'changeDirection'
      ] as const) {
        const bytes = await outputs.get(name)!.readAsync();
        const integer = name === 'changeDirection' && scene.bandCount !== 2;
        const ArrayType = integer ? Uint32Array : Float32Array;
        compareRasterParity(
          'GPUChangeDetection',
          scene.name,
          fixture.reference,
          name,
          new ArrayType(bytes.buffer as ArrayBuffer, bytes.byteOffset, scene.expected[name].length),
          scene.expected[name],
          integer ? 'E' : 'S'
        );
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
