// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUWeightedOverlay,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  type GPUWeightedOverlaySettings
} from '../../../src/gpu-raster/raster-algebra';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-overlay-showcase.json';
type OverlayFixture = {
  reference: string;
  cellCount: number;
  layerCount: number;
  remapBreaks: number[];
  cases: {
    name: string;
    settings: GPUWeightedOverlaySettings;
    frames: {
      stack: number[];
      remapValues: number[];
      expected: {score: number[]; scoreRange: number[]};
    }[];
  }[];
};
const fixture = decodeOracleFixture<OverlayFixture>(fixtureJSON);
for (const scene of fixture.cases)
  it(`Site suitability independent overlay: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const owner = new ParityRasterGraph(device, scene.name);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: getGPUWeightedOverlayParameterLength(fixture.layerCount)
    });
    const stack = owner.output('stack', 'float32', fixture.cellCount * fixture.layerCount);
    const table = owner.output('table', 'float32', fixture.layerCount * 9);
    const outputs = {
      score: owner.output('score', 'float32', fixture.cellCount),
      scoreRange: owner.output('range', 'float32', 2)
    };
    owner.graph.add(
      new GPUWeightedOverlay({
        stack: stack.view,
        layerCount: fixture.layerCount,
        cellCount: fixture.cellCount,
        noDataValue: -9999,
        parameters: parameters.importToGraph(owner.graph),
        remapBreaks: owner.input('breaks', 'float32', new Float32Array(fixture.remapBreaks)),
        remapValues: table.view,
        maximumBreakCount: 8,
        output: {score: outputs.score.view, scoreRange: outputs.scoreRange.view}
      })
    );
    const compiled = owner.graph.compile();
    try {
      parameters.write(getGPUWeightedOverlayParameterValues(scene.settings));
      for (const [index, frame] of scene.frames.entries()) {
        stack.buffer.write(new Float32Array(frame.stack));
        table.buffer.write(new Float32Array(frame.remapValues));
        submitGraph(device, compiled, undefined);
        for (const metric of ['score', 'scoreRange'] as const)
          compareRoundFour(
            'GPUWeightedOverlay',
            `${scene.name}/frame-${index}`,
            fixture.reference,
            metric,
            await readFloat32(outputs[metric].buffer, frame.expected[metric].length),
            frame.expected[metric],
            'R'
          );
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      owner.destroy();
    }
  });
