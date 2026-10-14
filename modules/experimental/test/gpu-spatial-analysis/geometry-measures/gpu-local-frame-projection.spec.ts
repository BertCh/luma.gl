// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import {GPULocalFrameProjection} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection';
import {requireTestDevice} from '../../utils/require-test-device';

// Expected values come from the binary64 CPU frame, which local-frame.node.spec.ts pins to pyproj.
const POINTS: readonly (readonly [number, number])[] = [
  [8.5417, 47.3769],
  [8.0123456789, 47.1],
  [9.3, 47.9],
  [8.54170001, 47.37690001],
  [7.6, 46.6]
];

for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
  it(`GPULocalFrameProjection projects binary64 lon/lat into a ${kind} frame`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const frame = createGPULocalFrame({origin: [8.5417, 47.3769], kind});
    const values = new Float64Array(POINTS.flat());
    const inputBuffer = device.createBuffer({
      byteLength: values.byteLength,
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    inputBuffer.write(new Uint32Array(values.buffer));
    const outputBuffer = device.createBuffer({
      byteLength: POINTS.length * 8,
      usage: Buffer.STORAGE | Buffer.COPY_SRC
    });
    const graph = new GPUCommandGraph(device, {id: `local-frame-${kind}`});
    const input = graph.createDataView(
      graph.importBuffer(
        {id: 'positions', byteLength: inputBuffer.byteLength, usage: inputBuffer.usage},
        inputBuffer
      ),
      {format: 'uint32x4', length: POINTS.length}
    );
    const output = graph.createDataView(
      graph.importBuffer(
        {id: 'projected', byteLength: outputBuffer.byteLength, usage: outputBuffer.usage},
        outputBuffer
      ),
      {format: 'float32x2', length: POINTS.length}
    );
    const contributor = new GPULocalFrameProjection({
      frame,
      positions: input,
      output,
      bounds: [7.5, 46.5, 9.5, 48]
    });
    expect(contributor.spatialContext.frame?.id).toBe(frame.spatialContext.frame?.id);
    graph.add(contributor);
    const compiled = graph.compile();
    const encoder = device.createCommandEncoder({id: 'local-frame-projection'});
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());

    const bytes = await outputBuffer.readAsync(0, POINTS.length * 8);
    const actual = new Float32Array(bytes.buffer, bytes.byteOffset, POINTS.length * 2);
    POINTS.forEach(([longitude, latitude], index) => {
      const [x, y] = frame.project(longitude, latitude);
      // Plan tolerance 0.01 m plus float32 rounding of up to ~100 km frame coordinates.
      expect(Math.abs(actual[index * 2] - x), `row ${index} x`).toBeLessThan(0.02);
      expect(Math.abs(actual[index * 2 + 1] - y), `row ${index} y`).toBeLessThan(0.02);
    });
    // The origin row projects to the frame origin, not to the plan's bounds center.
    expect(Math.abs(actual[0])).toBeLessThan(0.01);
    expect(Math.abs(actual[1])).toBeLessThan(0.01);

    compiled.destroy();
    contributor.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  });
}
