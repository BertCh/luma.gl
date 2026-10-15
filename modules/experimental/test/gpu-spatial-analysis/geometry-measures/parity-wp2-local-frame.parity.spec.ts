// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  createGPULocalFrame,
  type GPULocalFrameKind
} from '../../../src/analysis-core/contracts/local-frame';
import {GPULocalFrameProjection} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-local-frame.json';

type FrameCase = {
  name: string;
  origin: [number, number];
  bounds: [number, number, number, number];
  kind: GPULocalFrameKind;
  precision: 'float32' | 'float64';
  positions: number[];
  expected: number[];
  validity: number[];
  absolute: number;
  relative: number;
};
const fixture = decodeOracleFixture<{reference: string; cases: FrameCase[]}>(fixtureJson);
for (const entry of fixture.cases) {
  it(`GPULocalFrameProjection direct PROJ parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    const length = entry.validity.length;
    const words =
      entry.precision === 'float64'
        ? createInputBuffer(device, new Uint32Array(Float64Array.from(entry.positions).buffer))
        : undefined;
    const positions = words
      ? importGraphBuffer(rig.graph, 'binary64-positions', words, 'uint32x4', length)
      : rig.input('positions', entry.positions, 'float32x2').view;
    const projected = rig.output('projected', length, 'float32x2');
    const validity = rig.output('validity', length, 'uint32');
    const contributor = new GPULocalFrameProjection({
      frame: createGPULocalFrame({origin: entry.origin, kind: entry.kind}),
      positions,
      output: projected.view,
      validity: validity.view,
      bounds: entry.bounds,
      tolerance: 0.01
    });
    rig.graph.add(contributor);
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      compareParity(
        'GPULocalFrameProjection',
        entry.name,
        fixture.reference,
        'every validity row',
        await readUint32(validity.buffer, length),
        entry.validity
      );
      compareParity(
        'GPULocalFrameProjection',
        entry.name,
        fixture.reference,
        'every projected coordinate in meters',
        await readFloat32(projected.buffer, length * 2),
        entry.expected,
        entry.absolute,
        entry.relative
      );
    } finally {
      compiled.destroy();
      contributor.destroy();
      words?.destroy();
      rig.destroy();
      device.destroy?.();
    }
  }, 120_000);
}
