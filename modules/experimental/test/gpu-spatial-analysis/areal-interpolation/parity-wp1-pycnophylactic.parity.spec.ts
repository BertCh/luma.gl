// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUPycnophylactic} from '../../../src/gpu-spatial-analysis/areal-interpolation';
import fixture from '../../oracles/fixtures/parity-wp1-pycnophylactic.json';
import {compareReference} from '../../oracles/parity-wp1-round8-compare';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of fixture.cases) {
  it(`GPUPycnophylactic R pycno adapted reference ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const mass = rig.output('float32', scene.zones.length);
    const density = rig.output('float32', scene.zones.length);
    const converged = rig.output('uint32', 1);
    const iterationCount = rig.output('uint32', 1);
    const invalidCount = rig.output('uint32', 1);
    try {
      rig.run(
        new GPUPycnophylactic({
          grid: {
            width: scene.width,
            height: scene.height,
            affine: [1, 0, 0, 0, 1, 0],
            pixelInterpretation: 'area'
          },
          zones: rig.input(new Uint32Array(scene.zones), 'uint32'),
          totals: rig.input(new Float32Array(scene.totals), 'float32'),
          zoneCount: scene.totals.length,
          kernel: scene.kernel === 'rook' ? 'rook' : 'box',
          maximumIterations: scene.iterations,
          ...(scene.tolerance >= 0
            ? {tolerance: rig.input(new Float32Array([scene.tolerance]), 'float32')}
            : {}),
          output: {
            mass,
            density,
            status: {invalidCount, ...(scene.tolerance >= 0 ? {converged, iterationCount} : {})}
          }
        })
      );
      const result = await rig.readFloat(mass);
      const compare = (metric: string, actual: number[], expected: number[], exact = false) =>
        compareReference('GPUPycnophylactic', scene, metric, actual, expected, exact);
      compare('every-cell-mass', result, scene.expected.mass);
      compare('every-cell-density', await rig.readFloat(density), scene.expected.mass);
      compare(
        'every-zone-total',
        scene.totals.map((_, zone) =>
          result.reduce((sum, value, cell) => sum + (scene.zones[cell] === zone ? value : 0), 0)
        ),
        scene.zoneSums
      );
      compare('invalid-zones', await rig.readUint(invalidCount), [0], true);
      compare('negative-cell-count', [result.filter(value => value < 0).length], [0], true);
      if (scene.tolerance >= 0) {
        compare('converged', await rig.readUint(converged), [scene.expected.converged], true);
        compare(
          'iteration-count',
          await rig.readUint(iterationCount),
          [scene.expected.iterations],
          true
        );
      }
    } finally {
      rig.destroy();
    }
  });
}
