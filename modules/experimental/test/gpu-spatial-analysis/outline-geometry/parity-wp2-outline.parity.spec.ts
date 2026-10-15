// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUOutlineGeometry,
  getGPUOutlineGeometryParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry';
import {outlineFixtures} from '../../oracles/parity-wp2-round5-outline-fixtures';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of outlineFixtures)
  for (const entry of fixture.cases) {
    it(`GPUOutlineGeometry GEOS parity: ${entry.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = createParityGraph(device);
      const positions = rig.input('positions', entry.positions, 'float32x2').view;
      const offsets = rig.input('offsets', entry.lineOffsets, 'uint32').view;
      const identity = rig.input(
        'identity',
        entry.lineOffsets.map((_, index) => index),
        'uint32'
      ).view;
      const output = rig.output('output', entry.frames[0].positions.length / 2, 'float32x2');
      const parameters = rig.input('parameters', [0, 0, 0, 0], 'float32');
      const common = {
        positions,
        parameters: parameters.view,
        joinSegments: 16,
        output: {positions: output.view}
      };
      rig.graph.add(
        new GPUOutlineGeometry(
          entry.geometryType === 'points'
            ? {...common, geometryType: 'points'}
            : entry.geometryType === 'lines'
              ? {...common, geometryType: 'lines', lineOffsets: offsets}
              : {
                  ...common,
                  geometryType: 'polygons',
                  featureOffsets: identity,
                  polygonOffsets: identity,
                  ringOffsets: offsets
                }
        )
      );
      const compiled = rig.graph.compile();
      try {
        for (const frame of [...entry.frames, entry.frames[0]]) {
          parameters.buffer.write(getGPUOutlineGeometryParameterValues({distance: frame.distance}));
          submitGraph(device, compiled, undefined);
          compareParity(
            'GPUOutlineGeometry',
            `${entry.name}/width=${frame.distance}`,
            fixture.reference,
            'every triangle coordinate including degenerate quads',
            await readFloat32(output.buffer, frame.positions.length),
            frame.positions,
            entry.absolute
          );
        }
      } finally {
        compiled.destroy();
        rig.destroy();
        device.destroy?.();
      }
    });
  }
