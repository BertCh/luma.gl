// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPURasterSieve,
  getGPURasterSieveParameterValues
} from '../../../src/gpu-raster/raster-patches';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import fixture from '../../oracles/fixtures/parity-wp4-sieve-modes.json';
import {sieveMetrics} from '../../oracles/parity-wp4-round7-fixtures';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import type {
  GPURasterSieveMode,
  GPURasterSieveConnectivity
} from '../../../src/gpu-raster/raster-patches';
const sieveCases = decodeOracleFixture<{
  cases: (Omit<(typeof fixture.cases)[number], 'mode' | 'connectivity'> & {
    mode: GPURasterSieveMode;
    connectivity: GPURasterSieveConnectivity;
  })[];
}>(fixture).cases;

for (const scene of sieveCases) {
  for (const labelsOnly of [false, true]) {
    it(`GPURasterSieve independent parity: ${scene.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new ParityRasterGraph(device, scene.name);
      const parameters = new GPUParameterBuffer(device, {
        id: 'parameters',
        format: 'uint32',
        length: 4
      });
      const validity = new GPUParameterBuffer(device, {
        id: 'validity',
        format: 'uint32',
        length: scene.width * scene.height
      });
      const componentCount = new GPUParameterBuffer(device, {
        id: 'component-count',
        format: 'uint32',
        length: 1
      });
      const converged = new GPUParameterBuffer(device, {
        id: 'converged',
        format: 'uint32',
        length: 1
      });
      const overflow = new GPUParameterBuffer(device, {
        id: 'overflow',
        format: 'uint32',
        length: 1
      });
      const outputs = {
        labels: rig.output('output-labels', 'uint32', scene.width * scene.height),
        patchTargets: rig.output('targets', 'uint32', scene.capacity),
        sievedCount: rig.output('count', 'uint32', 1)
      };
      rig.graph.add(
        new GPURasterSieve({
          grid: {
            width: scene.width,
            height: scene.height,
            affine: [1, 0, 0, 0, 1, 0],
            pixelInterpretation: 'area'
          },
          labels: rig.input('labels', 'uint32', new Uint32Array(scene.labels)),
          validity: validity.importToGraph(rig.graph),
          status: {
            componentCount: componentCount.importToGraph(rig.graph),
            converged: converged.importToGraph(rig.graph),
            overflow: overflow.importToGraph(rig.graph)
          },
          patchCapacity: scene.capacity,
          mode: scene.mode,
          connectivity: scene.connectivity,
          parameters: parameters.importToGraph(rig.graph),
          output: {
            labels: outputs.labels.view,
            patchTargets: labelsOnly ? undefined : outputs.patchTargets.view,
            sievedCount: labelsOnly ? undefined : outputs.sievedCount.view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const [frameIndex, frame] of scene.frames.entries()) {
          validity.write(new Uint32Array(scene.width * scene.height).fill(frame.valid));
          componentCount.write(Uint32Array.of(frame.componentCount));
          converged.write(Uint32Array.of(frame.converged));
          overflow.write(Uint32Array.of(frame.overflow));
          parameters.write(getGPURasterSieveParameterValues(frame));
          submitGraph(device, compiled, undefined);
          for (const metric of sieveMetrics) {
            if (labelsOnly && metric !== 'labels') continue;
            const expected = frame.expected[metric];
            compareRasterParity(
              'GPURasterSieve',
              `${scene.name}/labels-only-${labelsOnly}/frame-${frameIndex}`,
              fixture.reference,
              metric,
              await readUint32(outputs[metric].buffer, expected.length),
              expected,
              'E'
            );
          }
        }
      } finally {
        compiled.destroy();
        parameters.destroy();
        validity.destroy();
        componentCount.destroy();
        converged.destroy();
        overflow.destroy();
        rig.destroy();
      }
    });
  }
}
