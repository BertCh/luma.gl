// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPURasterPatchMetrics} from '../../../src/gpu-raster/raster-patches';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {patchCases, patchMetrics} from '../../oracles/parity-wp4-round6-fixtures';

for (const scene of patchCases) {
  for (const selected of ['scene', ...patchMetrics] as const) {
    it(`GPURasterPatchMetrics independent polygon/region parity: ${scene.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const [labels, validity] = await Promise.all(
        [scene.labels, scene.validity].map(decodePackedNumbers)
      );
      const rig = new ParityRasterGraph(device, scene.name);
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
        pixelCounts: rig.output('counts', 'uint32', scene.capacity),
        perimeterFaceCounts: rig.output('faces', 'uint32', scene.capacity),
        minColumns: rig.output('min-columns', 'uint32', scene.capacity),
        minRows: rig.output('min-rows', 'uint32', scene.capacity),
        maxColumns: rig.output('max-columns', 'uint32', scene.capacity),
        maxRows: rig.output('max-rows', 'uint32', scene.capacity),
        areas: rig.output('areas', 'float32', scene.capacity),
        perimeters: rig.output('perimeters', 'float32', scene.capacity)
      };
      rig.graph.add(
        new GPURasterPatchMetrics({
          grid: scene.grid,
          labels: rig.input('labels', 'uint32', new Uint32Array(labels)),
          validity: rig.input('validity', 'uint32', new Uint32Array(validity)),
          countRasterBorder: scene.countRasterBorder,
          status: {
            componentCount: componentCount.importToGraph(rig.graph),
            converged: converged.importToGraph(rig.graph),
            overflow: overflow.importToGraph(rig.graph)
          },
          output: Object.fromEntries(
            patchMetrics
              .filter(name =>
                selected === 'scene' ? name !== 'perimeterFaceCounts' : name === selected
              )
              .map(name => [name, outputs[name].view])
          )
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const [frameIndex, frame] of scene.frames.entries()) {
          componentCount.write(new Uint32Array([frame.componentCount]));
          converged.write(new Uint32Array([frame.converged]));
          overflow.write(new Uint32Array([frame.overflow]));
          submitGraph(device, compiled, undefined);
          for (const metric of patchMetrics) {
            if (selected === 'scene' ? metric === 'perimeterFaceCounts' : metric !== selected)
              continue;
            const output = outputs[metric];
            const floating = metric === 'areas' || metric === 'perimeters';
            compareRasterParity(
              'GPURasterPatchMetrics',
              `${scene.name}/${selected}/frame-${frameIndex}`,
              scene.reference,
              metric,
              await (floating ? readFloat32 : readUint32)(output.buffer, scene.capacity),
              await decodePackedNumbers(frame.expected[metric]),
              floating ? 'S' : 'E'
            );
          }
        }
      } finally {
        compiled.destroy();
        componentCount.destroy();
        converged.destroy();
        overflow.destroy();
        rig.destroy();
      }
    });
  }
}
