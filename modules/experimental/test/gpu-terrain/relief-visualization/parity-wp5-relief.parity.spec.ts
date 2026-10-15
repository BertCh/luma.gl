// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPULocalDominance,
  getGPULocalDominanceParameterValues
} from '../../../src/gpu-terrain/relief-visualization/gpu-local-dominance';
import {
  GPUMultiScaleRelief,
  getGPUMultiScaleReliefParameterValues
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {RELIEF_CASES, EXACT_TOLERANCE, compareRound4Values} from '../parity-wp5-round4-fixtures';

for (const fixture of RELIEF_CASES) {
  it(`Local dominance and multi-scale relief reference parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = fixture.width * fixture.height;
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-relief'});
    const grid = makeGPURasterMetadata({width: fixture.width, height: fixture.height});
    const inputs = [
      createInputBuffer(device, Float32Array.from(fixture.elevation)),
      createInputBuffer(device, Uint32Array.from(fixture.validity))
    ];
    const outputs = {
      dominance: createOutputBuffer(device, count),
      relief: createOutputBuffer(device, count),
      dominanceValidity: createOutputBuffer(device, count),
      reliefValidity: createOutputBuffer(device, count)
    };
    const dominanceParameters = new GPUParameterBuffer(device, {
      id: 'dominance-parameters',
      format: 'float32',
      length: 4,
      values: getGPULocalDominanceParameterValues(fixture)
    });
    const reliefParameters = new GPUParameterBuffer(device, {
      id: 'relief-parameters',
      format: 'float32',
      length: 4,
      values: getGPUMultiScaleReliefParameterValues(fixture)
    });
    try {
      const elevation = {
        id: 'elevation',
        format: 'float32' as const,
        noDataValue: -9999,
        validity: importGraphBuffer(graph, 'validity', inputs[1], 'uint32', count),
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(graph, 'elevation', inputs[0], 'float32', count)
        }
      };
      graph.add(
        new GPULocalDominance({
          ...fixture.geometry,
          grid,
          elevation,
          parameters: dominanceParameters.importToGraph(graph),
          output: {
            dominance: importGraphBuffer(graph, 'dominance', outputs.dominance, 'float32', count),
            outputValidity: importGraphBuffer(
              graph,
              'dominance-validity',
              outputs.dominanceValidity,
              'uint32',
              count
            )
          }
        })
      );
      graph.add(
        new GPUMultiScaleRelief({
          ...fixture.scales,
          grid,
          elevation,
          parameters: reliefParameters.importToGraph(graph),
          output: {
            relief: importGraphBuffer(graph, 'relief', outputs.relief, 'float32', count),
            outputValidity: importGraphBuffer(
              graph,
              'relief-validity',
              outputs.reliefValidity,
              'uint32',
              count
            )
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        for (const [className, metric, validityMetric, reference] of [
          ['GPULocalDominance', 'dominance', 'dominanceValidity', 'rvt.vis.local_dominance'],
          [
            'GPUMultiScaleRelief',
            'relief',
            'reliefValidity',
            'Composite: scipy.ndimage.uniform_filter + Orengo/Petrie 2018 consecutive scale differences; cross-checked rvt.vis.msrm'
          ]
        ] as const) {
          compareRound4Values(
            className,
            fixture.name,
            reference,
            metric,
            await readFloat32(outputs[metric], count),
            fixture.expected[metric],
            fixture.tolerances[metric]
          );
          compareRound4Values(
            className,
            fixture.name,
            reference,
            'validity',
            await readUint32(outputs[validityMetric], count),
            fixture.expectedValidity,
            EXACT_TOLERANCE
          );
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      dominanceParameters.destroy();
      reliefParameters.destroy();
      for (const buffer of [...inputs, ...Object.values(outputs)]) buffer.destroy();
    }
  });
}
