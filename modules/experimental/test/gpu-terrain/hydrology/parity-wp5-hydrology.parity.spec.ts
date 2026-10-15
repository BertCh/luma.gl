// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  HYDROLOGY_CASES,
  INDEX_OUTPUTS,
  compareHydrologyValues,
  EXACT_HYDROLOGY_TOLERANCE
} from './parity-wp5-hydrology-fixtures';

for (const fixture of HYDROLOGY_CASES) {
  it(`HAND, hydrologic indices and Strahler reference parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const cellCount = fixture.width * fixture.height;
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-hydrology'});
    const inputs = [
      createInputBuffer(device, Float32Array.from(fixture.elevation)),
      createInputBuffer(device, Uint32Array.from(fixture.directions)),
      createInputBuffer(device, Uint32Array.from(fixture.streams)),
      createInputBuffer(device, Float32Array.from(fixture.accumulation))
    ];
    const outputs = {
      heightAboveDrainage: createOutputBuffer(device, cellCount),
      drainageCells: createOutputBuffer(device, cellCount),
      streamOrder: createOutputBuffer(device, cellCount),
      slope: createOutputBuffer(device, cellCount),
      specificCatchmentArea: createOutputBuffer(device, cellCount),
      wetnessIndex: createOutputBuffer(device, cellCount),
      streamPowerIndex: createOutputBuffer(device, cellCount),
      handConverged: createOutputBuffer(device, 1),
      orderConverged: createOutputBuffer(device, 1)
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8,
      values: getGPUTerrainHydrologicIndicesParameterValues(
        {minimumSlope: fixture.minimumSlope},
        grid
      )
    });
    try {
      const elevation = makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', inputs[0], 'float32', cellCount),
        grid,
        {id: 'elevation'}
      );
      const flowDirections = importGraphBuffer(graph, 'directions', inputs[1], 'uint32', cellCount);
      const streams = importGraphBuffer(graph, 'streams', inputs[2], 'uint32', cellCount);
      const outputView = <Format extends 'float32' | 'uint32'>(
        name: keyof typeof outputs,
        format: Format,
        length = cellCount
      ) => importGraphBuffer(graph, name, outputs[name], format, length);
      graph.add(
        new GPUTerrainHeightAboveDrainage({
          grid,
          elevation,
          flowDirections,
          streams,
          output: {
            heightAboveDrainage: outputView('heightAboveDrainage', 'float32'),
            drainageCells: outputView('drainageCells', 'uint32'),
            status: {converged: outputView('handConverged', 'uint32', 1)}
          }
        })
      );
      graph.add(
        new GPUTerrainStreamOrder({
          grid,
          flowDirections,
          streams,
          output: {
            streamOrder: outputView('streamOrder', 'uint32'),
            status: {converged: outputView('orderConverged', 'uint32', 1)}
          }
        })
      );
      graph.add(
        new GPUTerrainHydrologicIndices({
          grid,
          elevation,
          accumulation: importGraphBuffer(graph, 'accumulation', inputs[3], 'float32', cellCount),
          parameters: parameters.importToGraph(graph),
          output: {
            slope: outputView('slope', 'float32'),
            specificCatchmentArea: outputView('specificCatchmentArea', 'float32'),
            wetnessIndex: outputView('wetnessIndex', 'float32'),
            streamPowerIndex: outputView('streamPowerIndex', 'float32')
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        compareHydrologyValues(
          'GPUTerrainHeightAboveDrainage',
          fixture.name,
          'pysheds.Grid.compute_hand',
          'heightAboveDrainage',
          await readFloat32(outputs.heightAboveDrainage, cellCount),
          fixture.expected.heightAboveDrainage,
          fixture.tolerances.heightAboveDrainage
        );
        compareHydrologyValues(
          'GPUTerrainHeightAboveDrainage',
          fixture.name,
          'pysheds.Grid.compute_hand(return_index=True)',
          'drainageCells',
          await readUint32(outputs.drainageCells, cellCount),
          fixture.expected.drainageCells,
          EXACT_HYDROLOGY_TOLERANCE
        );
        compareHydrologyValues(
          'GPUTerrainStreamOrder',
          fixture.name,
          'Composite reference: Strahler 1957 + networkx.topological_sort/predecessors (pysheds differences recorded)',
          'streamOrder',
          await readUint32(outputs.streamOrder, cellCount),
          fixture.expected.streamOrder,
          EXACT_HYDROLOGY_TOLERANCE
        );
        for (const metric of INDEX_OUTPUTS) {
          compareHydrologyValues(
            'GPUTerrainHydrologicIndices',
            fixture.name,
            'Composite reference: pysheds.Grid.cell_slopes/accumulation + numpy.log; Beven/Kirkby, Moore et al.',
            metric,
            await readFloat32(outputs[metric], cellCount),
            fixture.expected[metric],
            fixture.tolerances[metric]
          );
        }
        for (const [className, status] of [
          ['GPUTerrainHeightAboveDrainage', 'handConverged'],
          ['GPUTerrainStreamOrder', 'orderConverged']
        ] as const) {
          compareHydrologyValues(
            className,
            fixture.name,
            'Finite acyclic supplied D8 graph',
            'converged',
            await readUint32(outputs[status], 1),
            [1],
            EXACT_HYDROLOGY_TOLERANCE
          );
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      parameters.destroy();
      for (const buffer of [...inputs, ...Object.values(outputs)]) buffer.destroy();
    }
  });
}
