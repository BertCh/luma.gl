// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createIsochronesCellFacilityNodes} from '../../../src/gpu-network/network-isochrones/network-isochrones-passes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUNetworkIsochrones cell minimum folds -0 and skips negative caller costs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two cells, keys (1, 0) and (2, 0). Cell 0 holds nodes 0 (-0, facility 7) and 1 (+0, facility
  // 3): -0 equals +0, so the tie goes to the lower facility 3. Cell 1 holds nodes 2 (-5, facility
  // 1, skipped) and 3 (4, facility 9).
  const nodeCount = 4;
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'cell-minimum'});
  const input = <Format extends 'uint32' | 'uint32x2' | 'float32'>(
    name: string,
    values: Uint32Array | Float32Array,
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const cellMinimum = output('cell-minimum', 2);
  const cellFacilities = output('cell-facilities', 2);
  graph.add(
    createIsochronesCellFacilityNodes(graph, {
      id: 'cells',
      nodeCount,
      nodeCells: input('node-cells', Uint32Array.of(1, 0, 1, 0, 2, 0, 2, 0), 'uint32x2', nodeCount),
      tableCells: input('table-cells', Uint32Array.of(1, 0, 2, 0), 'uint32x2', 2),
      tableCount: input('table-count', Uint32Array.of(2), 'uint32', 1),
      costs: input('costs', Float32Array.of(-0, 0, -5, 4), 'float32', nodeCount),
      assignments: input('assignments', Uint32Array.of(7, 3, 1, 9), 'uint32', nodeCount),
      cellMinimumCosts: cellMinimum.view,
      cellFacilities: cellFacilities.view
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const four = new Uint32Array(Float32Array.of(4).buffer)[0];
  expect(await readUint32(cellMinimum.buffer, 2)).toEqual([0, four]);
  expect(await readUint32(cellFacilities.buffer, 2)).toEqual([3, 9]);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
