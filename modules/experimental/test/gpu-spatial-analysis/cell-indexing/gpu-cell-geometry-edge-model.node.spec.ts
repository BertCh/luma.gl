// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellGeometry} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-geometry';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  ['h3', 'great-circle'],
  ['s2', 'great-circle'],
  ['a5', 'great-circle'],
  ['quadbin', 'linear'],
  ['quadkey', 'linear'],
  ['geohash', 'linear']
] as const)('GPUCellGeometry declares %s boundary edges as %s', (family, edgeModel) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const geometry = new GPUCellGeometry({
      family,
      cells: createTransientView(graph, 'cells', 'uint32x2', 1),
      output: {boundaries: createTransientView(graph, 'boundaries', 'float32x2', 10)},
      maximumVertexCount: 10
    });
    expect(geometry.edgeModel).toBe(edgeModel);
    expect(geometry.outputSpatialContext).toEqual({
      coordinateSpace: 'longitude-latitude',
      metric: 'none',
      edgeModel
    });
  } finally {
    device.destroy();
  }
});
