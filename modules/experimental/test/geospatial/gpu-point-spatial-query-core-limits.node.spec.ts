// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUPointSpatialQuery,
  type GPUPointSpatialQueryKind
} from '../../src/geospatial/gpu-point-spatial-query';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

// Review A §7.1 suspected 9 storage buffers in the refinement pass of an indexed polygon query
// with source IDs. Every combination of kind, index, source IDs and diagnostics must stay at or
// below the core limit of 8 storage buffers per stage (the query state is a uniform binding).
describe('GPUPointSpatialQuery at the core storage-buffer limit', () => {
  for (const kind of ['bounds', 'radius', 'polygon'] as GPUPointSpatialQueryKind[]) {
    for (const indexed of [false, true]) {
      it(`${kind} ${indexed ? 'indexed' : 'scan'} with source IDs and every diagnostic`, () => {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device, {id: 'point-query-limits'});
        const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
          name: string,
          format: Format,
          length: number
        ) => createTransientView(graph, name, format, length);
        const query = new GPUPointSpatialQuery({
          id: 'query',
          positions: view('positions', 'float32x2', 64),
          sourceIds: view('source-ids', 'uint32', 64),
          kind,
          parameters: view('parameters', 'float32', kind === 'radius' ? 3 : 4),
          index: indexed
            ? {
                gridSize: [4, 4],
                bounds: [0, 0, 1, 1],
                cellOffsets: view('cell-offsets', 'uint32', 17),
                rowIndices: view('row-indices', 'uint32', 64),
                count: view('index-count', 'uint32', 1),
                overflow: view('index-overflow', 'uint32', 1)
              }
            : undefined,
          polygon:
            kind === 'polygon'
              ? {
                  positions: view('polygon-positions', 'float32x2', 8),
                  ringOffsets: view('ring-offsets', 'uint32', 3)
                }
              : undefined,
          output: {
            ids: view('ids', 'uint32', 64),
            status: {
              count: view('count', 'uint32', 1),
              overflow: view('overflow', 'uint32', 1),
              requiredCount: view('required-count', 'uint32', 1)
            },
            intersectedCellCount: view('intersected-cell-count', 'uint32', 1),
            candidateCount: view('candidate-count', 'uint32', 1)
          }
        });
        const kernels = getGPUCommandNodeKernelSources(query.getCommandNodes(graph));
        expect(kernels.length).toBeGreaterThan(0);
        const counts = kernels.map(kernel => getStorageBindingCount(kernel.source));
        for (const [index, kernel] of kernels.entries()) {
          expect(counts[index], kernel.id).toBeLessThanOrEqual(8);
        }
        if (kind === 'polygon' && indexed) {
          // The suspected case is exactly at the limit: no further storage binding fits.
          expect(Math.max(...counts)).toBe(8);
        }
        device.destroy();
      });
    }
  }
});
