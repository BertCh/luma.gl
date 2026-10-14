// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  ['h3', 'great-circle'],
  ['quadbin', 'linear']
] as const)('GPUCellSetOutline carries %s edges as %s into ring assembly', (family, edgeModel) => {
  const device = createNullWebGPUDevice();
  const assembleRings = vi
    .spyOn(GPUSegmentRingAssembly.prototype, 'getCommandNodes')
    .mockReturnValue([]);
  try {
    const graph = new GPUCommandGraph(device);
    const outline = new GPUCellSetOutline({
      family,
      cells: createTransientView(graph, 'cells', 'uint32x2', 1),
      output: {
        rows: createTransientView(graph, 'rows', 'uint32', 10),
        cells: createTransientView(graph, 'output-cells', 'uint32x2', 10),
        edgeIndices: createTransientView(graph, 'edge-indices', 'uint32', 10),
        endpoints: createTransientView(graph, 'endpoints', 'float32x4', 10),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      },
      rings: {
        output: {
          positions: createTransientView(graph, 'positions', 'float32x2', 11),
          ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
          status: {
            count: createTransientView(graph, 'ring-count', 'uint32', 1),
            overflow: createTransientView(graph, 'ring-overflow', 'uint32', 1)
          }
        }
      }
    });
    expect(outline.edgeModel).toBe(edgeModel);
    expect(outline.outputSpatialContext).toEqual({
      coordinateSpace: 'longitude-latitude',
      metric: 'none',
      edgeModel
    });
    outline.getCommandNodes(graph);
    expect(assembleRings).toHaveBeenCalledOnce();
    expect(assembleRings.mock.contexts[0].spatialContext).toEqual(outline.outputSpatialContext);
  } finally {
    assembleRings.mockRestore();
    device.destroy();
  }
});
