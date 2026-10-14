// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUMinimumClearance} from '../../../src/gpu-spatial-analysis/minimum-clearance';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;
function view<Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

function getKernels(kind: 'lines' | 'polygons'): Map<string, string> {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `minimum-clearance-${kind}`});
  const geometry =
    kind === 'lines'
      ? {
          kind: 'lines' as const,
          positions: view(graph, 'float32x2', 12),
          lineOffsets: view(graph, 'uint32', 4)
        }
      : {
          kind: 'polygons' as const,
          positions: view(graph, 'float32x2', 12),
          featureOffsets: view(graph, 'uint32', 3),
          polygonOffsets: view(graph, 'uint32', 3),
          ringOffsets: view(graph, 'uint32', 4)
        };
  const featureCount = kind === 'lines' ? 3 : 2;
  const clearance = new GPUMinimumClearance({
    geometry,
    output: {
      clearances: view(graph, 'float32', featureCount),
      lines: view(graph, 'float32x4', featureCount),
      vertexIds: view(graph, 'uint32', featureCount)
    }
  });
  return getParsedKernelSources(clearance.getCommandNodes(graph));
}

// Library rule 14: the per-feature minimum is a workgroup reduction over each feature's vertex
// range, not per-vertex atomics onto two words per feature.
it('GPUMinimumClearance reduces each feature with one workgroup and no atomics', () => {
  for (const kind of ['lines', 'polygons'] as const) {
    const kernels = getKernels(kind);
    for (const [id, source] of kernels) {
      expect(source, id).not.toMatch(/atomic/);
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
    const reduction = [...kernels].find(([id]) => id.includes('feature-minimum'));
    expect(reduction, kind).toBeDefined();
    expect(reduction![1]).toMatch(/var<workgroup> laneBits/);
    expect(reduction![1]).toMatch(/workgroupBarrier\(\)/);
    expect([...kernels.keys()].some(id => /minimum-distance|minimum-row/.test(id))).toBe(false);
  }
});
