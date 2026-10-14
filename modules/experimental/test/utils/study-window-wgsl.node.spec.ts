// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getStudyWindowBinding,
  getStudyWindowPackedLayout,
  getStudyWindowWGSL,
  getStudyWindowWGSLOptions,
  getStudyWindowSampleCellWGSL,
  STUDY_WINDOW_BOUNDARY_CELL_BIT
} from '../../src/utils/study-window-wgsl';
import {getPolygonEdgeSlabNodes} from '../../src/utils/polygon-edge-slabs';
import {EXACT_GEOMETRY_WGSL} from '../../src/utils/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../src/utils/exact-orientation-wgsl';
import {IS_FINITE_BITS_WGSL} from '../../src/utils/wgsl-common';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from './gpu-contributor-test-utils';

it('preserves the coordinator packed word layout and derives sizes from views', () => {
  const layout = getStudyWindowPackedLayout({
    vertexCount: 12,
    slabOffsetCount: 5,
    slabEntryCount: 60,
    sampleCellCapacity: 20
  });
  expect(layout).toEqual({
    summary: 0,
    edges: 8,
    slabParams: 56,
    slabOffsets: 62,
    slabEntries: 67,
    sampleCellCount: 187,
    sampleCells: 188,
    length: 208
  });
  expect(STUDY_WINDOW_BOUNDARY_CELL_BIT).toBe(0x80000000);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {slabs} = getPolygonEdgeSlabNodes(graph, {
    id: 'window-slabs',
    operation: 'StudyWindowTest',
    owner: 'polygon',
    positions: createTransientView(graph, 'positions', 'float32x2', 12),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    entryCapacity: 48
  });
  const views = {
    edges: createTransientView(graph, 'edges', 'float32', 48),
    slabs,
    sampleCells: createTransientView(graph, 'cells', 'uint32', 20),
    sampleGrid: [5, 4] as const
  };
  expect(getStudyWindowWGSLOptions(views, 'quartic')).toEqual({
    layout,
    vertexCount: 12,
    entryCapacity: 48,
    sampleGrid: [5, 4],
    kernel: 'quartic'
  });
  const packed = createTransientView(graph, 'packed', 'uint32', layout.length);
  expect(getStudyWindowBinding({packed})).toEqual({
    name: 'studyWindow',
    view: packed,
    type: 'u32',
    access: 'read'
  });
});

for (const kernel of [undefined, 'gaussian', 'quartic', 'epanechnikov'] as const) {
  it(`study-window WGSL parses, audits and has one exact path (${kernel ?? 'no kernel'})`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const layout = getStudyWindowPackedLayout({
      vertexCount: 12,
      slabOffsetCount: 5,
      slabEntryCount: 60,
      sampleCellCapacity: 20
    });
    const declarations = getStudyWindowWGSL({
      layout,
      vertexCount: 12,
      entryCapacity: 48,
      kernel,
      sampleGrid: [5, 4]
    });
    for (const name of [
      'studyWindowSummaryValue',
      'studyWindowEdge',
      'isStudyWindowUsable',
      'isInsideStudyWindow',
      'getStudyWindowBoundaryDistance',
      'getStudyWindowCircleFraction',
      'getStudyWindowSampleCell'
    ]) {
      expect(declarations).toContain(`fn ${name}(`);
    }
    expect(declarations.match(/exactProbeCrossing\(/g)).toHaveLength(1);
    // All slabbed/walked loops and the boundary fan's full walk reject unused NaN edges.
    for (const match of declarations.matchAll(/let edge = studyWindowEdge\([^;]+;/g)) {
      expect(declarations.slice(match.index! + match[0].length).trimStart()).toMatch(
        /^if \(!studyWindowFinitePoint\(edge.xy\) \|\| !studyWindowFinitePoint\(edge.zw\)\) \{ continue; \}/
      );
    }
    expect(declarations.includes('fn getStudyWindowKernelMass(')).toBe(Boolean(kernel));
    expect(declarations).toContain('(studyWindowOffset + 56u)');
    expect(declarations).toContain('(studyWindowOffset + 62u)');
    expect(declarations).toContain('(studyWindowOffset + 67u)');
    expect(declarations).toContain(getStudyWindowSampleCellWGSL([5, 4], 'studyWindowSummaryValue'));
    expect(declarations.match(/edgeSlabIndex != max/g)).toHaveLength(kernel ? 2 : 1);
    const query = createWGSLKernelNode(graph, {
      id: 'window-query',
      operation: 'StudyWindowTest',
      invocationCount: 1,
      bindings: [
        getStudyWindowBinding({
          packed: createTransientView(graph, 'packed', 'uint32', layout.length)
        }),
        {
          name: 'result',
          view: createTransientView(graph, 'result', 'float32', 1),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `${IS_FINITE_BITS_WGSL}${EXACT_ORIENTATION_WGSL}${EXACT_GEOMETRY_WGSL}${declarations}`,
      body: `let point = vec2f(400.0, 500.0);
        result[resultOffset] = select(0.0, 1.0, isInsideStudyWindow(point)) +
          getStudyWindowBoundaryDistance(point, 1000.0) + getStudyWindowCircleFraction(point, 800.0) +
          getStudyWindowSampleCell(0x80000001u).x ${kernel ? '+ getStudyWindowKernelMass(point, 500.0)' : ''};`
    });
    const source = getParsedKernelSources([query]).get('window-query')!;
    expect(source.match(/\borientSign\(/g)).toHaveLength(2); // declaration + one call
    expect(source.match(/\bexactOrientSigns\(/g)).toHaveLength(3); // declaration + unused segment helper + crossing helper
    expect(getStorageBindingCount(source)).toBe(2);
    expect(auditWGSLModule(source)).toEqual([]);
  });
}
