// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSegmentRingAssembly,
  type GPUSegmentRingAssemblyProps
} from '../../../src/gpu-spatial-analysis/ring-assembly';
import {
  GPU_RING_CELL_BIAS,
  getRingCellSize,
  quantizeRingVertexCoordinate
} from '../../../src/gpu-spatial-analysis/ring-assembly/ring-vertex-quantizer';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSegmentRingAssemblyProps> = {}
): GPUSegmentRingAssemblyProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `ring-view-${serial++}`, format, length);
  return {
    endpoints: view('float32x4', 16),
    output: {
      ringOffsets: view('uint32', 5),
      positions: view('float32x2', 32),
      ringShells: view('uint32', 4),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    ...overrides
  };
}

it('GPUSegmentRingAssembly declares hash, sort, tracing, scan, placement and ring nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-node'});
  const nodes = new GPUSegmentRingAssembly(createProps(graph)).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(20);
  const single = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-node-single'});
  const fewer = new GPUSegmentRingAssembly(
    createProps(single, {splitTouchingRings: false})
  ).getCommandNodes(single);
  expect(fewer.length).toBeLessThan(nodes.length);
});

it('GPUSegmentRingAssembly validates its properties', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-validate'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `validate-${serial++}`, format, length);
  const props = createProps(graph);
  expect(() => new GPUSegmentRingAssembly({...props, vertexTolerance: 0})).toThrow(
    /vertexTolerance/
  );
  expect(() => new GPUSegmentRingAssembly({...props, groups: view('uint32', 3)})).toThrow(
    /groups length/
  );
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        output: {...props.output, ringShells: view('uint32', 7)}
      })
  ).toThrow(/ring capacity/);
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        output: {...props.output, ringGroups: view('uint32', 4)}
      })
  ).toThrow(/groups input/);
  expect(
    () =>
      new GPUSegmentRingAssembly({
        ...props,
        interiorSide: 'up' as 'left'
      })
  ).toThrow(/interiorSide/);
});

it('GPUSegmentRingAssembly selects cos(latitude) turn comparison with a longitude-latitude context', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-context'});
  const planar = new GPUSegmentRingAssembly(createProps(graph));
  expect(planar.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native'
  });
  const geographic = new GPUSegmentRingAssembly(
    createProps(graph, {
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
    })
  );
  expect(geographic.spatialContext.coordinateSpace).toBe('longitude-latitude');
  for (const spatialContext of [
    {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
    {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
  ] as const) {
    expect(() => new GPUSegmentRingAssembly(createProps(graph, {spatialContext}))).toThrow(
      /does not support/
    );
  }
});

it('ring vertex quantizer separates UTM-scale vertices at the default tolerances', () => {
  // The former i32 floor(v * k) saturated for |v| > 214,748, so every vertex of a UTM scene fell
  // into one cell. Relative to the first vertex, with an extent-adaptive cell, they stay distinct.
  const originX = 500000.25;
  const originY = 4500000.5;
  const extent = 38; // 20 vertices, 2 m apart
  for (const tolerance of [5e-5, 1e-4, 0.5]) {
    const cells = new Set<string>();
    for (let i = 0; i < 20; i++) {
      for (let j = 0; j < 20; j++) {
        const x = quantizeRingVertexCoordinate(originX + i * 2, originX, tolerance, extent);
        const y = quantizeRingVertexCoordinate(originY + j * 2, originY, tolerance, extent);
        expect(x.overflow || y.overflow).toBe(false);
        cells.add(`${x.cell},${y.cell}`);
      }
    }
    expect(cells.size, `tolerance ${tolerance}`).toBe(400);
  }
  expect(quantizeRingVertexCoordinate(originX, originX, 0.5, extent).cell).toBe(GPU_RING_CELL_BIAS);
});

it('ring vertex cells stay in f32-exact range and match within the tolerance for any extent', () => {
  for (const extent of [1, 1e3, 3e5, 2e7, 1e9]) {
    const tolerance = 1e-4;
    const size = getRingCellSize(tolerance, extent);
    expect(size).toBeGreaterThanOrEqual(tolerance);
    expect(extent / size).toBeLessThanOrEqual(2 ** 22);
    // Two vertices within the tolerance fall in the same or adjacent cells.
    const a = quantizeRingVertexCoordinate(extent, 0, tolerance, extent);
    const b = quantizeRingVertexCoordinate(extent - tolerance, 0, tolerance, extent);
    expect(Math.abs(a.cell - b.cell)).toBeLessThanOrEqual(1);
    expect(a.overflow).toBe(false);
  }
});

it('ring vertex quantizer reports non-finite coordinates as overflow', () => {
  expect(quantizeRingVertexCoordinate(Number.NaN, 0, 1e-4, 10).overflow).toBe(true);
  expect(quantizeRingVertexCoordinate(Number.POSITIVE_INFINITY, 0, 1e-4, 10).overflow).toBe(true);
  expect(quantizeRingVertexCoordinate(5, Number.NaN, 1e-4, 10).overflow).toBe(true);
  expect(quantizeRingVertexCoordinate(5, 0, 1e-4, 10).overflow).toBe(false);
});

it('GPUSegmentRingAssembly measures the frame extent before hashing', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'ring-quantize-extent'});
  const ids = new GPUSegmentRingAssembly(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  const extent = ids.findIndex(id => id.endsWith('-quantize-extent'));
  expect(ids.findIndex(id => id.endsWith('-quantize-init'))).toBeLessThan(extent);
  expect(extent).toBeGreaterThanOrEqual(0);
  expect(extent).toBeLessThan(ids.findIndex(id => id.endsWith('-hash')));
});
