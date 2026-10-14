// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPUBufferSurface,
  GPUPolygonOverlay,
  getGPUBufferSurfaceMaximumBoundaryRowCount,
  getGPUOffsetCurveRowsPerVertex
} from '../../../src/gpu-spatial-analysis';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {POLYGON_BUFFER_FIXTURES} from './polygon-buffer-fixtures';

type FillRule = 'even-odd' | 'positive' | 'negative' | 'nonzero';

/**
 * CPU replay of the winding membership in `GPUPolygonOverlay`: the half-open crossing rule of the
 * parity test with the side decided by orientation (an upward edge counts +1 when the probe is
 * strictly left of it, a downward edge -1 when strictly right).
 */
function getWinding(x: number, y: number, ring: readonly number[]): number {
  let winding = 0;
  const count = ring.length / 2;
  for (let row = 0, previous = count - 1; row < count; previous = row++) {
    const ax = ring[previous * 2];
    const ay = ring[previous * 2 + 1];
    const bx = ring[row * 2];
    const by = ring[row * 2 + 1];
    if (ay > y !== by > y) {
      const side = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
      if (by > ay) {
        if (side > 0) winding += 1;
      } else if (side < 0) {
        winding -= 1;
      }
    }
  }
  return winding;
}

/** Parity of the same crossings, as the even-odd line counts them. */
function getParity(x: number, y: number, ring: readonly number[]): boolean {
  let inside = false;
  const count = ring.length / 2;
  for (let row = 0, previous = count - 1; row < count; previous = row++) {
    const ax = ring[previous * 2];
    const ay = ring[previous * 2 + 1];
    const bx = ring[row * 2];
    const by = ring[row * 2 + 1];
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) {
      inside = !inside;
    }
  }
  return inside;
}

function isInside(rule: FillRule, x: number, y: number, rings: readonly number[][]): boolean {
  if (rule === 'even-odd') {
    return rings.reduce((parity, ring) => parity !== getParity(x, y, ring), false);
  }
  const winding = rings.reduce((sum, ring) => sum + getWinding(x, y, ring), 0);
  return rule === 'positive' ? winding > 0 : rule === 'negative' ? winding < 0 : winding !== 0;
}

/** Replay of the buffer's ring preparation: drop consecutive duplicates and a repeated closing vertex. */
function prepareRing(coordinates: readonly number[], dropClosingVertex: boolean): number[] {
  const rows: (readonly [number, number])[] = [];
  for (let row = 0; row < coordinates.length / 2; row++) {
    rows.push([coordinates[row * 2], coordinates[row * 2 + 1]]);
  }
  const same = (a: readonly [number, number], b: readonly [number, number]) =>
    a[0] === b[0] && a[1] === b[1];
  const kept = rows.filter((point, row) => {
    if (row > 0 && same(rows[row - 1], point)) return false;
    if (dropClosingVertex && row === rows.length - 1 && row > 0 && same(rows[0], point)) {
      return false;
    }
    return true;
  });
  return kept.flat();
}

it('winding replay matches shapely.buffer where even-odd parity does not', () => {
  // Oracle: shapely buffer membership grids in polygon-buffer-fixtures.ts (fixtures.py).
  const mismatchesByRule: Record<string, Record<FillRule, number>> = {};
  for (const fixture of POLYGON_BUFFER_FIXTURES) {
    if (fixture.rawRings.length === 0) {
      continue;
    }
    const [x0, y0, x1, y1] = fixture.bounds;
    const columns = Math.round((x1 - x0) / fixture.step);
    const rows = Math.round((y1 - y0) / fixture.step);
    const counts: Record<FillRule, number> = {'even-odd': 0, positive: 0, negative: 0, nonzero: 0};
    for (let row = 0; row <= rows; row++) {
      for (let column = 0; column <= columns; column++) {
        const expected = fixture.rows[row][column];
        if (expected === '?') continue;
        const x = x0 + column * fixture.step + 0.013;
        const y = y0 + row * fixture.step + 0.007;
        for (const rule of Object.keys(counts) as FillRule[]) {
          if (isInside(rule, x, y, fixture.rawRings) !== (expected === '1')) counts[rule]++;
        }
      }
    }
    mismatchesByRule[fixture.name] = counts;
    // Lines emit clockwise outlines (negative winding), polygons counter-clockwise (positive).
    expect(counts[fixture.kind === 'line' ? 'negative' : 'positive'], fixture.name).toBe(0);
    // The raw offsets cover regions twice, which parity reads as holes.
    if (fixture.name !== 'thin-neck polygon d=-0.7 mitre') {
      expect(counts['even-odd'], fixture.name).toBeGreaterThan(0);
    }
  }
  // 'nonzero' keeps the inverted pinch loop of a negative buffer; 'positive' discards it.
  expect(mismatchesByRule['thin-neck polygon d=-0.7 mitre'].nonzero).toBeGreaterThan(0);
});

it('winding and parity agree on parity for simple rings', () => {
  const ring = [0, 0, 4, 0, 4, 4, 0, 4];
  for (const [x, y] of [
    [1.3, 1.7],
    [5.2, 1.1],
    [-0.4, 3.9]
  ]) {
    expect(getWinding(x, y, ring) % 2 !== 0).toBe(getParity(x, y, ring));
  }
});

it('ring preparation drops closing and consecutive duplicate vertices', () => {
  expect(prepareRing([0, 0, 2, 0, 2, 2, 0, 2, 0, 0], true)).toEqual([0, 0, 2, 0, 2, 2, 0, 2]);
  expect(prepareRing([0, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 0], true)).toEqual([0, 0, 2, 0, 2, 2, 0, 2]);
  // Lines keep their endpoints even when the line is closed.
  expect(prepareRing([0, 0, 2, 0, 2, 2, 0, 0], false)).toEqual([0, 0, 2, 0, 2, 2, 0, 0]);
});

it('GPUPolygonOverlay validates fillRule and defaults to even-odd', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'fill-rule'});
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `fill-rule-${serial++}`, format, length);
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2)
  });
  const scalar = () => view('uint32', 1);
  const status = () => ({
    count: scalar(),
    requiredCount: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar()
  });
  const makeOutput = () => {
    const sourceIds = view('uint32', 4);
    return {
      geometry: {
        kind: 'polygons' as const,
        positions: view('float32x2', 16),
        featureOffsets: view('uint32', 5),
        polygonOffsets: view('uint32', 5),
        ringOffsets: view('uint32', 5),
        sourceIds
      },
      sourceIds,
      status: status(),
      boundary: {
        endpoints: view('float32x4', 16),
        operandIds: view('uint32', 16),
        sourceFeatureIds: view('uint32', 16),
        status: status()
      }
    };
  };
  const props = {
    left: polygons(),
    operation: 'dissolve' as const,
    capacity: {intersections: 16, nodedSegments: 32},
    vertexTolerance: 1e-5
  };
  expect(new GPUPolygonOverlay({...props, output: makeOutput()}).fillRule).toBe('even-odd');
  expect(
    new GPUPolygonOverlay({...props, fillRule: 'positive', output: makeOutput()}).fillRule
  ).toBe('positive');
  expect(
    () =>
      new GPUPolygonOverlay({
        ...props,
        fillRule: 'winding' as unknown as 'positive',
        output: makeOutput()
      })
  ).toThrow(/fillRule/);
  const nodeIds = new GPUPolygonOverlay({...props, fillRule: 'nonzero', output: makeOutput()})
    .getCommandNodes(graph)
    .map(node => node.id);
  // Unclamped ring requirements are published with source IDs.
  expect(nodeIds).toContain('polygon-overlay-source-ids');
  expect(nodeIds).not.toContain('polygon-overlay-publish-required');
  expect(nodeIds.at(-1)).toBe('polygon-overlay-publish');
});

function createPolygonBufferNodes(device: Device, vertexCount: number = 5) {
  const graph = new GPUCommandGraph(device, {id: 'buffer-prepare'});
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `buffer-prepare-${serial++}`, format, length);
  const scalar = () => view('uint32', 1);
  const status = () => ({
    count: scalar(),
    requiredCount: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar()
  });
  const sourceIds = view('uint32', 8);
  const surface = new GPUBufferSurface({
    geometry: {
      kind: 'polygons',
      positions: view('float32x2', vertexCount),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2)
    },
    parameters: view('float32', 4),
    joinStyle: 'mitre',
    vertexTolerance: 1e-5,
    capacity: {intersections: 64, nodedSegments: 128},
    output: {
      geometry: {
        kind: 'polygons',
        positions: view('float32x2', 64),
        featureOffsets: view('uint32', 9),
        polygonOffsets: view('uint32', 9),
        ringOffsets: view('uint32', 9),
        sourceIds
      },
      status: status(),
      boundary: {
        endpoints: view('float32x4', 64),
        operandIds: view('uint32', 64),
        sourceFeatureIds: view('uint32', 64),
        status: status()
      }
    }
  });
  return surface.getCommandNodes(graph);
}

it('GPUBufferSurface prepares rings and lines before offsetting', () => {
  const ids = createPolygonBufferNodes(createNullWebGPUDevice()).map(node => node.id);
  const compactIndex = ids.indexOf('buffer-surface-ring-compact-flags');
  expect(compactIndex).toBeGreaterThanOrEqual(0);
  expect(ids).toContain('buffer-surface-ring-compact-offsets');
  expect(ids).toContain('buffer-surface-ring-compact-scatter');
  expect(compactIndex).toBeLessThan(ids.indexOf('buffer-surface-stroke-left-offset-generate'));
});

it('GPUBufferSurface derives its boundary maximum from the graph device', () => {
  const device = createNullWebGPUDevice();
  const rowsPerVertex = getGPUOffsetCurveRowsPerVertex('mitre', 8, 'buffer');
  const getVertexCount = (boundaryRows: number) =>
    Math.floor((boundaryRows - 64 - 3) / (1 + 2 * rowsPerVertex));
  expect(getGPUBufferSurfaceMaximumBoundaryRowCount(device)).toBe(2 ** 20);
  const coreVertices = getVertexCount(2 ** 20);
  expect(() => createPolygonBufferNodes(device, coreVertices)).not.toThrow();
  expect(() => createPolygonBufferNodes(device, coreVertices + 1)).toThrow(
    'buffer-surface boundary rows must be at most 1048512'
  );
  Object.assign(device.limits, {
    maxStorageBufferBindingSize: 2 ** 30,
    maxBufferSize: 2 ** 30
  });
  expect(getGPUBufferSurfaceMaximumBoundaryRowCount(device)).toBe(2 ** 23);
  expect(() => createPolygonBufferNodes(device, coreVertices + 1)).not.toThrow();
  const largeVertices = getVertexCount(2 ** 23);
  expect(() => createPolygonBufferNodes(device, largeVertices)).not.toThrow();
  expect(() => createPolygonBufferNodes(device, largeVertices + 1)).toThrow(
    'buffer-surface boundary rows must be at most 8388544'
  );
});

it('GPUBufferSurface buffers polygons as the polygon and its ring strokes (D-02)', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'buffer-pieces'});
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `buffer-pieces-${serial++}`, format, length);
  const status = () => ({
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  const sourceIds = view('uint32', 8);
  const surface = new GPUBufferSurface({
    geometry: {
      kind: 'polygons',
      positions: view('float32x2', 9),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 3)
    },
    parameters: view('float32', 4),
    joinStyle: 'round',
    vertexTolerance: 1e-5,
    capacity: {intersections: 256, nodedSegments: 512},
    output: {
      geometry: {
        kind: 'polygons',
        positions: view('float32x2', 64),
        featureOffsets: view('uint32', 9),
        polygonOffsets: view('uint32', 9),
        ringOffsets: view('uint32', 9),
        sourceIds
      },
      status: status(),
      boundary: {
        endpoints: view('float32x4', 64),
        operandIds: view('uint32', 64),
        sourceFeatureIds: view('uint32', 64),
        status: status()
      }
    }
  });
  const nodes = surface.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  for (const id of [
    'stroke-collapse',
    'stroke-left-offset-generate',
    'stroke-right-offset-generate',
    'raw-layout',
    'raw-pieces'
  ]) {
    expect(ids, id).toContain(`buffer-surface-${id}`);
  }
  // No offset of the ring itself: the polygon enters the dissolve as reversed rings.
  expect(ids).not.toContain('buffer-surface-offset-generate');
  const kernels = getGPUCommandNodeKernelSources(nodes);
  for (const kernel of kernels) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    // Core-device limit: at most eight storage buffers per stage (raw-pieces uses all eight).
    expect(kernel.source.match(/var<storage/g)?.length ?? 0, kernel.id).toBeLessThanOrEqual(8);
  }
  // The dissolve keeps regions covered by reversed rings: the negative winding rule, applied where
  // the face labelling turns polygon windings into operand coverage.
  const coverage = kernels.find(kernel => kernel.id === 'buffer-surface-repair-group-coverage');
  expect(coverage?.source).toMatch(/winding < 0/);
});
