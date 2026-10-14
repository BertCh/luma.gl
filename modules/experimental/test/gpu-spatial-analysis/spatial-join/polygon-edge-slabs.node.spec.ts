// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getPolygonEdgeSlabBindings,
  getPolygonEdgeSlabLoopWGSL,
  getPolygonEdgeSlabNodes,
  getPolygonEdgeSlabQueryWGSL,
  POLYGON_EDGE_SLAB_MAXIMUM_SLABS,
  type PolygonEdgeSlabOwner
} from '../../../src/gpu-spatial-analysis/spatial-join/polygon-edge-slabs';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createLayout(graph: GPUCommandGraph, vertexCount: number, ringCount = 1) {
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `edge-slab-${serial++}`, format, length);
  return {
    positions: view('float32x2', vertexCount),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', ringCount + 1),
    view
  };
}

for (const owner of ['feature', 'polygon'] as PolygonEdgeSlabOwner[]) {
  for (const withGate of [false, true]) {
    it(`polygon edge slabs build and query WGSL parse (${owner}, gate ${withGate})`, () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `edge-slabs-${serial++}`});
      const layout = createLayout(graph, 64, 2);
      const counts = layout.view('uint32', 3);
      const {nodes, slabs} = getPolygonEdgeSlabNodes(graph, {
        id: 'slabs',
        operation: 'Test',
        owner,
        positions: layout.positions,
        featureOffsets: owner === 'feature' ? layout.featureOffsets : undefined,
        polygonOffsets: layout.polygonOffsets,
        ringOffsets: layout.ringOffsets,
        gate: withGate ? {counts, minimum: 16} : undefined
      });
      const ids = nodes.map(node => node.id);
      expect(new Set(ids).size).toBe(ids.length);
      expect(ids).toContain('slabs-slab-params');
      expect(ids).toContain('slabs-slab-edges-count');
      expect(ids).toContain('slabs-slab-edges-fill');
      expect(slabs.ownerCount).toBe(1);
      expect(slabs.vertexCount).toBe(64);
      expect(slabs.slabEntries.length).toBe(64 + slabs.entryCapacity);
      for (const source of getParsedKernelSources(nodes).values()) {
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
      }

      // Dummy query kernel: three slab views plus positions and an output.
      const out = layout.view('uint32', 1);
      const bindings = getPolygonEdgeSlabBindings(slabs);
      expect(bindings).toHaveLength(3);
      const query = createWGSLKernelNode(graph, {
        id: 'query',
        operation: 'Test',
        bindings: [
          {name: 'positions', view: layout.positions, type: 'f32', access: 'read'},
          ...bindings,
          {name: 'out', view: out, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${getPolygonEdgeSlabQueryWGSL(slabs)}
fn getVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}
fn countHits(owner: u32, y: f32) -> u32 {
  var hits = 0u;
  ${getPolygonEdgeSlabLoopWGSL({
    owner: 'owner',
    lowY: 'y',
    highY: 'y',
    getVertex: 'getVertex',
    body: 'if (min(a.y, b.y) <= y && y <= max(a.y, b.y)) { hits += edgeEnd - edgeStart; }'
  })}
  return hits;
}`,
        body: 'out[outOffset] = countHits(0u, 0.5);'
      });
      const sources = getParsedKernelSources([query]);
      const source = sources.get('query')!;
      expect(getStorageBindingCount(source)).toBe(5);
      // One call site of the body.
      expect(source.split('hits += edgeEnd').length - 1).toBe(1);
      expect(getGPUCommandNodeKernelSources([query])).toHaveLength(1);
    });
  }
}

it('polygon edge slabs validate the owner layout', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'edge-slabs-validation'});
  const layout = createLayout(graph, 8);
  expect(() =>
    getPolygonEdgeSlabNodes(graph, {
      id: 'slabs',
      operation: 'Test',
      owner: 'feature',
      positions: layout.positions,
      polygonOffsets: layout.polygonOffsets,
      ringOffsets: layout.ringOffsets
    })
  ).toThrow(/featureOffsets/);
});

// CPU model of the slab assignment, mirroring the f32 expression of `edgeSlabOf`.
type Point = [number, number];

function slabOf(y: number, count: number, origin: number, scale: number): number {
  if (count <= 1) {
    return 0;
  }
  const value = Math.floor(Math.fround(Math.fround(Math.fround(y) - origin) * scale));
  return Math.min(Math.max(value, 0), count - 1);
}

type SlabModel = {slabCount: number; origin: number; scale: number; slabs: number[][]};

function buildModel(ring: Point[]): SlabModel {
  const vertices = ring.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point);
  const finite = vertices.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  let minY = 1e30;
  let maxY = -1e30;
  for (const [x, y] of vertices) {
    if (Number.isFinite(x) && Number.isFinite(y)) {
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  let slabCount = 0;
  let scale = 0;
  if (finite && vertices.length >= 8) {
    let travel = 0;
    for (let vertex = 0; vertex < vertices.length; vertex++) {
      travel = Math.fround(
        travel +
          Math.abs(
            Math.fround(
              vertices[vertex][1] - vertices[(vertex + vertices.length - 1) % vertices.length][1]
            )
          )
      );
    }
    const wanted = Math.min(
      Math.floor(vertices.length / 4),
      POLYGON_EDGE_SLAB_MAXIMUM_SLABS,
      travel > 0
        ? Math.max(
            1,
            Math.floor(
              Math.fround(vertices.length * Math.fround(Math.fround(maxY - minY) / travel))
            )
          )
        : POLYGON_EDGE_SLAB_MAXIMUM_SLABS
    );
    const candidate = Math.fround(wanted / Math.fround(maxY - minY));
    if (candidate > 0 && candidate < 1e30) {
      slabCount = wanted;
      scale = candidate;
    }
  }
  const slabs: number[][] = Array.from({length: slabCount}, () => []);
  if (slabCount > 0) {
    vertices.forEach((a, index) => {
      const b = vertices[(index + 1) % vertices.length];
      const first = slabOf(Math.min(a[1], b[1]), slabCount, minY, scale);
      const last = slabOf(Math.max(a[1], b[1]), slabCount, minY, scale);
      for (let slab = first; slab <= last; slab++) {
        slabs[slab].push(index);
      }
    });
  }
  return {slabCount, origin: minY, scale, slabs};
}

function checkInvariant(ring: Point[], model: SlabModel, heights: number[]): void {
  const vertices = ring.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point);
  for (const y of heights.map(Math.fround)) {
    const slab = slabOf(y, model.slabCount, model.origin, model.scale);
    vertices.forEach((a, index) => {
      const b = vertices[(index + 1) % vertices.length];
      if (Math.min(a[1], b[1]) <= y && y <= Math.max(a[1], b[1])) {
        expect(model.slabs[slab], `edge ${index} at y=${y}`).toContain(index);
      }
    });
  }
}

function createCircle(count: number, radius = 1): Point[] {
  return Array.from({length: count}, (_, index) => {
    const angle = (2 * Math.PI * index) / count;
    return [radius * Math.cos(angle), radius * Math.sin(angle)] as Point;
  });
}

function getHeights(ring: Point[], model: SlabModel): number[] {
  const heights = ring.map(([, y]) => y);
  // Exactly on the slab boundaries and one ulp either side.
  for (let slab = 0; slab <= model.slabCount; slab++) {
    const boundary = Math.fround(model.origin + slab / model.scale);
    heights.push(
      boundary,
      Math.fround(boundary * (1 + 2 ** -23)),
      Math.fround(boundary * (1 - 2 ** -23))
    );
  }
  return heights;
}

it('polygon edge slab model lists every edge whose closed y range contains the query height', () => {
  const rings: Record<string, Point[]> = {
    circle: createCircle(64),
    // Horizontal edges at slab boundaries.
    staircase: Array.from({length: 40}, (_, index): Point => {
      const step = Math.floor(index / 2);
      return index % 2 === 0 ? [step, step] : [step + 1, step];
    }),
    // Every vertex shares one of a few heights, so edges are horizontal or vertical.
    grid: [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [3, 1],
      [3, 2],
      [3, 3],
      [2, 3],
      [1, 3],
      [0, 3],
      [0, 2],
      [0, 1]
    ],
    // Huge and thin: a long sliver.
    thin: [
      [0, 0],
      [1e6, 1e-3],
      [2e6, 0],
      [3e6, 1e-3],
      [4e6, 0],
      [4e6, 1e-3],
      [2e6, 2e-3],
      [0, 1e-3]
    ],
    // Large offset and a tiny range: rounding stress for the f32 expression.
    offset: Array.from({length: 32}, (_, index): Point => [index, 1e6 + (index % 5) * 0.0625])
  };
  for (const [name, ring] of Object.entries(rings)) {
    const model = buildModel(ring);
    expect(model.slabCount, name).toBeGreaterThan(0);
    checkInvariant(ring, model, getHeights(ring, model));
  }
});

it('polygon edge slab model leaves tiny, flat and non-finite owners unslabbed', () => {
  expect(buildModel(createCircle(6)).slabCount).toBe(0);
  // Zero height.
  expect(buildModel(Array.from({length: 16}, (_, index): Point => [index, 5])).slabCount).toBe(0);
  const nan = createCircle(32);
  nan[7] = [Number.NaN, 0.25];
  expect(buildModel(nan).slabCount).toBe(0);
  const infinite = createCircle(32);
  infinite[3] = [0, Number.POSITIVE_INFINITY];
  expect(buildModel(infinite).slabCount).toBe(0);
  // Single slab: an owner of 8 vertices has 2 slabs; one of 4..7 never slabs. Check 2 slabs.
  expect(buildModel(createCircle(8)).slabCount).toBe(2);
  // Cap on the slab count.
  expect(buildModel(createCircle(4 * (POLYGON_EDGE_SLAB_MAXIMUM_SLABS + 10))).slabCount).toBe(
    POLYGON_EDGE_SLAB_MAXIMUM_SLABS
  );
});

it('GPUCellCover slab kernels parse and stay within eight storage buffers', () => {
  for (const family of ['quadbin', 'h3'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `edge-slabs-cover-${family}`});
    const layout = createLayout(graph, 64);
    const view = layout.view;
    const output = {
      featureIds: view('uint32', 8),
      cells: createTransientView(graph, `edge-slab-${serial++}`, 'uint32x2', 8),
      core: view('uint32', 8),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1),
        invalidCount: view('uint32', 1),
        uncertainCount: view('uint32', 1)
      }
    };
    const cover = new GPUCellCover({
      family,
      resolution: 5,
      positions: layout.positions,
      featureOffsets: layout.featureOffsets,
      polygonOffsets: layout.polygonOffsets,
      ringOffsets: layout.ringOffsets,
      candidateCapacity: 64,
      edgeSlabs: true,
      output
    });
    const sources = getParsedKernelSources(cover.getCommandNodes(graph));
    expect(sources.size).toBeGreaterThan(8);
    for (const source of sources.values()) {
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    }
  }
});

it('fits the slab entries and unslabbed edges together at the maximum geometry size', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const vertexCount = 2 ** 22;
  const layout = createLayout(graph, vertexCount);
  const {slabs} = getPolygonEdgeSlabNodes(graph, {
    id: 'maximum-slabs',
    operation: 'Test',
    owner: 'polygon',
    ...layout
  });
  expect(slabs.entryCapacity).toBe(3 * vertexCount);
  expect(slabs.slabEntries.length * 8).toBe(graph.device.limits.maxStorageBufferBindingSize);
});

it('jagged 50k-vertex polygons fit the entry budget instead of reverting to full walks', () => {
  // Same high-frequency radial variation as the A/B scene, with deterministic alternating noise.
  const ring: Point[] = Array.from({length: 50_000}, (_, vertex) => {
    const angle = (vertex / 50_000) * Math.PI * 2;
    const radius = 1 + 0.25 * Math.sin(17 * angle) + (vertex % 2 ? 0.05 : -0.05);
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  });
  const model = buildModel(ring);
  const entryCount = model.slabs.reduce((total, edges) => total + edges.length, 0);
  expect(model.slabCount).toBeGreaterThan(8);
  expect(model.slabCount).toBeLessThan(4096);
  expect(entryCount).toBeLessThan(4 * ring.length);
  expect(Math.max(...model.slabs.map(edges => edges.length))).toBeLessThan(ring.length / 4);
  checkInvariant(ring, model, [-1, -0.5, 0, 0.5, 1]);
  const pipelines = [50_000, 100_000].map(vertexCount => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const {nodes} = getPolygonEdgeSlabNodes(graph, {
      id: 'adaptive',
      operation: 'Test',
      owner: 'polygon',
      ...createLayout(graph, vertexCount)
    });
    const sources = getParsedKernelSources(nodes);
    expect(sources.get('adaptive-slab-params')).toContain('verticalTravel');
    return getGPUCommandNodePipelineKeys(nodes).size;
  });
  expect(pipelines[0]).toBe(pipelines[1]);
});
