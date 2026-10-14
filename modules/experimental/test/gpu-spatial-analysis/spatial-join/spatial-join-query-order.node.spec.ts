// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {getNearestNeighborNodes} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-nodes';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getDefaultQuerySort,
  getHilbertQueryOrderNodes,
  getSpatialJoinProbeNodes,
  HILBERT_INDEX_WGSL,
  QUERY_SORT_MINIMUM_QUERIES,
  QUANTIZE_AXIS_WGSL
} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

/** TS port of the WGSL `getHilbertIndex` (classic xy2d on the 65536 x 65536 grid). */
function getHilbertIndexModel(cellX: number, cellY: number): number {
  let x = cellX;
  let y = cellY;
  let key = 0;
  for (let s = 32768; s > 0; s >>>= 1) {
    const rx = (x & s) !== 0 ? 1 : 0;
    const ry = (y & s) !== 0 ? 1 : 0;
    key = (key + s * s * ((3 * rx) ^ ry)) >>> 0;
    if (ry === 0) {
      if (rx === 1) {
        x = 65535 - x;
        y = 65535 - y;
      }
      [x, y] = [y, x];
    }
  }
  return key;
}

/** Inverse of xy2d (d2xy) for the round-trip check. */
function getHilbertCellModel(key: number): [number, number] {
  let t = key;
  let x = 0;
  let y = 0;
  for (let s = 1; s < 65536; s *= 2) {
    const rx = 1 & (t >>> 1);
    const ry = 1 & (t ^ rx);
    if (ry === 0) {
      if (rx === 1) {
        x = s - 1 - x;
        y = s - 1 - y;
      }
      [x, y] = [y, x];
    }
    x += s * rx;
    y += s * ry;
    t = Math.floor(t / 4);
  }
  return [x, y];
}

/** TS port of the WGSL `quantizeAxis`. */
function quantizeAxisModel(value: number, minimum: number, maximum: number): number {
  const extent = Math.fround(maximum - minimum);
  if (!(extent > 0) || !Number.isFinite(extent)) {
    return 0;
  }
  const normalized = Math.min(Math.max(Math.fround(Math.fround(value - minimum) / extent), 0), 1);
  return Math.min(Math.floor(Math.fround(normalized * 65535) + 0.5), 65535);
}

/** CPU model of the order: stable sort of rows by Hilbert key, non-finite points last. */
function getQueryOrderModel(
  points: [number, number][],
  rootMinimum: [number, number],
  rootMaximum: [number, number]
): number[] {
  const keys = points.map(([x, y]) =>
    Number.isFinite(x) && Number.isFinite(y)
      ? getHilbertIndexModel(
          quantizeAxisModel(x, rootMinimum[0], rootMaximum[0]),
          quantizeAxisModel(y, rootMinimum[1], rootMaximum[1])
        )
      : 0xffffffff
  );
  return points
    .map((_, row) => row)
    .sort((left, right) => keys[left] - keys[right] || left - right);
}

it('the Hilbert model matches the curve: origin, adjacency and round trip', () => {
  expect(getHilbertIndexModel(0, 0)).toBe(0);
  // First 2 x 2 block of the order-16 curve: (0,0) -> (1,0) -> (1,1) -> (0,1) up to the
  // orientation of the 16-bit recursion, so consecutive keys are always grid neighbors.
  for (let key = 0; key < 4096; key++) {
    const [x, y] = getHilbertCellModel(key);
    expect(getHilbertIndexModel(x, y)).toBe(key);
    const [nextX, nextY] = getHilbertCellModel(key + 1);
    expect(Math.abs(nextX - x) + Math.abs(nextY - y)).toBe(1);
  }
  for (const [x, y] of [
    [65535, 65535],
    [65535, 0],
    [0, 65535],
    [12345, 54321],
    [32768, 32767]
  ]) {
    expect(getHilbertCellModel(getHilbertIndexModel(x, y))).toEqual([x, y]);
  }
});

it('the order model is a stable permutation: clamped outside points, non-finite last, ties by row', () => {
  const points: [number, number][] = [
    [Number.NaN, 0],
    [5, 5],
    [5, 5],
    [-100, 5],
    [Number.POSITIVE_INFINITY, 1],
    [0, 0],
    [10, 10],
    [1000, 1000]
  ];
  const order = getQueryOrderModel(points, [0, 0], [10, 10]);
  expect([...order].sort((left, right) => left - right)).toEqual(points.map((_, row) => row));
  // (0, 0) has key 0, the duplicate (5, 5) rows keep row order, non-finite points are last.
  expect(order[0]).toBe(5);
  expect(order.indexOf(1)).toBeLessThan(order.indexOf(2));
  expect(order.slice(-2)).toEqual([0, 4]);
  // (1000, 1000) clamps to the same cell as (10, 10); ties keep row order.
  expect(order.indexOf(6)).toBeLessThan(order.indexOf(7));
  expect(order.indexOf(7) - order.indexOf(6)).toBe(1);
});

it('the query order kernel embeds the shared Hilbert WGSL and parses', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32x2'>(name: string, format: Format, n: number) =>
    createTransientView(graph, name, format, n);
  const {nodes, order} = getHilbertQueryOrderNodes(graph, {
    id: 'order',
    operation: 'GPUPointInPolygonJoin',
    points: view('points', 'float32x2', 1000),
    bvh: {
      nodeMinima: view('node-min', 'float32x2', 7),
      nodeMaxima: view('node-max', 'float32x2', 7)
    }
  });
  expect(order.length).toBe(1000);
  expect(order.format).toBe('uint32');
  const sources = getGPUCommandNodeKernelSources(nodes);
  const keyKernel = sources.find(kernel => kernel.id === 'order-keys');
  expect(keyKernel).toBeDefined();
  expect(keyKernel!.source).toContain(HILBERT_INDEX_WGSL);
  expect(keyKernel!.source).toContain(QUANTIZE_AXIS_WGSL);
  // The key kernel reads node 0 of the BVH: root bounds, no reduction pass of its own.
  expect(keyKernel!.source).toContain('nodeMinima[nodeMinimaOffset]');
  expect(keyKernel!.source).toContain('0xffffffffu');
  expect(getStorageBindingCount(keyKernel!.source)).toBeLessThanOrEqual(8);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
  }
  // Key kernel first, then the radix sort passes, in node order.
  expect(nodes[0].id).toBe('order-keys');
  expect(nodes.length).toBeGreaterThan(2);
  device.destroy();
});

/** Builds probe nodes over a fake BVH. */
function buildProbeNodes(options: {chunks: number[]; querySort: boolean; radius?: boolean}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    n: number
  ) => createTransientView(graph, `probe-${name}`, format, n);
  const chunkViews = options.chunks.map((length, index) =>
    view(`points-${index}`, 'float32x2', length)
  );
  const points =
    chunkViews.length === 1 ? chunkViews[0] : createVectorView('points', 'float32x2', chunkViews);
  const nodes = getSpatialJoinProbeNodes(graph, {
    id: 'probe',
    operation: 'GPUPointInPolygonJoin',
    points,
    bvh: {
      nodeMinima: view('node-min', 'float32x2', 7),
      nodeMaxima: view('node-max', 'float32x2', 7),
      leafIds: view('leaf-ids', 'uint32', 4),
      internalNodeCount: 3
    },
    featureCount: 4,
    candidateCapacity: 64,
    state: view('state', 'uint32', 4),
    candidatePairs: view('pairs', 'uint32x2' as never, 64) as never,
    candidatePoints: view('candidate-points', 'float32x2', 64),
    pointRowScale: 1,
    pointRowOffset: 0,
    radius: options.radius ? view('radius', 'float32', 1) : undefined,
    querySort: options.querySort
  });
  return {nodes, device};
}

it('probe passes read the order per sorted chunk and stay within 8 bindings', () => {
  const {nodes, device} = buildProbeNodes({chunks: [300, 200, 1], querySort: true});
  const sources = getGPUCommandNodeKernelSources(nodes);
  // Chunks of fewer than two points are not sorted.
  expect(nodes.filter(node => node.id.endsWith('-order-keys'))).toHaveLength(2);
  const counts = sources.filter(kernel => /probe-\d$/.test(kernel.id));
  const writes = sources.filter(kernel => /probe-\d-write$/.test(kernel.id));
  expect(counts).toHaveLength(3);
  expect(writes).toHaveLength(3);
  expect(counts.map(kernel => kernel.source.includes('queryOrder['))).toEqual([true, true, false]);
  expect(writes.map(kernel => kernel.source.includes('queryOrder['))).toEqual([true, true, false]);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  }
  // Results stay addressed by the original row: counts, slots and offsets use `row`.
  const ordered = counts[0].source;
  expect(ordered).toContain('let row = CHUNK_FIRST_ROW + localRow;');
  expect(ordered).toContain('pointCounts[pointCountsOffset + row] = found;');
  device.destroy();
});

it('probe passes without querySort are unchanged by the feature', () => {
  const {nodes, device} = buildProbeNodes({chunks: [300, 200], querySort: false});
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(nodes.some(node => node.id.includes('order'))).toBe(false);
  expect(sources.some(kernel => kernel.source.includes('queryOrder'))).toBe(false);
  device.destroy();
});

it('with a radius the write pass keeps row order so it stays within 8 bindings', () => {
  const {nodes, device} = buildProbeNodes({chunks: [300], querySort: true, radius: true});
  const sources = getGPUCommandNodeKernelSources(nodes);
  const count = sources.find(kernel => kernel.id === 'probe')!;
  const write = sources.find(kernel => kernel.id === 'probe-write')!;
  expect(count.source).toContain('queryOrder[');
  expect(write.source).not.toContain('queryOrder');
  expect(getStorageBindingCount(count.source)).toBe(7);
  expect(getStorageBindingCount(write.source)).toBe(8);
  device.destroy();
});

type NeighborOptions = {
  queryCount: number;
  querySort: boolean;
  scratchChunkQueryCount?: number;
  featureKind?: 'segments' | 'polygons';
  footPoints?: boolean;
  chunks?: number[];
  queryKind?: 'lines' | 'polygons';
  filters?: boolean;
  sphereRadius?: number;
};

function buildNeighborNodes(options: NeighborOptions) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const k = 4;
  const {queryCount} = options;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `n-${name}`, format, length);
  const features =
    options.featureKind === 'polygons'
      ? ({
          kind: 'polygons',
          positions: view('feature-positions', 'float32x2', 400),
          featureOffsets: view('feature-offsets', 'uint32', 101),
          polygonOffsets: view('polygon-offsets', 'uint32', 101),
          ringOffsets: view('ring-offsets', 'uint32', 101)
        } as const)
      : ({
          kind: 'segments',
          starts: view('starts', 'float32x2', 100),
          ends: view('ends', 'float32x2', 100)
        } as const);
  const chunkViews = (options.chunks ?? [queryCount]).map((length, index) =>
    view(`points-${index}`, 'float32x2', length)
  );
  const points =
    chunkViews.length === 1 ? chunkViews[0] : createVectorView('points', 'float32x2', chunkViews);
  const nodes = getNearestNeighborNodes(graph, {
    id: 'n',
    operation: 'GPUNearestFeatureJoin',
    queries: options.queryKind
      ? {
          kind: 'geometry',
          geometry:
            options.queryKind === 'lines'
              ? {
                  kind: 'lines',
                  positions: view('query-positions', 'float32x2', queryCount * 4),
                  lineOffsets: view('query-offsets', 'uint32', queryCount + 1)
                }
              : {
                  kind: 'polygons',
                  positions: view('query-positions', 'float32x2', queryCount * 8),
                  featureOffsets: view('query-offsets', 'uint32', queryCount + 1),
                  polygonOffsets: view('query-polygons', 'uint32', queryCount * 2 + 1),
                  ringOffsets: view('query-ring-offsets', 'uint32', queryCount * 3 + 1)
                }
        }
      : {kind: 'points', points},
    queryCount,
    features,
    featureCount: 100,
    k,
    capacity: k,
    leafCapacity: 128,
    spatialSort: false,
    neighborIds: view('ids', 'uint32', queryCount * k),
    neighborCounts: view('counts', 'uint32', queryCount),
    neighborDistances: view('distances', 'float32', queryCount * k),
    ...(options.footPoints
      ? {
          neighborFootPoints: view('foot', 'float32x2', queryCount * k),
          neighborQueryPoints: view('query-points', 'float32x2', queryCount * k),
          neighborSegmentIndices: view('segments', 'uint32', queryCount * k)
        }
      : {}),
    overflow: view('overflow', 'uint32', 1),
    scratchChunkQueryCount: options.scratchChunkQueryCount,
    sphereRadius: options.sphereRadius,
    ...(options.filters
      ? {
          exclusive: true,
          queryIds: view('query-ids', 'uint32', queryCount),
          featureIds: view('feature-ids', 'uint32', 100),
          onAttribute: {
            left: view('query-keys', 'uint32', queryCount),
            right: view('feature-keys', 'uint32', 100)
          }
        }
      : {}),
    querySort: options.querySort
  }).nodes;
  return {nodes, device};
}

it('nearest traversals read the order, stay within 8 bindings and parse (polygon features)', () => {
  for (const scratchChunkQueryCount of [undefined, 128]) {
    const {nodes, device} = buildNeighborNodes({
      queryCount: 300,
      querySort: true,
      featureKind: 'polygons',
      footPoints: true,
      scratchChunkQueryCount
    });
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const kernel of sources) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    }
    const ordered = sources.filter(kernel => kernel.source.includes('queryOrder['));
    // Unchunked: only the traversal reads the order. Chunked: traversal, both column passes and
    // the foot-point pass map scratch positions through it.
    const expectedIds =
      scratchChunkQueryCount === undefined
        ? ['n-traverse']
        : ['n-scratch-0-traverse', 'n-scratch-0-columns', 'n-scratch-0-foot-columns'];
    for (const id of expectedIds) {
      expect(ordered.map(kernel => kernel.id)).toContain(id);
    }
    expect(nodes.findIndex(node => node.id === 'n-order-keys')).toBeGreaterThanOrEqual(0);
    expect(nodes.findIndex(node => node.id === 'n-order-keys')).toBeLessThan(
      nodes.findIndex(node => node.id.endsWith('-traverse'))
    );
    device.destroy();
  }
});

it('unchunked nearest keeps the original row for topk, columns and filter addressing', () => {
  const {nodes, device} = buildNeighborNodes({queryCount: 300, querySort: true});
  const traverse = getGPUCommandNodeKernelSources(nodes).find(kernel => kernel.id === 'n-traverse');
  expect(traverse!.source).toContain('let localQuery = queryOrder[queryOrderOffset + index];');
  expect(traverse!.source).toContain('let header = topkOffset + row * STRIDE;');
  device.destroy();
});

it('nearest without querySort has no order nodes or bindings', () => {
  const {nodes, device} = buildNeighborNodes({queryCount: 300, querySort: false});
  expect(nodes.some(node => node.id.includes('order'))).toBe(false);
  expect(
    getGPUCommandNodeKernelSources(nodes).some(kernel => kernel.source.includes('queryOrder'))
  ).toBe(false);
  device.destroy();
});

it('sorted nearest shares pipelines across scratch chunks (rule 12)', () => {
  const countDistinct = (queryCount: number) => {
    const {nodes, device} = buildNeighborNodes({
      queryCount,
      querySort: true,
      footPoints: true,
      scratchChunkQueryCount: 128
    });
    const chunkCount = nodes.filter(node => node.id.endsWith('-advance-cursor')).length + 1;
    const keys = getGPUCommandNodePipelineKeys(nodes);
    device.destroy();
    return {chunkCount, distinct: keys.size};
  };
  // Differ by whole scratch chunks with the same remainder, as in nearest-scratch-chunking.
  const two = countDistinct(128 + 5);
  const three = countDistinct(2 * 128 + 5);
  const four = countDistinct(3 * 128 + 5);
  expect([two.chunkCount, three.chunkCount, four.chunkCount]).toEqual([2, 3, 4]);
  expect(three.distinct).toBe(two.distinct);
  expect(four.distinct).toBe(two.distinct);
});

it('sorted nearest builds one order per point chunk', () => {
  const {nodes, device} = buildNeighborNodes({
    queryCount: 500,
    querySort: true,
    chunks: [300, 200]
  });
  expect(nodes.filter(node => node.id.endsWith('-order-keys'))).toHaveLength(2);
  device.destroy();
});

it('querySort defaults on only for many queries and a deep enough tree', () => {
  expect(getDefaultQuerySort(QUERY_SORT_MINIMUM_QUERIES, 256)).toBe(true);
  expect(getDefaultQuerySort(QUERY_SORT_MINIMUM_QUERIES - 1, 256)).toBe(false);
  expect(getDefaultQuerySort(QUERY_SORT_MINIMUM_QUERIES, 255)).toBe(false);

  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32x2'>(name: string, format: Format, n: number) =>
    createTransientView(graph, name, format, n);
  const createJoin = (queryCount: number, querySort?: boolean) =>
    new GPUNearestFeatureJoin({
      id: `join-${queryCount}-${querySort}`,
      queries: {
        kind: 'points',
        positions: view(`q-${queryCount}-${querySort}`, 'float32x2', queryCount)
      },
      features: {
        kind: 'segments',
        starts: view(`s-${queryCount}-${querySort}`, 'float32x2', 300),
        ends: view(`e-${queryCount}-${querySort}`, 'float32x2', 300)
      },
      querySort,
      output: {
        neighborIds: view(`i-${queryCount}-${querySort}`, 'uint32', queryCount),
        neighborCounts: view(`c-${queryCount}-${querySort}`, 'uint32', queryCount),
        status: {overflow: view(`o-${queryCount}-${querySort}`, 'uint32', 1)}
      }
    });
  expect(createJoin(5000).querySort).toBe(true);
  expect(createJoin(100).querySort).toBe(false);
  expect(createJoin(100, true).querySort).toBe(true);
  expect(createJoin(5000, false).querySort).toBe(false);

  const createPip = (pointCount: number, querySort?: boolean) =>
    new GPUPointInPolygonJoin({
      id: `pip-${pointCount}-${querySort}`,
      points: {
        kind: 'points',
        positions: view(`pp-${pointCount}-${querySort}`, 'float32x2', pointCount)
      },
      polygons: {
        kind: 'polygons',
        positions: view(`pv-${pointCount}-${querySort}`, 'float32x2', 20),
        featureOffsets: view(`pf-${pointCount}-${querySort}`, 'uint32', 301),
        polygonOffsets: view(`pg-${pointCount}-${querySort}`, 'uint32', 301),
        ringOffsets: view(`pr-${pointCount}-${querySort}`, 'uint32', 6)
      },
      candidateCapacity: 16,
      querySort,
      output: {
        pointFeatureIds: view(`po-${pointCount}-${querySort}`, 'uint32', pointCount),
        status: {overflow: view(`ps-${pointCount}-${querySort}`, 'uint32', 1)}
      }
    });
  expect(createPip(5000).querySort).toBe(true);
  expect(createPip(100).querySort).toBe(false);
  expect(createPip(100, true).querySort).toBe(true);
  device.destroy();
});

it('GPUPointInPolygonJoin with querySort puts the order nodes before the probe', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32x2'>(name: string, format: Format, n: number) =>
    createTransientView(graph, name, format, n);
  const nodes = new GPUPointInPolygonJoin({
    id: 'pip',
    points: {kind: 'points', positions: view('points', 'float32x2', 5000)},
    polygons: {
      kind: 'polygons',
      positions: view('polygon-positions', 'float32x2', 20),
      featureOffsets: view('feature-offsets', 'uint32', 5),
      polygonOffsets: view('polygon-offsets', 'uint32', 5),
      ringOffsets: view('ring-offsets', 'uint32', 6)
    },
    candidateCapacity: 16,
    querySort: true,
    output: {
      pointFeatureIds: view('point-feature-ids', 'uint32', 5000),
      status: {overflow: view('overflow', 'uint32', 1)}
    }
  }).getCommandNodes(graph);
  const orderIndex = nodes.findIndex(node => node.id === 'pip-probe-order-keys');
  const probeIndex = nodes.findIndex(node => node.id === 'pip-probe');
  expect(orderIndex).toBeGreaterThan(-1);
  expect(orderIndex).toBeLessThan(probeIndex);
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  }
  device.destroy();
});

for (const queryKind of ['lines', 'polygons'] as const) {
  for (const scratchChunkQueryCount of [undefined, 128]) {
    it(`sorted ${queryKind} queries preserve row addressing and core limits (scratch ${scratchChunkQueryCount})`, () => {
      for (const sphereRadius of [undefined, 6371008.8]) {
        const {nodes, device} = buildNeighborNodes({
          queryKind,
          queryCount: 300,
          querySort: true,
          featureKind: 'polygons',
          footPoints: true,
          filters: true,
          scratchChunkQueryCount,
          sphereRadius
        });
        const sources = getGPUCommandNodeKernelSources(nodes);
        for (const kernel of sources) {
          expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
          expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
          expect(auditWGSLModule(kernel.source), kernel.id).toEqual([]);
        }
        const nodeIndex = (suffix: string) => nodes.findIndex(node => node.id === `n-${suffix}`);
        const traversal = sources.find(kernel => kernel.id.endsWith('-traverse'))!;
        expect(nodeIndex('query-bounds')).toBeGreaterThan(-1);
        expect(nodeIndex('query-centers')).toBeGreaterThan(nodeIndex('query-bounds'));
        expect(nodeIndex('order-keys')).toBeGreaterThan(nodeIndex('query-centers'));
        expect(nodes.findIndex(node => node.id === traversal.id)).toBeGreaterThan(
          Math.max(...nodes.map((node, index) => (node.id.startsWith('n-order-') ? index : -1)))
        );
        expect(traversal.source).toContain('let row = CHUNK_FIRST_ROW + localQuery;');
        // Candidate filters use the original row even when the scratch is in sorted order.
        expect(traversal.source).toContain('row * 2u');
        if (queryKind === 'polygons') {
          expect(nodeIndex('query-metadata')).toBeGreaterThan(nodeIndex('query-rings'));
          expect(nodeIndex('query-bounds')).toBeGreaterThan(nodeIndex('query-metadata'));
          expect(traversal.source).not.toContain('qRingOffsets');
          expect(traversal.source).toContain('qFeatureRingsOffset + 600u + r');
          expect(getStorageBindingCount(traversal.source)).toBe(8);
        }
        const foot = sources.find(kernel => kernel.id.endsWith('-foot-points'))!;
        const columns = sources.filter(kernel => kernel.id.endsWith('-columns'));
        if (scratchChunkQueryCount) {
          expect(traversal.source).toContain('queryOrder[queryOrderOffset + position]');
          expect(traversal.source).toContain('let header = topkOffset + index * STRIDE;');
          expect(foot.source).toContain('queryOrder[queryOrderOffset + position]');
          expect(foot.source).toContain('pairResult(localQuery,');
          for (const column of columns) {
            expect(column.source).toContain('queryOrder[queryOrderOffset + position]');
            expect(column.source).toContain('let outputIndex = row * CAPACITY + slot;');
          }
        } else {
          expect(traversal.source).toContain('queryOrder[queryOrderOffset + index]');
          expect(traversal.source).toContain('let header = topkOffset + row * STRIDE;');
          expect(foot.source).not.toContain('queryOrder');
          for (const column of columns) expect(column.source).not.toContain('queryOrder');
        }
        device.destroy();
      }
    });
  }

  it(`sorted ${queryKind} queries keep the pipeline count at N and N+1 scratch chunks`, () => {
    const counts = [133, 261, 389].map(queryCount => {
      const {nodes, device} = buildNeighborNodes({
        queryKind,
        queryCount,
        querySort: true,
        featureKind: 'polygons',
        footPoints: true,
        filters: true,
        scratchChunkQueryCount: 128
      });
      const result = {
        chunks: nodes.filter(node => node.id.endsWith('-advance-cursor')).length + 1,
        pipelines: getGPUCommandNodePipelineKeys(nodes).size
      };
      device.destroy();
      return result;
    });
    expect(counts.map(count => count.chunks)).toEqual([2, 3, 4]);
    expect(new Set(counts.map(count => count.pipelines)).size).toBe(1);
  });

  it(`${queryKind} queries skip all sorting work when disabled or fewer than two rows`, () => {
    for (const [queryCount, querySort] of [
      [300, false],
      [0, true],
      [1, true]
    ] as const) {
      const {nodes, device} = buildNeighborNodes({queryKind, queryCount, querySort});
      expect(nodes.some(node => /query-(metadata|bounds|centers)|order/.test(node.id))).toBe(false);
      device.destroy();
    }
  });

  it(`public ${queryKind} querySort honors the default and explicit settings`, () => {
    for (const [queryCount, requestedSort, expectedSort] of [
      // Line and polygon queries default off: the geometry sort A/B measured it slower.
      [4096, undefined, false],
      [4095, undefined, false],
      [4096, false, false],
      [3, true, true]
    ] as const) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'uint32' | 'float32x2'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      const positions = view('positions', 'float32x2', queryCount * 4);
      const join = new GPUNearestFeatureJoin({
        id: 'geometry-order',
        queries:
          queryKind === 'lines'
            ? {kind: 'lines', positions, lineOffsets: view('lines', 'uint32', queryCount + 1)}
            : {
                kind: 'polygons',
                positions,
                featureOffsets: view('features', 'uint32', queryCount + 1),
                polygonOffsets: view('polygons', 'uint32', queryCount + 1),
                ringOffsets: view('rings', 'uint32', queryCount + 1)
              },
        features: {kind: 'points', positions: view('feature-positions', 'float32x2', 256)},
        querySort: requestedSort,
        output: {
          neighborIds: view('ids', 'uint32', queryCount),
          neighborCounts: view('counts', 'uint32', queryCount),
          status: {overflow: view('overflow', 'uint32', 1)}
        }
      });
      expect(join.querySort).toBe(expectedSort);
      expect(join.getCommandNodes(graph).some(node => node.id.endsWith('-order-keys'))).toBe(
        expectedSort
      );
      device.destroy();
    }
  });
}
