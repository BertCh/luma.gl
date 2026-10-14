// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUPointInPolygonJoin,
  GPUSpatialJoinPrepared
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  EDGE_SLABS_MINIMUM_WORK,
  getDefaultEdgeSlabs,
  getPolygonEdgeLayoutLength
} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {HALF_OPEN_SCENES} from './half-open-fixtures';
import {createRandom, type OraclePolygonFeature} from './spatial-join-oracle';

type Classification = 'inside' | 'outside' | 'boundary-owned' | 'boundary-unowned';
type Vertex = [number, number];

/** Edges of every ring as `[ringIndex, from, to]`, closing edge included. */
function getRingEdges(ring: number[][]): [Vertex, Vertex][] {
  return ring.map((vertex, index) => [vertex as Vertex, ring[(index + 1) % ring.length] as Vertex]);
}

/** Port of the unslabbed `classifyExactly` (integer coordinates, so orientation is exact). */
function classifyWalk(point: Vertex, feature: OraclePolygonFeature): Classification {
  return classifyEdges(
    point,
    feature.map(polygon => polygon.flatMap(ring => ringEdges(ring)))
  );
}

/** Edges of a ring as the walk sees them; rings under 3 vertices are skipped. */
function ringEdges(ring: number[][]): [Vertex, Vertex][] {
  return ring.length < 3 ? [] : getRingEdges(ring);
}

/** Shared per-edge body of the walk and the slab refine (the WGSL has one copy of it). */
function classifyEdges(point: Vertex, edgesPerPart: [Vertex, Vertex][][]): Classification {
  let geometryInside = false;
  let geometryOwned = false;
  let onBoundary = false;
  for (const edges of edgesPerPart) {
    let inside = false;
    let owned = false;
    for (const [from, to] of edges) {
      const [previous, current] = [from, to];
      const sign = Math.sign(
        (current[0] - previous[0]) * (point[1] - previous[1]) -
          (current[1] - previous[1]) * (point[0] - previous[0])
      );
      let side = sign;
      if (sign === 0) {
        if (
          point[0] >= Math.min(previous[0], current[0]) &&
          point[0] <= Math.max(previous[0], current[0]) &&
          point[1] >= Math.min(previous[1], current[1]) &&
          point[1] <= Math.max(previous[1], current[1])
        ) {
          onBoundary = true;
        }
        side = current[0] > previous[0] ? 1 : current[0] < previous[0] ? -1 : 0;
        if (side === 0) {
          side = current[1] < previous[1] ? 1 : current[1] > previous[1] ? -1 : 0;
        }
      }
      if (previous[1] > point[1] !== current[1] > point[1]) {
        if (current[1] > previous[1] && sign > 0) inside = !inside;
        if (current[1] < previous[1] && sign < 0) inside = !inside;
        if ((current[1] > previous[1] && side > 0) || (current[1] < previous[1] && side < 0)) {
          owned = !owned;
        }
      }
    }
    geometryInside = geometryInside || inside;
    geometryOwned = geometryOwned || owned;
  }
  if (onBoundary) return geometryOwned ? 'boundary-owned' : 'boundary-unowned';
  return geometryInside ? 'inside' : 'outside';
}

type PartIndex = {
  usable: [Vertex, Vertex][];
  all: [Vertex, Vertex][];
  slabCount: number;
  origin: number;
  scale: number;
  slabs: [Vertex, Vertex][][];
};

/** CPU model of the polygon-owner slab index (polygon-edge-slabs.ts) for one part. */
function buildPartIndex(polygon: number[][][], minimumVertexCount: number): PartIndex {
  const vertices = polygon.flat();
  // The same edge objects are in `all` and `usable` so the slab filter can test membership.
  const ringEdgeLists = polygon.map(ring => getRingEdges(ring));
  const all = ringEdgeLists.flat();
  const usable = ringEdgeLists.filter((_, ring) => polygon[ring].length >= 3).flat();
  const index: PartIndex = {usable, all, slabCount: 0, origin: 0, scale: 0, slabs: []};
  if (vertices.length < minimumVertexCount || vertices.length === 0) return index;
  const minY = Math.min(...vertices.map(vertex => vertex[1]));
  const maxY = Math.max(...vertices.map(vertex => vertex[1]));
  const wanted = Math.min(Math.floor(vertices.length / 4), 4096);
  const scale = Math.fround(wanted / (maxY - minY));
  if (!(scale > 0 && scale < 1e30)) return index;
  index.slabCount = wanted;
  index.origin = minY;
  index.scale = scale;
  index.slabs = Array.from({length: wanted}, () => []);
  for (const edge of all) {
    const first = slabOf(index, Math.min(edge[0][1], edge[1][1]));
    const last = slabOf(index, Math.max(edge[0][1], edge[1][1]));
    for (let slab = first; slab <= last; slab++) index.slabs[slab].push(edge);
  }
  return index;
}

function slabOf(index: PartIndex, y: number): number {
  if (index.slabCount <= 1) return 0;
  const raw = Math.floor(Math.fround(Math.fround(y - index.origin) * index.scale));
  return Math.min(Math.max(raw, 0), index.slabCount - 1);
}

/** Slab refine: per part, the usable-ring edges of the slab at `point.y`, or all edges if unslabbed. */
function classifySlabs(point: Vertex, parts: PartIndex[]): Classification {
  return classifyEdges(
    point,
    parts.map(part => {
      if (part.slabCount === 0) return part.usable;
      // Shuffle-free but order independent: reverse to prove that order does not matter.
      const edges = part.slabs[slabOf(part, point[1])].filter(edge => part.usable.includes(edge));
      return edges.reverse();
    })
  );
}

function createRandomFeature(random: () => number, partCount: number): OraclePolygonFeature {
  const integer = (maximum: number) => Math.floor(random() * maximum);
  return Array.from({length: partCount}, () => {
    const ringCount = 1 + (random() < 0.3 ? integer(3) : 0);
    return Array.from({length: ringCount}, () => {
      const kind = random();
      const size = kind < 0.08 ? 1 + integer(2) : 3 + integer(random() < 0.4 ? 14 : 5);
      const centerX = integer(14);
      const centerY = integer(14);
      return Array.from({length: size}, () => [
        Math.min(20, centerX + integer(7)),
        // Repeated y values give horizontal edges and vertices on slab boundaries.
        Math.min(20, centerY + integer(7))
      ]);
    });
  });
}

function createGridPoints(): Vertex[] {
  const points: Vertex[] = [];
  for (let y = -1; y <= 21; y += 0.5) {
    for (let x = -1; x <= 21; x += 0.5) points.push([x, y]);
  }
  return points;
}

it('the slab refine equals the full walk per part on adversarial multipolygons', () => {
  const random = createRandom(2026);
  const points = createGridPoints();
  let slabbedParts = 0;
  for (let trial = 0; trial < 60; trial++) {
    const partCount = trial === 0 ? 40 : 1 + Math.floor(random() * 6);
    const feature = createRandomFeature(random, partCount);
    for (const minimumVertexCount of [0, 8]) {
      const parts = feature.map(polygon => buildPartIndex(polygon, minimumVertexCount));
      slabbedParts += parts.filter(part => part.slabCount > 0).length;
      for (const point of points) {
        expect(classifySlabs(point, parts), `trial ${trial} point ${point}`).toBe(
          classifyWalk(point, feature)
        );
      }
    }
  }
  expect(slabbedParts).toBeGreaterThan(250);
});

it('OR across overlapping parts differs from a feature-wide parity, so the parts stay separate', () => {
  const square = [
    [
      [0, 0],
      [4, 0],
      [4, 4],
      [0, 4]
    ]
  ];
  const feature: OraclePolygonFeature = [square, square];
  // Two identical overlapping parts: XOR over all edges would call (2, 2) outside.
  expect(classifyWalk([2, 2], feature)).toBe('inside');
  const parts = feature.map(polygon => buildPartIndex(polygon, 0));
  expect(classifySlabs([2, 2], parts)).toBe('inside');
});

it('rings under three vertices neither bound area nor mark boundary', () => {
  const feature: OraclePolygonFeature = [
    [
      [
        [0, 0],
        [4, 0]
      ],
      [[1, 1]],
      [
        [10, 10],
        [14, 10],
        [14, 14]
      ]
    ]
  ];
  for (const point of [
    [2, 0],
    [1, 1],
    [0, 0]
  ] as Vertex[]) {
    expect(classifyWalk(point, feature)).toBe('outside');
    expect(classifySlabs(point, [buildPartIndex(feature[0], 0)])).toBe('outside');
  }
});

it('the half-open scenes keep their owners through the slab refine', () => {
  for (const scene of Object.values(HALF_OPEN_SCENES)) {
    const parts = scene.features.map(feature => feature.map(polygon => buildPartIndex(polygon, 0)));
    for (const [pointIndex, point] of scene.points.entries()) {
      let owner: number | null = null;
      scene.features.forEach((feature, row) => {
        const result = classifySlabs(point as Vertex, parts[row]);
        if (owner === null && (result === 'inside' || result === 'boundary-owned')) owner = row;
        expect(result).toBe(classifyWalk(point as Vertex, feature));
      });
      expect(owner, `point ${pointIndex}`).toBe(scene.owners[pointIndex]);
    }
  }
});

function createJoin(options: {
  edgeSlabs?: boolean;
  prepared?: GPUSpatialJoinPrepared;
  querySort?: boolean;
  vertexCount?: number;
  featureCount?: number;
  pointCount?: number;
  candidateCapacity?: number;
}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const vertexCount = options.vertexCount ?? 64;
  const featureCount = options.featureCount ?? 4;
  const pointCount = options.pointCount ?? 100;
  const view = <Format extends 'uint32' | 'float32x2'>(name: string, format: Format, n: number) =>
    createTransientView(graph, name, format, n);
  const polygons = {
    kind: 'polygons',
    positions: view('polygon-positions', 'float32x2', vertexCount),
    featureOffsets: view('feature-offsets', 'uint32', featureCount + 1),
    polygonOffsets: view('polygon-offsets', 'uint32', featureCount + 1),
    ringOffsets: view('ring-offsets', 'uint32', featureCount + 1)
  } as const;
  const join = new GPUPointInPolygonJoin({
    id: 'pip',
    points: {kind: 'points', positions: view('points', 'float32x2', pointCount)},
    polygons,
    candidateCapacity: options.candidateCapacity ?? 256,
    edgeSlabs: options.edgeSlabs,
    querySort: options.querySort ?? false,
    output: {
      pointFeatureIds: view('point-feature-ids', 'uint32', pointCount),
      featureCounts: view('feature-counts', 'uint32', featureCount),
      status: {overflow: view('overflow', 'uint32', 1), uncertainCount: view('u', 'uint32', 1)}
    },
    prepared: options.prepared
  });
  return {join, graph, device, polygons};
}

it('edgeSlabs builds slab and layout nodes, eight bindings in the classify kernel, packed candidates', () => {
  const {join, graph, device} = createJoin({edgeSlabs: true});
  expect(join.edgeSlabs).toBe(true);
  const nodes = join.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('pip-edges-slab-params');
  expect(ids).toContain('pip-edges-edge-layout-bits');
  expect(ids).toContain('pip-edges-edge-layout-features');
  // Slabs and layout are built before the probe and the classify.
  expect(ids.indexOf('pip-edges-edge-layout-features')).toBeLessThan(ids.indexOf('pip-probe'));
  expect(ids.indexOf('pip-probe')).toBeLessThan(ids.indexOf('pip-classify'));
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  }
  const classify = sources.find(kernel => kernel.id === 'pip-classify')!;
  expect(getStorageBindingCount(classify.source)).toBe(8);
  expect(classify.source).toContain('fn classifySlabs(');
  // One call site of the exact orientation helper inside the slab loop.
  expect(classify.source.match(/orientSign\(previous, current, point\)/g)).toHaveLength(1);
  expect(classify.source).not.toContain('candidatePoints');
  const probeWrite = sources.find(kernel => kernel.id === 'pip-probe-write')!;
  expect(probeWrite.source).not.toContain('candidatePoints');
  expect(probeWrite.source).toContain('candidatePairsOffset + slot * 4u + 3u');
  const resolve = sources.find(kernel => kernel.id === 'pip-resolve')!;
  expect(resolve.source).toContain('candidatePairsOffset + index * 4u + 1u');
  device.destroy();
});

it("edgeSlabs false keeps today's kernels: no slab nodes, two-word candidates", () => {
  const {join, graph, device} = createJoin({edgeSlabs: false});
  const nodes = join.getCommandNodes(graph);
  expect(nodes.some(node => node.id.includes('slab') || node.id.includes('edge-layout'))).toBe(
    false
  );
  const sources = getGPUCommandNodeKernelSources(nodes);
  const classify = sources.find(kernel => kernel.id === 'pip-classify')!;
  expect(classify.source).toContain('fn classifyExactly(');
  expect(classify.source).not.toContain('edgeSlab');
  expect(getStorageBindingCount(classify.source)).toBe(8);
  const resolve = sources.find(kernel => kernel.id === 'pip-resolve')!;
  expect(resolve.source).toContain('candidatePairsOffset + index * 2u + 1u');
  device.destroy();
});

it('edgeSlabs composes with querySort and the pipeline count does not depend on sizes (rule 12)', () => {
  const count = (vertexCount: number, featureCount: number) => {
    const {join, graph, device} = createJoin({
      edgeSlabs: true,
      querySort: true,
      vertexCount,
      featureCount
    });
    const nodes = join.getCommandNodes(graph);
    const sizes = getGPUCommandNodePipelineKeys(nodes).size;
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    }
    device.destroy();
    return sizes;
  };
  expect(count(128, 8)).toBe(count(256, 16));
});

it('edgeSlabs defaults on for long refine loops and with a prepared handle that carries the index', () => {
  expect(getDefaultEdgeSlabs(1000, EDGE_SLABS_MINIMUM_WORK / 1000, 10)).toBe(true);
  expect(getDefaultEdgeSlabs(1000, EDGE_SLABS_MINIMUM_WORK / 1000 - 1, 10)).toBe(false);
  // Features under 8 vertices on average are never slabbed.
  expect(getDefaultEdgeSlabs(1000, 1e9, 1000)).toBe(false);
  expect(createJoin({}).join.edgeSlabs).toBe(false);
  expect(createJoin({vertexCount: 64, candidateCapacity: 100_000}).join.edgeSlabs).toBe(true);
  expect(
    createJoin({vertexCount: 64, candidateCapacity: 100_000, edgeSlabs: false}).join.edgeSlabs
  ).toBe(false);
});

it('a prepared handle with edgeSlabs builds persistent slabs once and the join reuses them', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32x2'>(name: string, format: Format, n: number) =>
    createTransientView(graph, name, format, n);
  const polygons = {
    kind: 'polygons',
    positions: view('polygon-positions', 'float32x2', 64),
    featureOffsets: view('feature-offsets', 'uint32', 5),
    polygonOffsets: view('polygon-offsets', 'uint32', 5),
    ringOffsets: view('ring-offsets', 'uint32', 5)
  } as const;
  const prepared = new GPUSpatialJoinPrepared({id: 'prep', geometry: polygons, edgeSlabs: true});
  const preparedNodes = prepared.getCommandNodes(graph);
  const preparedIds = preparedNodes.map(node => node.id);
  expect(preparedIds).toContain('prep-edges-slab-params');
  expect(preparedIds).toContain('prep-edges-copy-entries');
  expect(prepared.edgeSlabStorage).toBeDefined();
  // Every node of the handle is conditioned on the rebuild decision (once per rebuild).
  expect(preparedNodes.every(node => node.condition?.id === 'prep-rebuild')).toBe(true);
  const sources = getGPUCommandNodeKernelSources(preparedNodes);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  }
  const join = new GPUPointInPolygonJoin({
    id: 'pip',
    points: {kind: 'points', positions: view('points', 'float32x2', 50)},
    polygons,
    candidateCapacity: 64,
    prepared,
    output: {
      pointFeatureIds: view('point-feature-ids', 'uint32', 50),
      status: {overflow: view('overflow', 'uint32', 1)}
    }
  });
  expect(join.edgeSlabs).toBe(true);
  const joinIds = join.getCommandNodes(graph).map(node => node.id);
  // The join builds neither slabs nor layout of its own.
  expect(joinIds.some(id => id.includes('slab') || id.includes('edge-layout'))).toBe(false);
  expect(joinIds).toContain('pip-classify');
  expect(prepared.edgeSlabStorage!.layout.length).toBe(getPolygonEdgeLayoutLength(64, 4));
  device.destroy();
});

it('edgeSlabs on a prepared handle needs polygon geometry', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPUSpatialJoinPrepared({
        geometry: {kind: 'points', positions: createTransientView(graph, 'p', 'float32x2', 4)},
        edgeSlabs: true
      })
  ).toThrow(/polygon geometry/);
  device.destroy();
});
