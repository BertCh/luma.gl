// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUPolygonRasterizationExtentValues,
  GPU_POLYGON_RASTERIZATION_NO_ZONE,
  GPUPolygonRasterization,
  GPURasterJoin,
  type GPUPolygonRasterizationProps,
  type GPURasterJoinProps
} from '../../../src/gpu-raster/polygon-rasterization';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createPolygons,
  findContainingFeature,
  joinPointsOnCPU,
  NO_ZONE,
  rasterizePolygonsOnCPU
} from './polygon-rasterization-oracle';

function createRasterizationProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUPolygonRasterizationProps> = {},
  sizes: {width?: number; height?: number; polygonCount?: number} = {}
): GPUPolygonRasterizationProps {
  const width = sizes.width ?? 8;
  const height = sizes.height ?? 6;
  const polygonCount = sizes.polygonCount ?? 2;
  return {
    id: 'raster',
    width,
    height,
    extent: createTransientView(graph, 'extent', 'float32', 4),
    polygonPositions: createTransientView(graph, 'positions', 'float32x2', 12),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', polygonCount + 1),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 3),
    crossingCapacity: 64,
    zones: createTransientView(graph, 'zones', 'uint32', width * height),
    overflow: createTransientView(graph, 'overflow', 'uint32', 1),
    ...overrides
  };
}

function createJoinProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURasterJoinProps> = {}
): GPURasterJoinProps {
  return {
    id: 'join',
    width: 8,
    height: 6,
    extent: createTransientView(graph, 'extent', 'float32', 4),
    points: createTransientView(graph, 'points', 'float32x2', 10),
    zones: createTransientView(graph, 'zones', 'uint32', 48),
    zoneCount: 3,
    output: {counts: createTransientView(graph, 'counts', 'uint32', 3)},
    ...overrides
  };
}

it('GPUPolygonRasterization prefixes deterministic node IDs and schedules boundary on request', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUPolygonRasterization(createRasterizationProps(graph));
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.every(id => id.startsWith('raster-'))).toBe(true);
  expect(ids).toContain('raster-edges');
  expect(ids).toContain('raster-fill');
  expect(ids.some(id => id.startsWith('raster-boundary'))).toBe(false);
  expect(contributor.singleSort).toBe(true);
  const repeatGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const repeatIds = new GPUPolygonRasterization(createRasterizationProps(repeatGraph))
    .getCommandNodes(repeatGraph)
    .map(node => node.id);
  expect(repeatIds).toEqual(ids);

  const boundaryGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const boundaryIds = new GPUPolygonRasterization(
    createRasterizationProps(boundaryGraph, {
      boundary: createTransientView(boundaryGraph, 'boundary', 'uint32', 48)
    })
  )
    .getCommandNodes(boundaryGraph)
    .map(node => node.id);
  expect(boundaryIds).toContain('raster-boundary');
});

it('GPUPolygonRasterization switches to two stable sorts when the packed key exceeds 32 bits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 64 * 65536 row-column keys need 22 bits; 1024 polygons plus the sentinel need 11.
  const contributor = new GPUPolygonRasterization(
    createRasterizationProps(
      graph,
      {zones: createTransientView(graph, 'big-zones', 'uint32', 65535 * 64)},
      {width: 65535, height: 64, polygonCount: 1024}
    )
  );
  expect(contributor.singleSort).toBe(false);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => id.startsWith('raster-sort-row-column'))).toBe(true);
  expect(ids.some(id => id.startsWith('raster-sort-polygon'))).toBe(true);
});

it('GPUPolygonRasterization validates dimensions, capacities, and views', () => {
  const create =
    (getOverrides: (graph: GPUCommandGraph) => Partial<GPUPolygonRasterizationProps>) => () => {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      return new GPUPolygonRasterization(createRasterizationProps(graph, getOverrides(graph)));
    };
  expect(create(() => ({width: 0}))).toThrow(/width must be a positive integer/);
  expect(create(() => ({height: 1.5}))).toThrow(/height must be a positive integer/);
  expect(create(() => ({crossingCapacity: 1}))).toThrow(/crossingCapacity/);
  expect(
    create(graph => ({extent: createTransientView(graph, 'short-extent', 'float32', 3)}))
  ).toThrow(/extent must contain 4/);
  expect(
    create(graph => ({zones: createTransientView(graph, 'short-zones', 'uint32', 47)}))
  ).toThrow(/zones must contain 48/);
  expect(
    create(graph => ({boundary: createTransientView(graph, 'short-boundary', 'uint32', 4)}))
  ).toThrow(/boundary must contain 48/);
  expect(
    create(graph => ({overflow: createTransientView(graph, 'long-overflow', 'uint32', 2)}))
  ).toThrow(/overflow must contain 1/);
  expect(
    create(graph => ({featureOffsets: createTransientView(graph, 'one-offset', 'uint32', 1)}))
  ).toThrow(/featureOffsets requires at least one range/);
  expect(
    create(graph => {
      const shared = createTransientView(graph, 'shared', 'uint32', 48);
      return {zones: shared, ringOffsets: shared};
    })
  ).toThrow(/must not share buffers/);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const foreign = new GPUPolygonRasterization(createRasterizationProps(otherGraph));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/belong to the target graph/);
});

it('GPURasterJoin validates outputs and schedules only requested passes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const ids = new GPURasterJoin(createJoinProps(graph)).getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual(['join-lookup', 'join-counts-reset', 'join-aggregate']);

  const sumGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const sumIds = new GPURasterJoin(
    createJoinProps(sumGraph, {
      values: createTransientView(sumGraph, 'values', 'float32', 10),
      output: {sums: createTransientView(sumGraph, 'sums', 'float32', 3)}
    })
  )
    .getCommandNodes(sumGraph)
    .map(node => node.id);
  expect(sumIds).toContain('join-contributions');
  expect(sumIds.some(id => id.startsWith('join-sorted-'))).toBe(true);

  const create = (getOverrides: (graph: GPUCommandGraph) => Partial<GPURasterJoinProps>) => () => {
    const joinGraph = new GPUCommandGraph(createNullWebGPUDevice());
    return new GPURasterJoin(createJoinProps(joinGraph, getOverrides(joinGraph)));
  };
  expect(create(() => ({output: {}}))).toThrow(/at least one output/);
  expect(
    create(joinGraph => ({output: {sums: createTransientView(joinGraph, 's', 'float32', 3)}}))
  ).toThrow(/sums require values/);
  expect(
    create(joinGraph => ({
      output: {boundaryCounts: createTransientView(joinGraph, 'b', 'uint32', 3)}
    }))
  ).toThrow(/boundary outputs require boundary/);
  expect(create(() => ({zoneCount: 0}))).toThrow(/zoneCount/);
  expect(create(joinGraph => ({zones: createTransientView(joinGraph, 'z', 'uint32', 47)}))).toThrow(
    /zones must contain width \* height/
  );
  expect(
    create(joinGraph => ({output: {pointZones: createTransientView(joinGraph, 'pz', 'uint32', 9)}}))
  ).toThrow(/pointZones must contain 10/);
});

it('getGPUPolygonRasterizationExtentValues packs the extent layout', () => {
  expect(Array.from(getGPUPolygonRasterizationExtentValues(1, 2, 0.5, 0.25))).toEqual([
    1, 2, 0.5, 0.25
  ]);
  expect(() => getGPUPolygonRasterizationExtentValues(0, 0, 1, 1, new Float32Array(3))).toThrow();
  expect(GPU_POLYGON_RASTERIZATION_NO_ZONE).toBe(NO_ZONE);
});

it('polygon rasterization oracle honors holes, overlaps, and the half-open center rule', () => {
  const polygons = createPolygons([
    // Feature 0: 4x4 square with a 2x2 hole.
    [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ],
        [
          [1, 1],
          [3, 1],
          [3, 3],
          [1, 3]
        ]
      ]
    ],
    // Feature 1 overlaps feature 0's right column; its edges lie on cell centers.
    [
      [
        [
          [3.5, 0.5],
          [6.5, 0.5],
          [6.5, 2.5],
          [3.5, 2.5]
        ]
      ]
    ]
  ]);
  const result = rasterizePolygonsOnCPU(polygons, {width: 8, height: 4, extent: [0, 0, 1, 1]});
  const row = (index: number) => Array.from(result.zones.slice(index * 8, index * 8 + 8));
  const N = NO_ZONE;
  // Centers at y = 0.5 and 1.5 are inside feature 1 ([0.5, 2.5)); x in [3.5, 6.5) -> columns 3..5.
  expect(row(0)).toEqual([0, 0, 0, 0, 1, 1, N, N]);
  expect(row(1)).toEqual([0, N, N, 0, 1, 1, N, N]);
  expect(row(2)).toEqual([0, N, N, 0, N, N, N, N]);
  expect(row(3)).toEqual([0, 0, 0, 0, N, N, N, N]);
  expect(findContainingFeature(polygons, 2, 2)).toBe(NO_ZONE);
  expect(findContainingFeature(polygons, 3.7, 1)).toBe(0);
  // 4 vertical edges per ring cover 4, 4, 2, 2 and 2, 2 rows.
  expect(result.crossingCount).toBe(16);

  const join = joinPointsOnCPU({
    raster: {width: 8, height: 4, extent: [0, 0, 1, 1]},
    zones: result.zones,
    boundary: result.boundary,
    zoneCount: 2,
    points: Float32Array.from([0.5, 0.5, 4.5, 0.5, 2, 2, -1, 0, 4.2, 1.2]),
    values: Float32Array.from([1, 2, 3, 4, Number.NaN])
  });
  expect(Array.from(join.pointZones)).toEqual([0, 1, N, N, 1]);
  expect(join.counts).toEqual([1, 2]);
  expect(join.sums).toEqual([1, 2]);
  expect(join.outsideCount).toBe(1);
});
