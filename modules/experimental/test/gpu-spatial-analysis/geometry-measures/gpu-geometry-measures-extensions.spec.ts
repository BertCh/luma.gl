// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUGeometryMeasures,
  type GPUGeometryMeasuresProps
} from '../../../src/gpu-spatial-analysis/geometry-measures';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createFlatGeometry, roundFeatures, type NestedFeatures} from './geometry-measures-oracle';

type Readback = {
  lengths: number[];
  areas: number[];
  signedAreas: number[];
  centroids: number[];
  bounds: number[];
  vertexCounts: number[];
  extremes: number[];
};

/** Runs one measures graph over nested features and reads every column of the feature output. */
async function measure(
  device: Device,
  features: NestedFeatures,
  options: Pick<GPUGeometryMeasuresProps, 'geometryType' | 'coordinateSystem'>,
  group?: {groupIds: number[]; groupCount: number}
): Promise<{features: Readback; groupCentroids?: number[]; groupVertexCounts?: number[]}> {
  const flat = createFlatGeometry(features);
  const count = features.length;
  const isPolygon = options.geometryType === 'polygons';
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'measures-extensions'});
  const lengths = track(createOutputBuffer(device, count));
  const areas = track(createOutputBuffer(device, count));
  const signedAreas = track(createOutputBuffer(device, count));
  const centroids = track(createOutputBuffer(device, 2 * count));
  const bounds = track(createOutputBuffer(device, 4 * count));
  const vertexCounts = track(createOutputBuffer(device, count));
  const extremes = track(createOutputBuffer(device, 4 * count));
  const groupCentroids = track(createOutputBuffer(device, 2 * (group?.groupCount ?? 1)));
  const groupVertexCounts = track(createOutputBuffer(device, group?.groupCount ?? 1));
  const view = (name: string, buffer: Buffer, format: 'float32', length: number) =>
    importGraphBuffer(graph, name, buffer, format, length);
  graph.add(
    new GPUGeometryMeasures({
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, flat.positions)),
        'float32x2',
        flat.positions.length / 2
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        track(createInputBuffer(device, flat.ringOffsets)),
        'uint32',
        flat.ringOffsets.length
      ),
      featureRingOffsets: importGraphBuffer(
        graph,
        'feature-ring-offsets',
        track(createInputBuffer(device, flat.featureRingOffsets)),
        'uint32',
        flat.featureRingOffsets.length
      ),
      ...options,
      output: {
        lengths: view('lengths', lengths, 'float32', count),
        ...(isPolygon
          ? {
              areas: view('areas', areas, 'float32', count),
              signedAreas: view('signed-areas', signedAreas, 'float32', count)
            }
          : {}),
        centroids: importGraphBuffer(graph, 'centroids', centroids, 'float32x2', count),
        bounds: importGraphBuffer(graph, 'bounds', bounds, 'float32x4', count),
        vertexCounts: importGraphBuffer(graph, 'vertex-counts', vertexCounts, 'uint32', count),
        extremeVertices: importGraphBuffer(graph, 'extremes', extremes, 'uint32x4', count)
      },
      ...(group
        ? {
            groupIds: importGraphBuffer(
              graph,
              'group-ids',
              track(createInputBuffer(device, new Uint32Array(group.groupIds))),
              'uint32',
              count
            ),
            groupCount: group.groupCount,
            groupOutput: {
              centroids: importGraphBuffer(
                graph,
                'group-centroids',
                groupCentroids,
                'float32x2',
                group.groupCount
              ),
              vertexCounts: importGraphBuffer(
                graph,
                'group-vertex-counts',
                groupVertexCounts,
                'uint32',
                group.groupCount
              )
            }
          }
        : {})
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    features: {
      lengths: await readFloat32(lengths, count),
      areas: await readFloat32(areas, count),
      signedAreas: await readFloat32(signedAreas, count),
      centroids: await readFloat32(centroids, 2 * count),
      bounds: await readFloat32(bounds, 4 * count),
      vertexCounts: await readUint32(vertexCounts, count),
      extremes: await readUint32(extremes, 4 * count)
    },
    groupCentroids: group ? await readFloat32(groupCentroids, 2 * group.groupCount) : undefined,
    groupVertexCounts: group ? await readUint32(groupVertexCounts, group.groupCount) : undefined
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

// Pinned from pyproj 3.8.0 `Geod(ellps='WGS84').geometry_area_perimeter(shapely.Polygon(ring))`
// (GeographicLib / Karney, f64); see scratchpad build/L/oracle.py. [lon, lat] degrees, CCW.
const GEODESIC_CASES: {
  name: string;
  ring: number[][];
  area: number;
  perimeter: number;
  areaTolerance: number;
}[] = [
  {
    name: 'Colorado-sized box',
    ring: [
      [-109, 37],
      [-102, 37],
      [-102, 41],
      [-109, 41]
    ],
    area: 269154549884.01074,
    perimeter: 2099854.381922906,
    areaTolerance: 5e-6
  },
  {
    name: 'continental triangle',
    ring: [
      [0, 0],
      [40, 10],
      [10, 50]
    ],
    area: 12195245913068.016,
    perimeter: 15412364.827327196,
    areaTolerance: 5e-6
  },
  {
    name: 'city polygon (edges near the 20 km threshold)',
    ring: [
      [-0.2, 51.4],
      [0.1, 51.4],
      [0.1, 51.6],
      [-0.2, 51.6]
    ],
    area: 463546956.6654968,
    perimeter: 86167.35011837249,
    areaTolerance: 5e-6
  },
  {
    name: 'southern hemisphere box',
    ring: [
      [150, -30],
      [150, -40],
      [165, -40],
      [165, -30]
    ],
    area: 1517312015144.1504,
    perimeter: 4944477.9127559215,
    areaTolerance: 5e-6
  },
  {
    name: 'near the antimeridian',
    ring: [
      [170, -10],
      [179, -5],
      [179.5, 5],
      [172, 8]
    ],
    area: 1405990432048.8142,
    perimeter: 5139926.123055087,
    areaTolerance: 5e-6
  },
  {
    name: 'very long edges (capped at 16 spans)',
    ring: [
      [-100, 10],
      [-20, 10],
      [-60, 70]
    ],
    area: 31077731009743.53,
    perimeter: 23209101.85266169,
    areaTolerance: 5e-6
  }
];

it('GPUGeometryMeasures geodesic area and perimeter match pyproj Geod (Karney)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = roundFeatures(GEODESIC_CASES.map(({ring}) => [ring]));
  const geodesic = await measure(device, features, {
    geometryType: 'polygons',
    coordinateSystem: 'geodesic'
  });
  const wgs84 = await measure(device, features, {
    geometryType: 'polygons',
    coordinateSystem: 'wgs84'
  });
  const report: string[] = [];
  GEODESIC_CASES.forEach((testCase, feature) => {
    const areaError = Math.abs(geodesic.features.areas[feature] / testCase.area - 1);
    const straightError = Math.abs(wgs84.features.areas[feature] / testCase.area - 1);
    const perimeterError = Math.abs(geodesic.features.lengths[feature] / testCase.perimeter - 1);
    report.push(
      `${testCase.name}: area ${areaError.toExponential(1)} (straight-edge wgs84 ${straightError.toExponential(1)}), perimeter ${perimeterError.toExponential(1)}`
    );
    expect(areaError, testCase.name).toBeLessThanOrEqual(testCase.areaTolerance);
    expect(perimeterError, `${testCase.name} perimeter`).toBeLessThanOrEqual(2e-5);
    // CCW rings in longitude/latitude order are positive, as pyproj reports.
    expect(geodesic.features.signedAreas[feature]).toBeGreaterThan(0);
    // Geodesic mode only changes areas: lengths and bounds equal the wgs84 mode's.
    expect(geodesic.features.lengths[feature]).toBe(wgs84.features.lengths[feature]);
    expect(geodesic.features.bounds.slice(4 * feature, 4 * feature + 4)).toEqual(
      wgs84.features.bounds.slice(4 * feature, 4 * feature + 4)
    );
  });
  // eslint-disable-next-line no-console
  console.log(`geodesic parity vs pyproj Geod:\n  ${report.join('\n  ')}`);
  // The geodesic correction is real: on the long-edge polygon straight equal-area edges are far off.
  const last = GEODESIC_CASES.length - 1;
  expect(Math.abs(wgs84.features.areas[last] / GEODESIC_CASES[last].area - 1)).toBeGreaterThan(
    10 * GEODESIC_CASES[last].areaTolerance
  );
  // Reversed winding flips the sign (pyproj: -269154549884.01074 for the reversed box).
  const reversed = await measure(device, roundFeatures([[[...GEODESIC_CASES[0].ring].reverse()]]), {
    geometryType: 'polygons',
    coordinateSystem: 'geodesic'
  });
  expect(reversed.features.signedAreas[0]).toBeLessThan(0);
  expect(reversed.features.areas[0] / geodesic.features.areas[0] - 1).toBeCloseTo(0, 5);
});

it('GPUGeometryMeasures reports extreme vertices with lowest-row ties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // numpy argmin/argmax of x and y (first occurrence, i.e. the lowest row) per feature, pinned from
  // the MultiPoint in build/L/oracle.py and by hand for the polygon (rows are global).
  const features = roundFeatures([
    [
      [
        [2, 3],
        [-1, 7],
        [5, 3],
        [5, -4],
        [-1, 9],
        [0, 0]
      ]
    ],
    // Closed square with a repeated closing vertex: ties resolve to the first vertex.
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0]
      ]
    ],
    // Empty feature.
    [[]]
  ]);
  const points = await measure(device, features, {geometryType: 'points'});
  expect(points.features.extremes.slice(0, 4)).toEqual([1, 3, 2, 4]);
  expect(points.features.extremes.slice(4, 8)).toEqual([6, 6, 7, 8]);
  expect(points.features.extremes.slice(8, 12)).toEqual([
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff
  ]);
  const polygons = await measure(device, features, {geometryType: 'polygons'});
  expect(polygons.features.extremes.slice(0, 4)).toEqual([1, 3, 2, 4]);
  expect(polygons.features.extremes.slice(4, 8)).toEqual([6, 6, 7, 8]);
  const lines = await measure(device, features, {geometryType: 'lines'});
  expect(lines.features.extremes.slice(0, 4)).toEqual([1, 3, 2, 4]);
});

it('GPUGeometryMeasures point features match shapely MultiPoint centroids', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // shapely.centroid(MultiPoint(pts)) = (1.6666666666666667, 3.0) for the first feature.
  const features = roundFeatures([
    [
      [
        [2, 3],
        [-1, 7],
        [5, 3],
        [5, -4],
        [-1, 9],
        [0, 0]
      ]
    ],
    // A single point.
    [[[7, -2]]],
    [[[1e6 + 1, 1]], [[1e6 + 3, 3]]]
  ]);
  const result = await measure(
    device,
    features,
    {geometryType: 'points'},
    {groupIds: [0, 1, 1], groupCount: 2}
  );
  expect(result.features.vertexCounts).toEqual([6, 1, 2]);
  expect(result.features.lengths).toEqual([0, 0, 0]);
  expect(result.features.centroids[0]).toBeCloseTo(1.6666666666666667, 5);
  expect(result.features.centroids[1]).toBeCloseTo(3, 5);
  expect(result.features.centroids.slice(2, 4)).toEqual([7, -2]);
  expect(result.features.centroids[4]).toBe(Math.fround(1e6 + 2));
  expect(result.features.centroids[5]).toBe(2);
  expect(Array.from(result.features.bounds.slice(0, 4))).toEqual([-1, -4, 5, 9]);
  // Group 1 holds points (7, -2), (1e6 + 1, 1), (1e6 + 3, 3): mean of the three points, not of
  // the two feature centroids.
  expect(result.groupVertexCounts).toEqual([6, 3]);
  expect(result.groupCentroids?.[2]).toBeCloseTo((7 + 2e6 + 4) / 3, -1);
  expect(result.groupCentroids?.[3]).toBeCloseTo(2 / 3, 4);
});
