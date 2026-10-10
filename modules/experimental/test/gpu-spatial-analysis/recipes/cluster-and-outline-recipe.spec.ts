// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/index';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {getGPUSpatialClusteringParameterValues} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {computeGroupConvexHullOracle} from '../group-geometry/group-geometry-oracle';
import {
  clusterPointsOracle,
  createClusteredPoints,
  separateNearEpsilonPairs
} from '../spatial-clustering/spatial-clustering-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

const EPSILON = 4;
const MINIMUM_POINTS = 4;
const MAXIMUM_CLUSTERS = 16;
const EXTENT = 100;

it('addClusterAndOutlineRecipe clusters points and outlines each cluster with its convex hull', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = separateNearEpsilonPairs(
    createClusteredPoints(3, {
      pointCount: 400,
      blobCount: 3,
      extent: EXTENT,
      blobSigma: 3,
      noiseFraction: 0.1
    }),
    [EPSILON]
  );
  const pointCount = positions.length / 2;
  const parameters = {
    bounds: [0, 0, EXTENT, EXTENT] as const,
    epsilon: EPSILON,
    minimumPoints: MINIMUM_POINTS
  };
  const clustering = clusterPointsOracle(positions, parameters);
  const clusterCount = clustering.clusterCount;
  expect(clusterCount).toBeGreaterThanOrEqual(2);
  expect(clusterCount).toBeLessThanOrEqual(MAXIMUM_CLUSTERS);
  const keys = clustering.labels.map(label =>
    label < MAXIMUM_CLUSTERS ? label : MAXIMUM_CLUSTERS
  );
  const hullOracle = computeGroupConvexHullOracle(positions, keys, MAXIMUM_CLUSTERS);

  const fixture = new RecipeTestFixture(device, 'cluster-outline-recipe-test');
  try {
    const labels = fixture.output('labels', 'uint32', pointCount);
    const count = fixture.output('cluster-count', 'uint32', 1);
    const memberCounts = fixture.output('counts', 'uint32', MAXIMUM_CLUSTERS);
    const meanCenters = fixture.output('mean-centers', 'float32x2', MAXIMUM_CLUSTERS);
    const hullOffsets = fixture.output('hull-offsets', 'uint32', MAXIMUM_CLUSTERS + 1);
    const hullCounts = fixture.output('hull-counts', 'uint32', MAXIMUM_CLUSTERS);
    const hullPositions = fixture.output('hull-positions', 'float32x2', 1024);
    const hullIndices = fixture.output('hull-indices', 'uint32', 1024);
    const hullOverflow = fixture.output('hull-overflow', 'uint32', 1);
    const areas = fixture.output('areas', 'float32', MAXIMUM_CLUSTERS);
    const perimeters = fixture.output('perimeters', 'float32', MAXIMUM_CLUSTERS);
    const recipe = addClusterAndOutlineRecipe(fixture.graph, {
      positions: fixture.input('positions', positions, 'float32x2', pointCount),
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      clusteringParameters: fixture.parameters(
        'clustering-parameters',
        'float32',
        getGPUSpatialClusteringParameterValues(parameters)
      ),
      gridSize: [32, 32],
      geometryParameters: fixture.parameters(
        'geometry-parameters',
        'float32',
        getGPUGeographicDistributionParameterValues()
      ),
      maximumClusterCount: MAXIMUM_CLUSTERS,
      maximumVerticesPerHull: 64,
      hullCapacity: 1024,
      outputs: {
        labels: labels.view,
        clusterCount: count.view,
        hullOffsets: hullOffsets.view,
        hullPositions: hullPositions.view,
        hullOverflow: hullOverflow.view,
        areas: areas.view,
        perimeters: perimeters.view
      },
      scratch: {
        counts: memberCounts.view,
        meanCenters: meanCenters.view,
        hullCounts: hullCounts.view,
        hullVertexIndices: hullIndices.view
      }
    });
    expect(recipe.contributors.length).toBe(4);
    fixture.run();

    expect(await fixture.readUint32(labels, pointCount)).toEqual(clustering.labels);
    expect((await fixture.readUint32(count, 1))[0]).toBe(clusterCount);
    expect((await fixture.readUint32(hullOverflow, 1))[0]).toBe(0);
    const gpuCounts = await fixture.readUint32(memberCounts, MAXIMUM_CLUSTERS);
    const gpuHullCounts = await fixture.readUint32(hullCounts, MAXIMUM_CLUSTERS);
    const gpuOffsets = await fixture.readUint32(hullOffsets, MAXIMUM_CLUSTERS + 1);
    const gpuIndices = await fixture.readUint32(hullIndices, 1024);
    const gpuAreas = await fixture.readFloat32(areas, MAXIMUM_CLUSTERS);
    const gpuPerimeters = await fixture.readFloat32(perimeters, MAXIMUM_CLUSTERS);
    const centers = await fixture.readFloat32(meanCenters, 2 * MAXIMUM_CLUSTERS);
    let totalArea = 0;
    for (let cluster = 0; cluster < clusterCount; cluster++) {
      // Members: core and border points of the cluster.
      const members = clustering.labels.filter(label => label === cluster).length;
      expect(gpuCounts[cluster]).toBe(members);
      const hull = hullOracle.hulls[cluster];
      expect(hull.length).toBeGreaterThanOrEqual(3);
      expect(gpuHullCounts[cluster]).toBe(hull.length);
      expect(gpuIndices.slice(gpuOffsets[cluster], gpuOffsets[cluster] + hull.length)).toEqual(
        hull
      );
      // Shoelace area and perimeter of the oracle hull in float64.
      let twiceArea = 0;
      let perimeter = 0;
      for (let vertex = 0; vertex < hull.length; vertex++) {
        const a = hull[vertex];
        const b = hull[(vertex + 1) % hull.length];
        twiceArea +=
          positions[2 * a] * positions[2 * b + 1] - positions[2 * b] * positions[2 * a + 1];
        perimeter += Math.hypot(
          positions[2 * b] - positions[2 * a],
          positions[2 * b + 1] - positions[2 * a + 1]
        );
      }
      expect(twiceArea).toBeGreaterThan(0);
      expect(isClose(gpuAreas[cluster], twiceArea / 2, 1e-2, 1e-4), `area ${cluster}`).toBe(true);
      expect(isClose(gpuPerimeters[cluster], perimeter, 1e-2, 1e-4), `perimeter ${cluster}`).toBe(
        true
      );
      totalArea += gpuAreas[cluster];
      // Mean center of the members.
      let sumX = 0;
      let sumY = 0;
      for (let row = 0; row < pointCount; row++) {
        if (clustering.labels[row] === cluster) {
          sumX += positions[2 * row];
          sumY += positions[2 * row + 1];
        }
      }
      expect(isClose(centers[2 * cluster], sumX / members, 1e-3, 1e-4)).toBe(true);
      expect(isClose(centers[2 * cluster + 1], sumY / members, 1e-3, 1e-4)).toBe(true);
    }
    expect(totalArea).toBeGreaterThan(10);
    // Slots past the cluster count are empty.
    for (let cluster = clusterCount; cluster < MAXIMUM_CLUSTERS; cluster++) {
      expect(gpuCounts[cluster]).toBe(0);
      expect(gpuHullCounts[cluster]).toBe(0);
    }
  } finally {
    fixture.destroy();
  }
}, 120000);
