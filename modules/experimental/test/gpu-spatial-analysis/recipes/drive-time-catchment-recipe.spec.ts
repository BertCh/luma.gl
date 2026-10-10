// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {getGPUNetworkIsochroneParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import {
  addDriveTimeCatchmentRecipe,
  GPU_DRIVE_TIME_CATCHMENT_NO_BAND
} from '../../../src/gpu-spatial-analysis/recipes/drive-time-catchment-recipe';
import {
  buildCSR,
  createGridFixture,
  dijkstra,
  NONE,
  snapOracle
} from '../../gpu-network/network-accessibility/network-accessibility-oracle';
import {quadbinCellToTile, quadbinPointToCell} from '../cell-aggregation/cell-aggregation-oracle';
import {webMercatorTileBounds} from '../cell-indexing/cell-indexing-oracle';
import {GPU_SPATIAL_JOIN_NO_FEATURE} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {createSeededRandom} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {RecipeTestFixture, isClose} from './recipe-harness';

const WIDTH = 6;
const HEIGHT = 5;
const NODE_COUNT = WIDTH * HEIGHT;
const BAND_BREAKS = [3.3, 6.7, 11.1, 20.3];

it('addDriveTimeCatchmentRecipe bands demand by drive time to the nearest facility', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {positions, edges} = createGridFixture(3, WIDTH, HEIGHT);
  const csr = buildCSR(NODE_COUNT, edges);
  const random = createSeededRandom(21);
  const facilities = Float32Array.from([1.2, 1.3, 4.4, 3.1]);
  const demandCount = 90;
  const demand = new Float32Array(demandCount * 2);
  const demandValues = new Float32Array(demandCount);
  for (let row = 0; row < demandCount; row++) {
    demand[2 * row] = -0.4 + random() * (WIDTH - 1 + 0.8);
    demand[2 * row + 1] = -0.4 + random() * (HEIGHT - 1 + 0.8);
    demandValues[row] = 1 + Math.floor(random() * 9);
  }

  // CPU oracle: exhaustive snapping, multi-source Dijkstra, demand time over both edge ends.
  const snapOptions = {edgeCosts: csr.weights};
  const facilitySnaps = snapOracle(facilities, positions, csr.sources, csr.neighbors, snapOptions);
  const seeds = facilitySnaps.flatMap(snap => [
    {node: csr.sources[snap.edge], cost: snap.sourceCost},
    {node: csr.neighbors[snap.edge], cost: snap.targetCost}
  ]);
  const nodeCosts = dijkstra(csr, NODE_COUNT, seeds);
  const demandSnaps = snapOracle(demand, positions, csr.sources, csr.neighbors, snapOptions);
  const times = demandSnaps.map(snap =>
    snap.edge === NONE
      ? Infinity
      : Math.min(
          nodeCosts[csr.sources[snap.edge]] + snap.sourceCost,
          nodeCosts[csr.neighbors[snap.edge]] + snap.targetCost
        )
  );
  const bandOf = (time: number) => {
    const band = BAND_BREAKS.findIndex(limit => time < limit);
    return band < 0 ? GPU_DRIVE_TIME_CATCHMENT_NO_BAND : band;
  };

  const fixture = new RecipeTestFixture(device, 'drive-time-catchment-test');
  try {
    const {graph} = fixture;
    const outputs = {
      assignments: fixture.output('assignments', 'uint32', NODE_COUNT),
      nodeCosts: fixture.output('node-costs', 'float32', NODE_COUNT),
      times: fixture.output('demand-times', 'float32', demandCount),
      bands: fixture.output('demand-bands', 'uint32', demandCount),
      keys: fixture.output('band-keys', 'uint32', 8),
      counts: fixture.output('band-counts', 'uint32', 8),
      count: fixture.output('band-count', 'uint32', 1),
      overflow: fixture.output('band-overflow', 'uint32', 1),
      sums: fixture.output('band-sums', 'float32', 8),
      means: fixture.output('band-means', 'float32', 8),
      triangles: fixture.output('triangles', 'float32x2', 3 * 4000),
      triangleBands: fixture.output('triangle-bands', 'uint32', 4000),
      triangleCount: fixture.output('triangle-count', 'uint32', 1),
      triangleOverflow: fixture.output('triangle-overflow', 'uint32', 1)
    };
    const recipe = addDriveTimeCatchmentRecipe(graph, {
      network: {
        offsets: fixture.input('offsets', csr.offsets, 'uint32', NODE_COUNT + 1),
        neighbors: fixture.input('neighbors', csr.neighbors, 'uint32', csr.neighbors.length),
        weights: fixture.input('weights', csr.weights, 'float32', csr.weights.length),
        nodePositions: fixture.input('positions', positions, 'float32x2', NODE_COUNT)
      },
      facilities: fixture.input('facilities', facilities, 'float32x2', 2),
      demand: fixture.input('demand', demand, 'float32x2', demandCount),
      demandValues: fixture.input('demand-values', demandValues, 'float32', demandCount),
      maxIterations: 64,
      bandBreaks: fixture.input('band-breaks', Float32Array.from(BAND_BREAKS), 'float32', 4),
      bandCapacity: 8,
      outputs: {
        assignments: outputs.assignments.view,
        nodeCosts: outputs.nodeCosts.view,
        demandTimes: outputs.times.view,
        demandBands: outputs.bands.view,
        bands: {
          keys: outputs.keys.view,
          counts: outputs.counts.view,
          count: outputs.count.view,
          overflow: outputs.overflow.view,
          sumValues: outputs.sums.view,
          means: outputs.means.view
        }
      },
      isochrones: {
        breaks: fixture.input('iso-breaks', Float32Array.from(BAND_BREAKS), 'float32', 4),
        parameters: fixture.parameters(
          'iso-parameters',
          'float32',
          getGPUNetworkIsochroneParameterValues({
            breakCount: 4,
            extent: [-1, -1, WIDTH, HEIGHT],
            bufferRadius: 0.4
          })
        ),
        raster: {
          width: 48,
          height: 40,
          output: {
            triangles: outputs.triangles.view,
            triangleBands: outputs.triangleBands.view,
            count: outputs.triangleCount.view,
            overflow: outputs.triangleOverflow.view
          }
        }
      }
    });
    expect(recipe.contributors.length).toBe(5);
    expect(recipe.outputs?.demandTimes).toBe(recipe.demandTimes);
    expect(recipe.intermediates?.seedNodes).toBe(recipe.seedNodes);
    expect(recipe.status.stages.map(({stage}) => stage)).toEqual([
      'service-areas',
      'band-statistics'
    ]);
    expect(recipe.status.stages[1].status.overflow).toBe(recipe.bands.overflow);
    fixture.run();

    const gpuCosts = await fixture.readFloat32(outputs.nodeCosts, NODE_COUNT);
    for (let node = 0; node < NODE_COUNT; node++) {
      expect(gpuCosts[node], `node ${node}`).toBeCloseTo(nodeCosts[node], 3);
    }
    expect(Math.max(...nodeCosts)).toBeGreaterThan(5);
    const assignments = await fixture.readUint32(outputs.assignments, NODE_COUNT);
    expect(assignments.every(row => row < 4)).toBe(true);
    // Facility ids (row / 2) of both facilities appear.
    expect(new Set(assignments.map(row => row >> 1)).size).toBe(2);

    const gpuTimes = await fixture.readFloat32(outputs.times, demandCount);
    const gpuBands = await fixture.readUint32(outputs.bands, demandCount);
    const expectedCounts = new Map<number, {count: number; sum: number}>();
    let compared = 0;
    for (let row = 0; row < demandCount; row++) {
      expect(isClose(gpuTimes[row], times[row], 1e-3, 1e-4), `time ${row}`).toBe(true);
      const nearBreak = BAND_BREAKS.some(limit => Math.abs(times[row] - limit) < 1e-3);
      if (nearBreak) {
        continue;
      }
      expect(gpuBands[row], `band ${row}`).toBe(bandOf(times[row]));
      const entry = expectedCounts.get(bandOf(times[row])) ?? {count: 0, sum: 0};
      entry.count++;
      entry.sum += demandValues[row];
      expectedCounts.set(bandOf(times[row]), entry);
      compared++;
    }
    expect(compared).toBeGreaterThan(demandCount * 0.8);
    expect(new Set(gpuBands).size).toBeGreaterThan(2);

    const [bandCount] = await fixture.readUint32(outputs.count, 1);
    const keys = await fixture.readUint32(outputs.keys, bandCount);
    const groupCounts = await fixture.readUint32(outputs.counts, bandCount);
    const sums = await fixture.readFloat32(outputs.sums, bandCount);
    const expectedKeys = [...new Set(gpuBands.filter(band => band !== NONE))].sort((a, b) => a - b);
    expect(keys).toEqual(expectedKeys);
    for (const [index, key] of keys.entries()) {
      const expected = expectedCounts.get(key);
      if (expected && !gpuBands.some((band, row) => band === key && times[row] !== times[row])) {
        // Rows near a break may sit on either side; only exact when none were skipped.
        expect(Math.abs(groupCounts[index] - expected.count)).toBeLessThanOrEqual(3);
        expect(sums[index]).toBeGreaterThan(0);
      }
    }
    expect(groupCounts.reduce((total, value) => total + value, 0)).toBe(
      gpuBands.filter(band => band !== NONE).length
    );

    // Isochrone polygons of the same costs were produced.
    expect((await fixture.readUint32(outputs.triangleCount, 1))[0]).toBeGreaterThan(0);
    expect((await fixture.readUint32(outputs.triangleOverflow, 1))[0]).toBe(0);
  } finally {
    fixture.destroy();
  }
}, 120000);

it('addDriveTimeCatchmentRecipe returns isochrone rings and joins demand against them', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // The same grid network placed in longitude/latitude (the cell path needs degrees).
  const origin = [8.4, 47.1];
  const spacing = 0.012;
  const {positions: gridPositions, edges} = createGridFixture(3, WIDTH, HEIGHT);
  const positions = Float32Array.from(gridPositions, (value, index) =>
    Math.fround(origin[index % 2] + spacing * value)
  );
  const csr = buildCSR(NODE_COUNT, edges);
  const random = createSeededRandom(5);
  const facilities = Float32Array.from([
    origin[0] + spacing * 1.2,
    origin[1] + spacing * 1.3,
    origin[0] + spacing * 4.4,
    origin[1] + spacing * 3.1
  ]);
  const demandCount = 200;
  const demand = new Float32Array(demandCount * 2);
  for (let row = 0; row < demandCount; row++) {
    demand[2 * row] = origin[0] + spacing * (-0.6 + random() * (WIDTH - 1 + 1.2));
    demand[2 * row + 1] = origin[1] + spacing * (-0.6 + random() * (HEIGHT - 1 + 1.2));
  }
  const resolution = 14;
  const costLimit = 9;
  const ringCapacity = 64;
  const vertexCapacity = 2048;
  const segmentCapacity = 2048;

  const fixture = new RecipeTestFixture(device, 'drive-time-rings-test');
  try {
    const {graph} = fixture;
    const outputs = {
      nodeCosts: fixture.output('node-costs', 'float32', NODE_COUNT),
      rows: fixture.output('seg-rows', 'uint32', segmentCapacity),
      cells: fixture.output('seg-cells', 'uint32x2', segmentCapacity),
      edgeIndices: fixture.output('seg-edges', 'uint32', segmentCapacity),
      endpoints: fixture.output('seg-endpoints', 'float32x4', segmentCapacity),
      segmentCount: fixture.output('seg-count', 'uint32', 1),
      segmentOverflow: fixture.output('seg-overflow', 'uint32', 1),
      ringOffsets: fixture.output('ring-offsets', 'uint32', ringCapacity + 1),
      positions: fixture.output('ring-positions', 'float32x2', vertexCapacity),
      ringAreas: fixture.output('ring-areas', 'float32', ringCapacity),
      ringIsHole: fixture.output('ring-is-hole', 'uint32', ringCapacity),
      ringShells: fixture.output('ring-shells', 'uint32', ringCapacity),
      ringCount: fixture.output('ring-count', 'uint32', 1),
      ringOverflow: fixture.output('ring-overflow', 'uint32', 1),
      ringOpen: fixture.output('ring-open', 'uint32', 1),
      polygonPositions: fixture.output('polygon-positions', 'float32x2', vertexCapacity),
      polygonRingOffsets: fixture.output('polygon-ring-offsets', 'uint32', ringCapacity + 1),
      polygonOffsets: fixture.output('polygon-offsets', 'uint32', ringCapacity + 1),
      featureOffsets: fixture.output('feature-offsets', 'uint32', ringCapacity + 1),
      inside: fixture.output('isochrone-demand', 'uint32', demandCount),
      joinOverflow: fixture.output('join-overflow', 'uint32', 1)
    };
    const recipe = addDriveTimeCatchmentRecipe(graph, {
      network: {
        offsets: fixture.input('offsets', csr.offsets, 'uint32', NODE_COUNT + 1),
        neighbors: fixture.input('neighbors', csr.neighbors, 'uint32', csr.neighbors.length),
        weights: fixture.input('weights', csr.weights, 'float32', csr.weights.length),
        nodePositions: fixture.input('positions', positions, 'float32x2', NODE_COUNT)
      },
      facilities: fixture.input('facilities', facilities, 'float32x2', 2),
      demand: fixture.input('demand', demand, 'float32x2', demandCount),
      maxIterations: 64,
      bandBreaks: fixture.input('band-breaks', Float32Array.from(BAND_BREAKS), 'float32', 4),
      outputs: {
        nodeCosts: outputs.nodeCosts.view,
        isochroneDemand: {
          pointFeatureIds: outputs.inside.view,
          overflow: outputs.joinOverflow.view
        }
      },
      isochrones: {
        breaks: fixture.input('iso-breaks', Float32Array.of(costLimit), 'float32', 1),
        parameters: fixture.parameters(
          'iso-parameters',
          'float32',
          getGPUNetworkIsochroneParameterValues({
            breakCount: 1,
            extent: [8.3, 47, 8.6, 47.3],
            cellCostLimit: costLimit
          })
        ),
        cellOutline: {
          family: 'quadbin',
          resolution,
          output: {
            rows: outputs.rows.view,
            cells: outputs.cells.view,
            edgeIndices: outputs.edgeIndices.view,
            endpoints: outputs.endpoints.view,
            count: outputs.segmentCount.view,
            overflow: outputs.segmentOverflow.view
          },
          rings: {
            normalizeWinding: true,
            output: {
              ringOffsets: outputs.ringOffsets.view,
              positions: outputs.positions.view,
              ringAreas: outputs.ringAreas.view,
              ringIsHole: outputs.ringIsHole.view,
              ringShells: outputs.ringShells.view,
              polygons: {
                kind: 'polygons',
                positions: outputs.polygonPositions.view,
                ringOffsets: outputs.polygonRingOffsets.view,
                polygonOffsets: outputs.polygonOffsets.view,
                featureOffsets: outputs.featureOffsets.view
              },
              count: outputs.ringCount.view,
              overflow: outputs.ringOverflow.view,
              openSegmentCount: outputs.ringOpen.view
            }
          }
        },
        joinDemand: {candidateCapacity: 2 * demandCount}
      }
    });
    expect(recipe.isochroneRings).toBeDefined();
    expect(recipe.isochroneDemand).toBeDefined();
    fixture.run();

    // Oracle: reached nodes -> Quadbin cells -> union of tiles.
    const gpuCosts = await fixture.readFloat32(outputs.nodeCosts, NODE_COUNT);
    const cells = new Set<bigint>();
    for (let node = 0; node < NODE_COUNT; node++) {
      if (gpuCosts[node] <= costLimit) {
        cells.add(quadbinPointToCell(positions[2 * node], positions[2 * node + 1], resolution));
      }
    }
    expect(cells.size).toBeGreaterThan(3);
    const tileBounds = [...cells].map(cell => {
      const {x, y, z} = quadbinCellToTile(cell);
      return webMercatorTileBounds(x, y, z);
    });
    const tileArea = tileBounds.reduce(
      (sum, [west, south, east, north]) => sum + (east - west) * (north - south),
      0
    );

    expect((await fixture.readUint32(outputs.segmentOverflow, 1))[0]).toBe(0);
    expect((await fixture.readUint32(outputs.ringOverflow, 1))[0]).toBe(0);
    expect((await fixture.readUint32(outputs.ringOpen, 1))[0]).toBe(0);
    const [ringCount] = await fixture.readUint32(outputs.ringCount, 1);
    expect(ringCount).toBeGreaterThan(0);
    const areas = await fixture.readFloat32(outputs.ringAreas, ringCount);
    const isHole = await fixture.readUint32(outputs.ringIsHole, ringCount);
    const shells = await fixture.readUint32(outputs.ringShells, ringCount);
    const signedTotal = areas.reduce((sum, value) => sum + value, 0);
    expect(Math.abs(signedTotal - tileArea) / tileArea).toBeLessThan(1e-3);
    for (let ring = 0; ring < ringCount; ring++) {
      expect(areas[ring] > 0).toBe(isHole[ring] === 0);
      if (isHole[ring]) {
        expect(shells[ring]).toBeLessThan(ringCount);
      }
    }
    const offsets = await fixture.readUint32(outputs.ringOffsets, ringCount + 1);
    expect(offsets[ringCount]).toBeGreaterThan(4 * ringCount);

    // Demand inside the isochrone polygons equals demand inside a reached tile.
    const inside = await fixture.readUint32(outputs.inside, demandCount);
    expect((await fixture.readUint32(outputs.joinOverflow, 1))[0]).toBe(0);
    let expectedInside = 0;
    let compared = 0;
    const margin = 2e-5;
    for (let row = 0; row < demandCount; row++) {
      const x = demand[2 * row];
      const y = demand[2 * row + 1];
      const nearEdge = tileBounds.some(
        ([west, south, east, north]) =>
          x > west - margin &&
          x < east + margin &&
          y > south - margin &&
          y < north + margin &&
          !(x > west + margin && x < east - margin && y > south + margin && y < north - margin)
      );
      if (nearEdge) {
        continue;
      }
      const expected = tileBounds.some(
        ([west, south, east, north]) => x > west && x < east && y > south && y < north
      );
      expect(inside[row] !== GPU_SPATIAL_JOIN_NO_FEATURE, `demand ${row}`).toBe(expected);
      expectedInside += expected ? 1 : 0;
      compared++;
    }
    expect(compared).toBeGreaterThan(demandCount * 0.7);
    expect(expectedInside).toBeGreaterThan(10);
    expect(expectedInside).toBeLessThan(compared - 10);
  } finally {
    fixture.destroy();
  }
}, 120000);
