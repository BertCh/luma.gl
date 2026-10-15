// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {statSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import '../../src/computation-lenses/scenarios/index';
import {getLensScenario} from '../../src/computation-lenses/index';
import {
  loadS3ChicagoCommunityAreas,
  loadS3ChicagoParks,
  loadS3ChicagoPlaces,
  loadS3ChicagoRoads,
  loadS3NewYorkHarborTracks,
  loadS3NewYorkHarborZones,
  loadS3OpenFlightsHub
} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-data';
import {CELLS_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/cells';
import {GEOMETRY_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/geometry';
import {LINES_OPERATORS} from '../../src/gpu-spatial-analysis/computation-pack/operators/lines';

/** Elements of the three S3 docs pages (operators-cells/geometry/lines.md), in page order. */
const S3_ELEMENTS = [
  ...[
    'GPUCellAggregation',
    'GPUCellRollup',
    'GPUCellPyramid',
    'GPUCellLevelSelection',
    'GPUPointToCell',
    'GPUCellGeometry',
    'GPUCellTopology',
    'GPUCellCompaction',
    'GPUCellGridPath',
    'GPUCellMeasures',
    'GPUCellCover',
    'GPUCellSetOutline',
    'GPUSegmentRingAssembly',
    'GPULineNoding',
    'GPUPolygonize',
    'GPUMakeValid',
    'GPUCellTableCompare'
  ],
  ...[
    'GPUOutlineGeometry',
    'GPUOffsetCurve',
    'GPUVertexSnap',
    'GPUPolygonOverlay',
    'GPUBufferSurface',
    'GPUAffineTransform',
    'GPUGeometryOrientation',
    'GPUGeometryCleanup',
    'GPUMinimumBounds',
    'GPUPolygonTriangulation',
    'GPUDelaunayTessellation',
    'GPUVoronoiDiagram',
    'GPULabelPoint',
    'GPUShapeDescriptors',
    'GPULineDensity',
    'GPULineLengthPerPolygon',
    'GPUGridGenerator',
    'GPUShapeGenerator',
    'GPUHilbertKeys',
    'GPULonLatPolygonPreparation',
    'GPURectangleClip'
  ],
  ...[
    'GPULineSegmentize',
    'GPUGreatCircleArcs',
    'GPULineSmooth',
    'GPULineChunk',
    'GPUGeometryMeasures',
    'GPUGeodesicPairs',
    'GPUGeodesicDestination',
    'GPULocalFrameProjection',
    'GPULinearReferencing',
    'GPULineLocate',
    'GPULineSplit',
    'GPULineMerge',
    'GPULineClipByPolygon',
    'GPUSharedPaths',
    'GPULineSimplification',
    'GPUCoverageSimplification',
    'GPUCoverageDissolve',
    'GPUTrajectoryMetrics',
    'GPUTrajectoryPlayhead',
    'GPUTrajectoryResample',
    'GPUZoneEvents',
    'addFleetDwellZoneEventsRecipe',
    'GPUTrajectoryEncounters',
    'GPUTrajectoryClosestApproach',
    'GPUTrackSimilarity'
  ]
];

/**
 * Set once every S3 element has its scenario; until then only the scenarios that exist are checked.
 * The coverage contract (`lens-coverage.node.spec.ts`) enforces completeness independently.
 */
const REQUIRE_ALL_SCENARIOS = true;

/** `GPUMinimumBounds` → `minimum-bounds`, `addFleetDwellZoneEventsRecipe` → `fleet-dwell-zone-events-recipe`. */
function getScenarioId(element: string): string {
  return `gpu-spatial-analysis.${element
    .replace(/^GPU(?=[A-Z])|^add(?=[A-Z])/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
    .toLowerCase()}`;
}

function expectOffsets(offsets: Uint32Array, end: number, label: string): void {
  expect(offsets[0], label).toBe(0);
  expect(offsets[offsets.length - 1], label).toBe(end);
  for (let index = 1; index < offsets.length; index++) {
    expect(offsets[index] >= offsets[index - 1], `${label} monotone at ${index}`).toBe(true);
  }
}

function expectCoordinates(lonLat: Float32Array, projected: Float32Array, label: string): void {
  expect(projected.length, label).toBe(lonLat.length);
  expect(lonLat.every(Number.isFinite) && projected.every(Number.isFinite), label).toBe(true);
}

describe('S3 geometry lens data', () => {
  test('every data file stays within the 150 KB docs budget', () => {
    for (const name of [
      's3-chicago-parks',
      's3-chicago-community-areas',
      's3-chicago-roads',
      's3-chicago-places',
      's3-ny-harbor-ais',
      's3-ny-harbor-zones',
      's3-openflights-ord'
    ]) {
      const path = new URL(`../../src/computation-lenses/data/${name}.json`, import.meta.url);
      expect(statSync(path).size, name).toBeLessThanOrEqual(150 * 1024);
    }
  });

  test('polygon datasets have consistent four-level offsets', async () => {
    for (const polygons of [
      await loadS3ChicagoParks(),
      await loadS3ChicagoCommunityAreas(),
      await loadS3NewYorkHarborZones()
    ]) {
      const vertexCount = polygons.positionsLonLat.length / 2;
      expectCoordinates(polygons.positionsLonLat, polygons.positionsProjected, polygons.id);
      expectOffsets(polygons.ringOffsets, vertexCount, `${polygons.id} rings`);
      expectOffsets(
        polygons.polygonOffsets,
        polygons.ringOffsets.length - 1,
        `${polygons.id} polygons`
      );
      expectOffsets(
        polygons.featureOffsets,
        polygons.polygonOffsets.length - 1,
        `${polygons.id} features`
      );
      expect(polygons.featureOffsets.length - 1).toBe(polygons.featureCount);
      expect(polygons.projected.unit).toBe('m');
    }
  });

  test('line, point, track and flight datasets are consistent', async () => {
    const roads = await loadS3ChicagoRoads();
    expectCoordinates(roads.positionsLonLat, roads.positionsProjected, roads.id);
    expectOffsets(roads.lineOffsets, roads.positionsLonLat.length / 2, roads.id);
    expect(roads.edgeClass.length).toBe(roads.lineCount);
    const places = await loadS3ChicagoPlaces();
    expectCoordinates(places.positionsLonLat, places.positionsProjected, places.id);
    expect(places.category.length).toBe(places.pointCount);
    const tracks = await loadS3NewYorkHarborTracks();
    expectCoordinates(tracks.positionsLonLat, tracks.positionsProjected, tracks.id);
    expectOffsets(tracks.trackOffsets, tracks.timestamps.length, tracks.id);
    expect(tracks.positionsLonLat.length / 2).toBe(tracks.timestamps.length);
    for (let track = 0; track < tracks.trackCount; track++) {
      for (let fix = tracks.trackOffsets[track] + 1; fix < tracks.trackOffsets[track + 1]; fix++) {
        expect(
          tracks.timestamps[fix] > tracks.timestamps[fix - 1],
          `track ${track} fix ${fix}`
        ).toBe(true);
      }
    }
    const flights = await loadS3OpenFlightsHub();
    expect(flights.destinationsLonLat.length / 2).toBe(flights.destinationCount);
    expect(flights.iata.length).toBe(flights.destinationCount);
  });
});

describe('S3 geometry lens scenarios', () => {
  const operators = {...CELLS_OPERATORS, ...GEOMETRY_OPERATORS, ...LINES_OPERATORS};

  test('every registered S3 scenario matches its pack entry and card contract', () => {
    for (const element of S3_ELEMENTS) {
      const scenario = getLensScenario(getScenarioId(element));
      if (!scenario) {
        expect(REQUIRE_ALL_SCENARIOS, `${element}: no scenario ${getScenarioId(element)}`).toBe(
          false
        );
        continue;
      }
      const entry = operators[element];
      expect(entry, `${element}: pack entry`).toBeDefined();
      expect(scenario.element).toBe(element);
      expect(scenario.entry).toBe('gpu-spatial-analysis');
      expect(scenario.audience).toBe('analyst');
      // Card order: result first; the graph rung is always available from describe().
      expect(scenario.rungs[0], element).toBe('result');
      expect(scenario.defaultRung, element).toBe('result');
      expect(scenario.rungs, element).toContain('graph');
      // FormulaCard is listed only when the pack has a closed form.
      expect(scenario.rungs.includes('formula'), element).toBe('tex' in entry.formula);
      expect(scenario.dataset?.licence, element).toBeTruthy();
    }
  });
});
