// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {assert} from '@luma.gl/core';

const SCENARIO_ID = 'gpu-spatial-analysis.polygon-overlay';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUPolygonOverlay',
  title: 'Chicago parks intersected with the Uptown community area',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-parks + s3-chicago-community-areas',
    source:
      'showcase public/data/chicago-parks (OpenStreetMap via Overpass) + showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors + City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data, areas] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./polygon-overlay.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks()),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    // Uptown (community area 3) is one polygon with one ring; it holds 8 parks and cuts one more.
    const areaIndex = areas.names.indexOf('area 3');
    assert(areaIndex >= 0);
    const polygonIndex = areas.featureOffsets[areaIndex];
    const ringIndex = areas.polygonOffsets[polygonIndex];
    assert(areas.featureOffsets[areaIndex + 1] === polygonIndex + 1);
    assert(areas.polygonOffsets[polygonIndex + 1] === ringIndex + 1);
    const areaPositions = areas.positionsProjected.slice(
      areas.ringOffsets[ringIndex] * 2,
      areas.ringOffsets[ringIndex + 1] * 2
    );
    const capacity = {
      intersections: 4096,
      nodedSegments: 16384,
      boundary: 4096,
      rings: 256,
      vertices: 4096 + 256
    };
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUPolygonOverlay',
          datasets: [data.id, areas.id],
          retain: [
            'overlayPositions',
            'overlayRingOffsets',
            'overlayCount',
            'overlayRequiredCount',
            'overlayOverflow',
            'overlayCandidateOverflow',
            'boundaryOperandIds',
            'boundarySourceFeatureIds',
            'boundaryCount',
            'boundaryOverflow'
          ],
          keySpaces: [
            {id: 'segment', label: 'kept boundary edge', size: capacity.boundary, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'ring', label: 'output ring', size: capacity.rings, indexSpace: '1d'},
            {
              id: 'area-vertex',
              label: 'community-area vertex',
              size: areas.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {operation: 'intersection', fillRule: 'even-odd', vertexTolerance: 0.05},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const area = {
              positions: uploads.view('areaPositions', areaPositions, 'float32x2'),
              ringOffsets: uploads.view(
                'areaRingOffsets',
                new Uint32Array([0, areaPositions.length / 2]),
                'uint32'
              ),
              polygonOffsets: uploads.view('areaPolygonOffsets', new Uint32Array([0, 1]), 'uint32'),
              featureOffsets: uploads.view('areaFeatureOffsets', new Uint32Array([0, 1]), 'uint32')
            };
            const coverage = {
              positions: uploads.view('coveragePositions', areas.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('coverageRingOffsets', areas.ringOffsets, 'uint32')
            };
            const output = await runExample(graph, {parks, area, capacity});
            return {
              coverage: {
                view: coverage.positions,
                role: 'context',
                keySpace: 'area-vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'coverageRingOffsets'}),
                meaning: '17 community areas'
              },
              coverageRingOffsets: {view: coverage.ringOffsets, role: 'intermediate'},
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate'},
              uptown: {
                view: area.positions,
                role: 'input',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'areaRingOffsets'}),
                meaning: 'Uptown, Chicago community area 3'
              },
              areaRingOffsets: {view: area.ringOffsets, role: 'intermediate'},
              boundaryOperandIds: {
                view: output.boundary.operandIds,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'operand of the kept edge: 0 = park, 1 = community area'
              },
              boundarySourceFeatureIds: {
                view: output.boundary.sourceFeatureIds,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'source feature of the kept edge within its operand (zero-based)'
              },
              boundaryCount: {view: output.boundary.status.count, role: 'intermediate'},
              boundaryOverflow: {view: output.boundary.status.overflow, role: 'intermediate'},
              overlayCount: {view: output.status.count, role: 'intermediate'},
              overlayRequiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              overlayOverflow: {view: output.status.overflow, role: 'intermediate'},
              overlayCandidateOverflow: {
                view: output.status.candidateOverflow,
                role: 'intermediate'
              },
              overlayRingOffsets: {
                view: output.overlap.ringOffsets,
                role: 'intermediate',
                keySpace: 'ring'
              },
              overlayPositions: {
                view: output.overlap.positions,
                role: 'output',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'overlayRingOffsets'}),
                meaning:
                  'parks clipped to Uptown; ring offsets are padded past the used rings, so unused rings are empty'
              }
            };
          }
        })
    };
  }
});
