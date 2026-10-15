// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {assert} from '@luma.gl/core';

const SCENARIO_ID = 'gpu-spatial-analysis.line-clip-by-polygon';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineClipByPolygon',
  title: 'Clip Wicker Park streets to West Town',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads + s3-chicago-community-areas',
    source:
      'showcase public/data/chicago-roads (OpenStreetMap) + showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors + City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data, areas] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./line-clip-by-polygon.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads()),
        import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
      ]);
    const intersectionCapacity = 4096;
    const pieceCapacity = data.lineCount + 4 * intersectionCapacity;
    const vertexCapacity = data.positionsProjected.length / 2 + 8 * intersectionCapacity;
    // The selected West Town feature (area 24) has exactly one polygon and one ring.
    const areaIndex = areas.names.indexOf('area 24');
    assert(areaIndex >= 0);
    const polygonIndex = areas.featureOffsets[areaIndex];
    const ringIndex = areas.polygonOffsets[polygonIndex];
    assert(areas.featureOffsets[areaIndex + 1] === polygonIndex + 1);
    assert(areas.polygonOffsets[polygonIndex + 1] === ringIndex + 1);
    const polygonPositions = areas.positionsProjected.slice(
      areas.ringOffsets[ringIndex] * 2,
      areas.ringOffsets[ringIndex + 1] * 2
    );
    const candidateCapacity = data.positionsProjected.length / 2 + 2 * intersectionCapacity;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineClipByPolygon',
          datasets: [data.id, areas.id],
          retain: [
            'piecePositions',
            'pieceOffsets',
            'sourceIds',
            'count',
            'vertexCount',
            'overflow',
            'uncertainCount'
          ],
          keySpaces: [
            {id: 'piece', label: 'piece', size: pieceCapacity, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {mode: 'inside', intersectionCapacity, candidateCapacity},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const polygons = {
              positions: uploads.view('polygonPositions', polygonPositions, 'float32x2'),
              ringOffsets: uploads.view(
                'ringOffsets',
                new Uint32Array([0, polygonPositions.length / 2]),
                'uint32'
              ),
              polygonOffsets: uploads.view('polygonOffsets', new Uint32Array([0, 1]), 'uint32'),
              featureOffsets: uploads.view('featureOffsets', new Uint32Array([0, 1]), 'uint32')
            };
            const output = await runExample(graph, {
              roads,
              intersectionCapacity,
              pieceCapacity,
              vertexCapacity,
              polygons,
              candidateCapacity
            });
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              clipArea: {
                view: polygons.positions,
                role: 'input',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning: 'West Town, Chicago community area 24'
              },
              ringOffsets: {view: polygons.ringOffsets, role: 'intermediate'},
              pieceOffsets: {view: output.pieceOffsets, role: 'intermediate'},
              sourceIds: {
                view: output.pieces.sourceIds,
                role: 'intermediate',
                keySpace: 'piece',
                meaning: 'source road index (zero-based); unused slots are 0xffffffff'
              },
              count: {view: output.pieces.status.count, role: 'intermediate'},
              vertexCount: {view: output.pieces.status.vertexCount, role: 'intermediate'},
              overflow: {view: output.pieces.status.overflow, role: 'intermediate'},
              uncertainCount: {view: output.uncertainCount, role: 'intermediate'},
              piecePositions: {
                view: output.piecePositions,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'pieceOffsets'}),
                meaning: 'GPU piece offsets delimit only complete emitted paths'
              }
            };
          }
        })
    };
  }
});
