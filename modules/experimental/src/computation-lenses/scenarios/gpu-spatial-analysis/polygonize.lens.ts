// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.polygonize';
const SEGMENT_CAPACITY = 2048;
const RING_CAPACITY = 512;
const VERTEX_CAPACITY = 8192;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUPolygonize',
  title: 'Wicker Park street centrelines polygonized into city blocks',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads',
    source: 'showcase public/data/chicago-roads (OpenStreetMap)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./polygonize.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads())
      ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUPolygonize',
          datasets: [data.id],
          retain: [
            'nodedCount',
            'edgeClasses',
            'polygonCount',
            'polygonRequiredCount',
            'polygonOverflow',
            'polygonCandidateOverflow',
            'polygonPositions',
            'polygonRingOffsets'
          ],
          keySpaces: [
            {id: 'segment', label: 'noded segment', size: SEGMENT_CAPACITY, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'road vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'},
            {id: 'faceVertex', label: 'face ring vertex', size: VERTEX_CAPACITY, indexSpace: '1d'}
          ],
          choices: {'precision.vertexTolerance': 0.01},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const output = await runExample(graph, {
              roads,
              segmentCapacity: SEGMENT_CAPACITY,
              ringCapacity: RING_CAPACITY,
              vertexCapacity: VERTEX_CAPACITY
            });
            const {nodedSegments, diagnostics, polygons, status} = output;
            return {
              roads: {
                view: roads.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'}),
                meaning: '223 Wicker Park street centrelines, UTM 16N metres'
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate', keySpace: 'road'},
              nodedCount: {
                view: nodedSegments.status.count,
                role: 'intermediate',
                meaning: 'noded segments written'
              },
              edgeClasses: {
                view: diagnostics.edgeClasses,
                role: 'intermediate',
                keySpace: 'segment',
                meaning:
                  'class bits per noded segment: 1 closed ring, 2 open chain, 4 cut, 8 dangle, 16 unused'
              },
              polygonCount: {view: status.count, role: 'intermediate', meaning: 'faces written'},
              polygonRequiredCount: {
                view: status.requiredCount,
                role: 'intermediate',
                meaning: 'face rings needed (ring capacity to allocate)'
              },
              polygonOverflow: {view: status.overflow, role: 'intermediate'},
              polygonCandidateOverflow: {view: status.candidateOverflow, role: 'intermediate'},
              polygonRingOffsets: {view: polygons.ringOffsets, role: 'intermediate'},
              polygonPositions: {
                view: polygons.positions,
                role: 'output',
                keySpace: 'faceVertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'polygonRingOffsets'}),
                meaning:
                  'bounded faces of the street network (city blocks), UTM 16N metres; GPU ring offsets repeat past the last face, so unused slots are empty'
              }
            };
          }
        })
    };
  }
});
