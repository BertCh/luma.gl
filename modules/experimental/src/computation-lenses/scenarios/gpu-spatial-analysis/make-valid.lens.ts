// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.make-valid';
const SEGMENT_CAPACITY = 1024;
const RING_CAPACITY = 128;
const VERTEX_CAPACITY = 2048;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUMakeValid',
  title: 'Chicago park polygons rebuilt by make-valid',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-parks',
    source: 'showcase public/data/chicago-parks (OpenStreetMap via Overpass, 2026-10-07)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./make-valid.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    // The park dataset has exactly one ring per feature, so face r of the output is park r's face.
    assert(data.ringOffsets.length === data.featureCount + 1);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUMakeValid',
          datasets: [data.id],
          retain: [
            'polygonCount',
            'polygonRequiredCount',
            'polygonOverflow',
            'polygonCandidateOverflow',
            'polygonSourceIds',
            'polygonPositions',
            'polygonRingOffsets'
          ],
          keySpaces: [
            {id: 'face', label: 'kept face', size: RING_CAPACITY, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'park ring vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'park', label: 'park', size: data.featureCount, indexSpace: '1d'},
            {id: 'faceVertex', label: 'face ring vertex', size: VERTEX_CAPACITY, indexSpace: '1d'}
          ],
          choices: {'precision.vertexTolerance': 0.01},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const output = await runExample(graph, {
              parks,
              segmentCapacity: SEGMENT_CAPACITY,
              ringCapacity: RING_CAPACITY,
              vertexCapacity: VERTEX_CAPACITY
            });
            const {polygons, status} = output;
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning:
                  '51 Chicago parks, one ring each; none self-crosses (checked), so they are already valid'
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate', keySpace: 'park'},
              polygonCount: {
                view: status.count,
                role: 'intermediate',
                meaning: 'polygons kept (51 parks in)'
              },
              polygonRequiredCount: {
                view: status.requiredCount,
                role: 'intermediate',
                meaning: 'face rings needed (ring capacity to allocate)'
              },
              polygonOverflow: {view: status.overflow, role: 'intermediate'},
              polygonCandidateOverflow: {view: status.candidateOverflow, role: 'intermediate'},
              polygonSourceIds: {
                view: polygons.sourceIds,
                role: 'intermediate',
                keySpace: 'face',
                meaning: 'source park of each kept face (zero-based feature row)'
              },
              polygonRingOffsets: {view: polygons.ringOffsets, role: 'intermediate'},
              polygonPositions: {
                view: polygons.positions,
                role: 'output',
                keySpace: 'faceVertex',
                geometry: projectedGeometry('polygons', {
                  offsetsCapture: 'polygonRingOffsets',
                  valuesCapture: 'polygonSourceIds'
                }),
                meaning:
                  'kept faces, UTM 16N metres, valued by source park; GPU ring offsets repeat past the last face, so unused slots are empty'
              }
            };
          }
        })
    };
  }
});
