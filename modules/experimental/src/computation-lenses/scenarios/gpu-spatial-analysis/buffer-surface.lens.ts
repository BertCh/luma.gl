// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.buffer-surface';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUBufferSurface',
  title: '50 m buffers of Chicago parks',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-parks',
    source: 'OpenStreetMap via Overpass (showcase chicago-parks, 2026-10-07)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./buffer-surface.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    const {getGPUOffsetCurveParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    // Raw boundary rows are 35 per park vertex; the dissolved result needs far fewer.
    const capacity = {
      intersections: 8192,
      nodedSegments: 65536,
      boundary: 16384,
      rings: 256,
      vertices: 16384 + 256
    };
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUBufferSurface',
          datasets: [data.id],
          retain: [
            'bufferPositions',
            'bufferRingOffsets',
            'bufferCount',
            'bufferRequiredCount',
            'bufferOverflow',
            'bufferCandidateOverflow'
          ],
          keySpaces: [
            {id: 'ring', label: 'buffer ring', size: capacity.rings, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {
            distance: 50,
            mitreLimit: 5,
            joinStyle: 'round',
            capStyle: 'round',
            quadSegments: 8,
            vertexTolerance: 0.05
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPUOffsetCurveParameterValues({distance: 50}),
              'float32'
            );
            const output = await runExample(graph, {parks, parameters, capacity});
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate'},
              bufferCount: {view: output.status.count, role: 'intermediate'},
              bufferRequiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              bufferOverflow: {view: output.status.overflow, role: 'intermediate'},
              bufferCandidateOverflow: {
                view: output.status.candidateOverflow,
                role: 'intermediate'
              },
              bufferRingOffsets: {
                view: output.buffers.ringOffsets,
                role: 'intermediate',
                keySpace: 'ring'
              },
              bufferPositions: {
                view: output.buffers.positions,
                role: 'output',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'bufferRingOffsets'}),
                meaning:
                  '50 m buffers dissolved into polygons; ring offsets are padded past the used rings, so unused rings are empty'
              }
            };
          }
        })
    };
  }
});
