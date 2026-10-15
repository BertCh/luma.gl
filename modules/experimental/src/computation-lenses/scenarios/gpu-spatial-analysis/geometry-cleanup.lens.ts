// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.geometry-cleanup';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeometryCleanup',
  title: 'Chicago parks snapped to a 10 m grid with repeats removed',
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
        import('./geometry-cleanup.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    const {getGPUGeometryCleanupParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGeometryCleanup',
          datasets: [data.id],
          retain: [
            'cleanedParks',
            'cleanedRingOffsets',
            'cleanedCount',
            'cleanedOverflow',
            'cleanedRequiredCount',
            'collapsedCount'
          ],
          keySpaces: [
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'ring', label: 'ring', size: data.ringOffsets.length - 1, indexSpace: '1d'}
          ],
          choices: {gridSize: 10, tolerance: 20, removeRepeatedPoints: true},
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
              getGPUGeometryCleanupParameterValues({
                gridSize: 10,
                tolerance: 20,
                removeRepeatedPoints: true
              }),
              'float32'
            );
            const output = await runExample(graph, {parks, parameters});
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate', keySpace: 'ring'},
              cleanedCount: {view: output.status.count, role: 'intermediate'},
              cleanedOverflow: {view: output.status.overflow, role: 'intermediate'},
              cleanedRequiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              collapsedCount: {view: output.status.collapsedCount, role: 'intermediate'},
              cleanedRingOffsets: {view: output.cleanedRingOffsets, role: 'intermediate'},
              cleanedParks: {
                view: output.cleanedParks,
                role: 'output',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'cleanedRingOffsets'}),
                meaning:
                  'vertices snapped to a 10 m grid, then repeats within 20 m removed; compacted, offsets exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
