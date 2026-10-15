// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.affine-transform';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUAffineTransform',
  title: 'Chicago parks rotated and shrunk about their centroids',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
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
        import('./affine-transform.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
      ]);
    const {getGPUAffineTransformParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUAffineTransform',
          datasets: [data.id],
          retain: ['transformedParks'],
          keySpaces: [
            {
              id: 'vertex',
              label: 'park vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'ring', label: 'ring', size: data.ringOffsets.length - 1, indexSpace: '1d'}
          ],
          choices: {
            rotate: 30,
            useRadians: false,
            scale: 0.6,
            skew: '0, 0',
            translate: '0, 0 (metres)',
            origin: 'centroid'
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
              getGPUAffineTransformParameterValues({rotate: 30, scale: 0.6, origin: 'centroid'}),
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
              transformedParks: {
                view: output.transformedParks,
                role: 'output',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning: 'park vertices rotated 30 degrees and scaled by 0.6 about each centroid'
              }
            };
          }
        })
    };
  }
});
