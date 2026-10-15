// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.minimum-bounds';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUMinimumBounds',
  title: 'Minimum bounding rectangles of Chicago parks',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-parks',
    source: 'OpenStreetMap via Overpass (showcase chicago-parks, 2026-10-07)',
    licence: 'ODbL 1.0, (c) OpenStreetMap contributors'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata, createFixedOffsets},
      {runExample},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./minimum-bounds.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
    ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUMinimumBounds',
          datasets: [data.id],
          retain: ['rectangleCorners', 'rectangleSizes', 'circles', 'overflow'],
          keySpaces: [
            {id: 'park', label: 'park', size: data.featureCount, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'ring vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            },
            {id: 'ring', label: 'ring', size: data.ringOffsets.length - 1, indexSpace: '1d'},
            {id: 'corner', label: 'rectangle corner', size: data.featureCount * 4, indexSpace: '1d'}
          ],
          choices: {maximumHullVertices: 256},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            // Display index for the 4-corner output rings; not data.
            const rectangleOffsets = uploads.view(
              'rectangleOffsets',
              createFixedOffsets(data.featureCount, 4),
              'uint32'
            );
            const output = await runExample(graph, {parks});
            return {
              positions: {
                view: parks.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate', keySpace: 'ring'},
              rectangleOffsets: {view: rectangleOffsets, role: 'intermediate'},
              rectangleCorners: {
                view: output.rectangleCorners,
                role: 'output',
                keySpace: 'corner',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'rectangleOffsets'}),
                meaning: 'minimum-area rotated rectangle, 4 corners per park'
              },
              rectangleSizes: {
                view: output.rectangleSizes,
                role: 'intermediate',
                keySpace: 'park',
                meaning: 'rectangle [width, height, angle, area] in m, m, rad, m²'
              },
              circles: {
                view: output.circles,
                role: 'intermediate',
                keySpace: 'park',
                meaning: 'minimum bounding circle [centerX, centerY, radius] in m'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'}
            };
          }
        })
    };
  }
});
