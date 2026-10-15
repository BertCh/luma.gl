// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.polygon-triangulation';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUPolygonTriangulation',
  title: 'Earcut triangles of Chicago parks',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
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
      import('./polygon-triangulation.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
    ]);
    const {
      GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK,
      GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES,
      getPolygonTriangulationIndexCount
    } = await import('@luma.gl/experimental/gpu-spatial-analysis');
    const vertexCount = data.positionsProjected.length / 2;
    const indexCount = getPolygonTriangulationIndexCount(
      vertexCount,
      data.ringOffsets.length - 1,
      data.polygonOffsets.length - 1
    );
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUPolygonTriangulation',
          datasets: [data.id],
          retain: ['triangleIndices', 'valid', 'triangleCount', 'triangleOffsets', 'triangles'],
          keySpaces: [
            {id: 'park', label: 'park', size: data.featureCount, indexSpace: '1d'},
            {id: 'vertex', label: 'park vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'index', label: 'triangle index', size: indexCount, indexSpace: '1d'}
          ],
          choices: {
            maximumWork: GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK,
            largePolygonMethod: 'monotone',
            monotoneMinimumVertices: GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES,
            monotoneMaximumVertices: vertexCount,
            useZOrderHash: true
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const parks = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            // Display index for the gathered triangle corners (3 per triangle); not data.
            const triangleCornerOffsets = uploads.view(
              'triangleCornerOffsets',
              createFixedOffsets(indexCount / 3, 3),
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
              ringOffsets: {view: parks.ringOffsets, role: 'intermediate'},
              triangleIndices: {
                view: output.indices,
                role: 'intermediate',
                keySpace: 'index',
                meaning:
                  'park vertex row of each triangle corner (three per triangle); unused slots repeat the first row'
              },
              valid: {
                view: output.valid,
                role: 'intermediate',
                keySpace: 'park',
                meaning: '1 when the park triangulated, 0 when Earcut failed'
              },
              triangleCount: {
                view: output.triangleCount,
                role: 'intermediate',
                keySpace: 'park',
                meaning: 'triangles in the park (count)'
              },
              triangleOffsets: {view: output.triangleOffsets, role: 'intermediate'},
              triangleCornerOffsets: {view: triangleCornerOffsets, role: 'intermediate'},
              triangles: {
                view: output.triangles,
                role: 'output',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'triangleCornerOffsets'}),
                meaning: 'triangle corners gathered on the GPU for display'
              }
            };
          }
        })
    };
  }
});
