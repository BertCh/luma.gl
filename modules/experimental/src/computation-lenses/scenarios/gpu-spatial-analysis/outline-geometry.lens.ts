// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.outline-geometry';

/** Triangles in each round-join disc of the example; the example passes the same value. */
const JOIN_SEGMENTS = 8;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUOutlineGeometry',
  title: '100 m outlines of Chicago parks',
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
      import('./outline-geometry.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoParks())
    ]);
    const {getGPUOutlineGeometryParameterValues, getGPUOutlineGeometryVerticesPerInput} =
      await import('@luma.gl/experimental/gpu-spatial-analysis');
    const vertexCount = data.positionsProjected.length / 2;
    const verticesPerInput = getGPUOutlineGeometryVerticesPerInput(JOIN_SEGMENTS);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUOutlineGeometry',
          datasets: [data.id],
          retain: ['outline'],
          keySpaces: [
            {id: 'vertex', label: 'park vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'ring', label: 'ring', size: data.ringOffsets.length - 1, indexSpace: '1d'},
            {
              id: 'outline-vertex',
              label: 'outline triangle vertex',
              size: vertexCount * verticesPerInput,
              indexSpace: '1d'
            }
          ],
          choices: {distance: 100, joinSegments: JOIN_SEGMENTS, geometryType: 'polygons'},
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
              getGPUOutlineGeometryParameterValues({distance: 100}),
              'float32'
            );
            // Display index for the triangle-list output (3 vertices per triangle); not data.
            const triangleOffsets = uploads.view(
              'triangleOffsets',
              createFixedOffsets(vertexCount * (JOIN_SEGMENTS + 2), 3),
              'uint32'
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
              triangleOffsets: {view: triangleOffsets, role: 'intermediate'},
              outline: {
                view: output.outline,
                role: 'output',
                keySpace: 'outline-vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'triangleOffsets'}),
                meaning: `outline triangles, ${verticesPerInput} vertices per park vertex (${JOIN_SEGMENTS} join disc triangles, 2 quad triangles); unused triangles are degenerate`
              }
            };
          }
        })
    };
  }
});
