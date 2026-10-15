// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.shape-generator';
/** Every 12th place gets a circle: 107 of the 1,275 places. */
const CENTER_STRIDE = 12;
const RADIUS = 60;
const SEGMENT_COUNT = 24;
const MAXIMUM_SEGMENTS = 32;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUShapeGenerator',
  title: 'Circles around Wicker Park places',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-places',
    source: 'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
    licence: 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata},
      {runExample},
      data,
      {getGPUShapeGeneratorParameterValues, getGPUShapeVertexCount}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./shape-generator.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces()),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    // Sub-selection: every 12th place, in dataset order.
    const centerCount = Math.ceil(data.pointCount / CENTER_STRIDE);
    const centers = new Float32Array(2 * centerCount);
    for (let center = 0; center < centerCount; center++) {
      const row = 2 * center * CENTER_STRIDE;
      centers.set(data.positionsProjected.subarray(row, row + 2), 2 * center);
    }
    const vertexCapacity = centerCount * getGPUShapeVertexCount('circle', MAXIMUM_SEGMENTS);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUShapeGenerator',
          datasets: [data.id],
          retain: ['circleVertices', 'circleOffsets', 'vertexCount'],
          keySpaces: [
            {id: 'place', label: 'place (every 12th)', size: centerCount, indexSpace: '1d'},
            {id: 'vertex', label: 'circle vertex', size: vertexCapacity, indexSpace: '1d'},
            {id: 'allPlace', label: 'place', size: data.pointCount, indexSpace: '1d'}
          ],
          choices: {
            shape: 'circle',
            segmentCount: SEGMENT_COUNT,
            maximumSegments: MAXIMUM_SEGMENTS,
            radiusScale: 1
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const allPlaces = uploads.view('places', data.positionsProjected, 'float32x2');
            const centerRows = uploads.view('centers', centers, 'float32x2');
            const radii = uploads.view(
              'radii',
              new Float32Array(centerCount).fill(RADIUS),
              'float32'
            );
            const parameters = uploads.view(
              'parameters',
              getGPUShapeGeneratorParameterValues({segmentCount: SEGMENT_COUNT, radiusScale: 1}),
              'float32'
            );
            const output = await runExample(graph, {centers: centerRows, radii, parameters});
            return {
              places: {
                view: allPlaces,
                role: 'context',
                keySpace: 'allPlace',
                geometry: projectedGeometry('points')
              },
              centers: {
                view: centerRows,
                role: 'input',
                keySpace: 'place',
                geometry: projectedGeometry('points'),
                meaning: `every ${CENTER_STRIDE}th place of the Wicker Park places`
              },
              radii: {
                view: radii,
                role: 'intermediate',
                keySpace: 'place',
                meaning: `circle radius in m (a parameter: ${RADIUS} m for every place)`
              },
              circleOffsets: {view: output.offsets, role: 'intermediate'},
              vertexCount: {view: output.vertexCount, role: 'intermediate'},
              circles: {
                view: output.positions,
                role: 'output',
                keySpace: 'vertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'circleOffsets'}),
                meaning: `${SEGMENT_COUNT}-segment circle rings of ${RADIUS} m; capacity past the written vertices is not drawn`
              }
            };
          }
        })
    };
  }
});
