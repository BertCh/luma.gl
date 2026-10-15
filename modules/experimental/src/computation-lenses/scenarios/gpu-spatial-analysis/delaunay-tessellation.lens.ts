// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.delaunay-tessellation';
/** Every fourth place is a site: 319 of the 1,275 places keep the triangles readable and the record small. */
const SITE_STRIDE = 4;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUDelaunayTessellation',
  title: 'Delaunay triangulation of Wicker Park places',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-places',
    source: 'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
    licence: 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata, createFixedOffsets},
      {runExample},
      data,
      {getDelaunayMaximumTriangleCount}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./delaunay-tessellation.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces()),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    // Sub-selection: every fourth place, in dataset order.
    const sites = new Float32Array(2 * Math.ceil(data.pointCount / SITE_STRIDE));
    for (let site = 0; site * SITE_STRIDE < data.pointCount; site++) {
      sites.set(
        data.positionsProjected.subarray(2 * site * SITE_STRIDE, 2 * site * SITE_STRIDE + 2),
        2 * site
      );
    }
    const siteCount = sites.length / 2;
    const triangleCapacity = getDelaunayMaximumTriangleCount(siteCount);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUDelaunayTessellation',
          datasets: [data.id],
          retain: ['count', 'requiredCount', 'overflow', 'duplicateCount', 'triangleCorners'],
          keySpaces: [
            {id: 'triangle', label: 'Delaunay triangle', size: triangleCapacity, indexSpace: '1d'},
            {id: 'place', label: 'place (every 4th)', size: siteCount, indexSpace: '1d'}
          ],
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const places = uploads.view('places', sites, 'float32x2');
            const triangleSlots = uploads.view(
              'triangles',
              new Uint32Array(3 * triangleCapacity),
              'uint32x3'
            );
            // Display index for the gathered triangle corners (3 per triangle); not data.
            const triangleCornerOffsets = uploads.view(
              'triangleCornerOffsets',
              createFixedOffsets(triangleCapacity, 3),
              'uint32'
            );
            const output = await runExample(graph, {places, triangleSlots});
            return {
              places: {
                view: places,
                role: 'input',
                keySpace: 'place',
                geometry: projectedGeometry('points'),
                meaning: `every ${SITE_STRIDE}th place of the Wicker Park places, used as Delaunay sites`
              },
              triangleCornerOffsets: {view: triangleCornerOffsets, role: 'intermediate'},
              triangles: {
                view: output.triangles,
                role: 'intermediate',
                keySpace: 'triangle',
                meaning:
                  'counter-clockwise triple of place rows; slots past the written count keep row 0 (uploaded zeros)'
              },
              count: {view: output.status.count, role: 'intermediate'},
              requiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              duplicateCount: {view: output.status.duplicateCount, role: 'intermediate'},
              triangleCorners: {
                view: output.triangleCorners,
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
