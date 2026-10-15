// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.voronoi-diagram';
/** Every fourth place is a site: 319 of the 1,275 places keep the cells readable and the record small. */
const SITE_STRIDE = 4;
/** Hull rays are clipped this far beyond the sites' extent, in metres. */
const CLIP_MARGIN = 100;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUVoronoiDiagram',
  title: 'Voronoi cells of Wicker Park places',
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
      {getDelaunayMaximumTriangleCount, getVoronoiMaximumSegmentCount}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./voronoi-diagram.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces()),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    // Sub-selection: every fourth place, in dataset order.
    const sites = new Float32Array(2 * Math.ceil(data.pointCount / SITE_STRIDE));
    for (let site = 0; site * SITE_STRIDE < data.pointCount; site++) {
      const row = 2 * site * SITE_STRIDE;
      sites.set(data.positionsProjected.subarray(row, row + 2), 2 * site);
    }
    const siteCount = sites.length / 2;
    const triangleCapacity = getDelaunayMaximumTriangleCount(siteCount);
    const segmentCapacity = getVoronoiMaximumSegmentCount(triangleCapacity);
    const extent = [Infinity, Infinity, -Infinity, -Infinity];
    for (let site = 0; site < siteCount; site++) {
      extent[0] = Math.min(extent[0], sites[2 * site] - CLIP_MARGIN);
      extent[1] = Math.min(extent[1], sites[2 * site + 1] - CLIP_MARGIN);
      extent[2] = Math.max(extent[2], sites[2 * site] + CLIP_MARGIN);
      extent[3] = Math.max(extent[3], sites[2 * site + 1] + CLIP_MARGIN);
    }
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUVoronoiDiagram',
          datasets: [data.id],
          retain: ['triangleCount', 'siteIds', 'count', 'requiredCount', 'overflow'],
          keySpaces: [
            {id: 'segment', label: 'Voronoi segment', size: segmentCapacity, indexSpace: '1d'},
            {id: 'place', label: 'place (every 4th)', size: siteCount, indexSpace: '1d'}
          ],
          choices: {
            clipBounds: `places extent + ${CLIP_MARGIN} m: ${extent.map(Math.round).join(', ')}`
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const places = uploads.view('places', sites, 'float32x2');
            const triangleSlots = uploads.view(
              'triangles',
              new Uint32Array(3 * triangleCapacity),
              'uint32x3'
            );
            const segmentSlots = uploads.view(
              'segments',
              new Float32Array(4 * segmentCapacity).fill(Number.NaN),
              'float32x4'
            );
            const clipBounds = uploads.view('clipBounds', Float32Array.from(extent), 'float32x4');
            // Display only: the same segment bytes read as two float32x2 endpoints per segment.
            const endpoints = graph.createDataView(segmentSlots.buffer, {
              format: 'float32x2',
              length: 2 * segmentCapacity
            });
            // Display index for the segment endpoints (2 per segment); not data.
            const segmentOffsets = uploads.view(
              'segmentOffsets',
              createFixedOffsets(segmentCapacity, 2),
              'uint32'
            );
            const output = await runExample(graph, {
              places,
              triangleSlots,
              segmentSlots,
              clipBounds
            });
            return {
              places: {
                view: places,
                role: 'input',
                keySpace: 'place',
                geometry: projectedGeometry('points'),
                meaning: `every ${SITE_STRIDE}th place of the Wicker Park places, used as Voronoi sites`
              },
              segmentOffsets: {view: segmentOffsets, role: 'intermediate'},
              triangleCount: {view: output.triangleCount, role: 'intermediate'},
              siteIds: {
                view: output.siteIds,
                role: 'intermediate',
                keySpace: 'segment',
                meaning:
                  'the two place rows whose bisector owns each segment; unwritten slots are unspecified'
              },
              count: {view: output.status.count, role: 'intermediate'},
              requiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              voronoiEdges: {
                view: endpoints,
                role: 'output',
                geometry: projectedGeometry('lines', {offsetsCapture: 'segmentOffsets'}),
                meaning:
                  'Voronoi segments, two endpoints each; hull rays clipped to the places extent plus 100 m; unwritten slots are NaN (uploaded as NaN)'
              }
            };
          }
        })
    };
  }
});
