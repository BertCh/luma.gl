// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.lon-lat-polygon-preparation';
/** Densification step in degrees (about 0.5 km of latitude). */
const MAXIMUM_SEGMENT_DEGREES = 0.005;
const MAXIMUM_PIECE_COUNT = 64;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULonLatPolygonPreparation',
  title: 'Prepared lon/lat harbour zones',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-ny-harbor-zones',
    source: 'showcase public/data/ais-zones (NOAA Coast Survey ENCDirect)',
    licence: 'Public domain (NOAA)'
  },
  load: async () => {
    const [
      {runS3Lens, geographicGeometry, geographicMetadata},
      {runExample},
      data,
      {getGPULonLatPolygonPreparationParameterValues}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./lon-lat-polygon-preparation.example'),
      import('./s3-lens-data').then(module => module.loadS3NewYorkHarborZones()),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    const ringCount = data.ringOffsets.length - 1;
    // Upper bound of the densified vertex count: each edge splits into at most
    // ceil(length / step) pieces. The element reports a conservative bound of
    // 8 * (dense count + 3 per ring) for both clipped copies; the capacity follows it.
    let denseBound = ringCount;
    for (let ring = 0; ring < ringCount; ring++) {
      const start = data.ringOffsets[ring];
      const end = data.ringOffsets[ring + 1];
      for (let row = start; row < end; row++) {
        const next = row + 1 < end ? row + 1 : start;
        const length = Math.hypot(
          data.positionsLonLat[2 * next] - data.positionsLonLat[2 * row],
          data.positionsLonLat[2 * next + 1] - data.positionsLonLat[2 * row + 1]
        );
        denseBound += Math.min(
          MAXIMUM_PIECE_COUNT,
          Math.max(1, Math.ceil(length / MAXIMUM_SEGMENT_DEGREES))
        );
      }
    }
    const vertexCapacity = 8 * (denseBound + 3 * ringCount);
    const vertexCount = data.positionsLonLat.length / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULonLatPolygonPreparation',
          datasets: [data.id],
          retain: [
            'prepared',
            'preparedFeatureOffsets',
            'preparedPolygonOffsets',
            'preparedRingOffsets',
            'ringFlags',
            'count',
            'overflow',
            'requiredCount',
            'invalidCount',
            'approximation'
          ],
          keySpaces: [
            {id: 'ring', label: 'input ring', size: ringCount, indexSpace: '1d'},
            {id: 'vertex', label: 'zone vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'preparedVertex', label: 'prepared vertex', size: vertexCapacity, indexSpace: '1d'}
          ],
          choices: {
            maximumSegmentDegrees: MAXIMUM_SEGMENT_DEGREES,
            maximumPieceCount: MAXIMUM_PIECE_COUNT,
            antimeridian: 'split',
            poleRings: 'cap'
          },
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            const zones = {
              positions: uploads.view('positions', data.positionsLonLat, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const parameters = uploads.view(
              'parameters',
              getGPULonLatPolygonPreparationParameterValues('linear', {
                maximumSegmentDegrees: MAXIMUM_SEGMENT_DEGREES
              }),
              'float32'
            );
            const output = await runExample(graph, {zones, parameters, vertexCapacity});
            return {
              positions: {
                view: zones.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'ringOffsets'})
              },
              ringOffsets: {view: zones.ringOffsets, role: 'intermediate'},
              ringFlags: {
                view: output.ringFlags,
                role: 'intermediate',
                keySpace: 'ring',
                meaning:
                  'per input ring: antimeridian crossing, capped, ambiguous, over-wide, invalid coordinate, clamped bits'
              },
              preparedRingOffsets: {view: output.ringOffsets, role: 'intermediate'},
              count: {view: output.status.count, role: 'intermediate'},
              requiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              invalidCount: {view: output.status.invalidCount, role: 'intermediate'},
              approximation: {view: output.status.approximation, role: 'intermediate'},
              prepared: {
                view: output.prepared,
                role: 'output',
                keySpace: 'preparedVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'preparedRingOffsets'}),
                meaning:
                  'zones densified to 0.005 degrees; main rings then empty east copies (no zone crosses the antimeridian)'
              }
            };
          }
        })
    };
  }
});
