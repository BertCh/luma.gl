// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.linear-referencing';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULinearReferencing',
  title: 'Locate 40 Overture places on Wicker Park streets',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-roads + s3-chicago-places',
    source:
      'showcase public/data/chicago-roads (OpenStreetMap) + showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
    licence:
      'ODbL 1.0, (c) OpenStreetMap contributors + CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data, places] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./linear-referencing.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads()),
        import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
      ]);
    // The first 40 rows of the Overture dataset, without resampling or fabricated values.
    const placeCount = 40;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULinearReferencing',
          datasets: [data.id, places.id],
          retain: ['footPoints', 'lineIndices', 'measures', 'distances', 'overflow'],
          keySpaces: [
            {id: 'place', label: 'place', size: placeCount, indexSpace: '1d'},
            {id: 'road', label: 'road', size: data.lineCount, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'vertex',
              size: data.positionsProjected.length / 2,
              indexSpace: '1d'
            }
          ],
          choices: {
            radius: 500,
            candidateCapacity: (data.positionsProjected.length / 2) * placeCount,
            spatialContext: 'native planar (UTM 16N metres)'
          },
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const roads = {
              positions: uploads.view('positions', data.positionsProjected, 'float32x2'),
              lineOffsets: uploads.view('lineOffsets', data.lineOffsets, 'uint32')
            };
            const queryPositions = uploads.view(
              'places',
              places.positionsProjected.slice(0, placeCount * 2),
              'float32x2'
            );
            const radius = uploads.view('radius', new Float32Array([500]), 'float32');
            const output = await runExample(graph, {roads, places: queryPositions, radius});
            return {
              places: {
                view: queryPositions,
                role: 'input',
                keySpace: 'place',
                geometry: projectedGeometry('points'),
                meaning: 'first 40 Overture places in dataset order'
              },
              roads: {
                view: roads.positions,
                role: 'context',
                keySpace: 'vertex',
                geometry: projectedGeometry('lines', {offsetsCapture: 'lineOffsets'})
              },
              lineOffsets: {view: roads.lineOffsets, role: 'intermediate'},
              lineIndices: {
                view: output.lineIndices,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'nearest road index (zero-based); 0xffffffff means unmatched'
              },
              measures: {
                view: output.measures,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'distance along nearest road in m'
              },
              distances: {
                view: output.distances,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'distance to nearest road in m; -1 means unmatched'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'},
              footPoints: {
                view: output.footPoints,
                role: 'output',
                keySpace: 'place',
                geometry: projectedGeometry('points')
              }
            };
          }
        })
    };
  }
});
