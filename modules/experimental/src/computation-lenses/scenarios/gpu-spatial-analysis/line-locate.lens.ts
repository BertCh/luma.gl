// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.line-locate';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULineLocate',
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
        import('./line-locate.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoRoads()),
        import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
      ]);
    // The first 40 rows of the Overture dataset, without resampling or fabricated values.
    const placeCount = 40;
    const [{runExample: referencePlaces}, {getGPULineLocateParameterValues}] = await Promise.all([
      import('./linear-referencing.example'),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPULineLocate',
          datasets: [data.id, places.id],
          retain: [
            'footPoints',
            'lineIndices',
            'measures',
            'distances',
            'overflow',
            'locatedPlaces',
            'statuses'
          ],
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
            spatialContext: 'native planar (UTM 16N metres)',
            measureMode: 'distance',
            measureScale: 1,
            measureOffset: 0
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
            const referenced = await referencePlaces(graph, {
              roads,
              places: queryPositions,
              radius
            });
            const parameters = uploads.view(
              'parameters',
              getGPULineLocateParameterValues({measureScale: 1, measureOffset: 0}),
              'float32'
            );
            const output = await runExample(graph, {
              roads,
              eventLineIndices: referenced.lineIndices,
              eventMeasures: referenced.measures,
              parameters
            });
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
                view: referenced.lineIndices,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'GPU nearest road index; 0xffffffff means unmatched'
              },
              measures: {
                view: referenced.measures,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'GPU nearest-road measure in m used as the event measure'
              },
              distances: {
                view: referenced.distances,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'distance from original place to nearest road in m; -1 means unmatched'
              },
              footPoints: {
                view: referenced.footPoints,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'nearest-road foot point in UTM 16N m'
              },
              overflow: {view: referenced.status.overflow, role: 'intermediate'},
              statuses: {
                view: output.statuses,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'event status code (unitless): 0 valid, 1 clamped, 2 invalid'
              },
              locatedPlaces: {
                view: output.locatedPlaces,
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
