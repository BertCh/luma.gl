// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.hilbert-keys';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUHilbertKeys',
  title: 'Hilbert curve order of Wicker Park places',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-places',
    source: 'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
    licence: 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [{runS3Lens, projectedGeometry, projectedMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./hilbert-keys.example'),
        import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
      ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUHilbertKeys',
          datasets: [data.id],
          retain: ['keys', 'sortedRows', 'sortedKeys'],
          keySpaces: [
            {id: 'place', label: 'place', size: data.pointCount, indexSpace: '1d'},
            {id: 'rank', label: 'curve rank', size: data.pointCount, indexSpace: '1d'}
          ],
          choices: {order: 6, bounds: 'bounds of the places, reduced on the GPU'},
          metadata: projectedMetadata(data.projected),
          async build(graph, uploads) {
            const places = uploads.view('places', data.positionsProjected, 'float32x2');
            const output = await runExample(graph, {places});
            return {
              places: {
                view: places,
                role: 'input',
                keySpace: 'place',
                geometry: projectedGeometry('points')
              },
              sortedRows: {
                view: output.sortedRows,
                role: 'intermediate',
                keySpace: 'rank',
                meaning: 'place row at each curve rank (stable ascending sort of the keys)'
              },
              sortedKeys: {
                view: output.sortedKeys,
                role: 'intermediate',
                keySpace: 'rank',
                meaning: 'Hilbert index at each curve rank, ascending'
              },
              keys: {
                view: output.keys,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'Hilbert curve index of the place in [0, 4096) at order 6'
              },
              placesByKey: {
                view: places,
                role: 'output',
                keySpace: 'place',
                geometry: projectedGeometry('points', {valuesCapture: 'keys'}),
                meaning: 'places coloured by Hilbert key'
              }
            };
          }
        })
    };
  }
});
