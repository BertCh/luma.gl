// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.point-to-cell';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUPointToCell',
  title: 'Overture places keyed to H3 cells at resolution 9',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-places',
    source:
      'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1), every 4th place',
    licence: 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [
      {runS3Lens, geographicGeometry, geographicMetadata, createFixedOffsets},
      {runExample},
      {addCellBoundaries, selectEveryNthPoint, CELL_DISPLAY_H3_VERTEX_COUNT},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./point-to-cell.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // Every 4th place of the Overture window (about 320 rows), without resampling.
    const positions = selectEveryNthPoint(data.positionsLonLat, 4);
    const placeCount = positions.length / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUPointToCell',
          datasets: [data.id],
          retain: ['cells', 'boundaries', 'boundariesCenters'],
          keySpaces: [
            {id: 'place', label: 'place', size: placeCount, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: placeCount * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'h3', resolution: 9},
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            const places = uploads.view('places', positions, 'float32x2');
            const output = await runExample(graph, {places});
            // Boundaries from GPUCellGeometry for display only: the element writes cell keys.
            const display = addCellBoundaries(graph, output.cells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT
            });
            const boundaryOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(placeCount, CELL_DISPLAY_H3_VERTEX_COUNT),
              'uint32'
            );
            return {
              places: {
                view: places,
                role: 'input',
                keySpace: 'place',
                geometry: geographicGeometry('points')
              },
              cells: {
                view: output.cells,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'H3 cell key of each place (resolution 9) as (low, high) uint32 words'
              },
              boundaryOffsets: {view: boundaryOffsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'boundaryOffsets'}),
                meaning:
                  'resolution-9 cell of each place; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
