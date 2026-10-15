// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-topology';
const DISK_SIZE = 61; // 3k(k+1)+1 cells for k = 4

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellTopology',
  title: 'H3 grid disk of radius 4 around the central Wicker Park place',
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
      {runS3Lens, geographicGeometry, geographicMetadata, createFixedOffsets},
      {runExample},
      {addCellBoundaries, addH3Cells, findCentralPlaceIndex, CELL_DISPLAY_H3_VERTEX_COUNT},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-topology.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // One input row: the place nearest the mean position of the 1,275 places.
    const central = findCentralPlaceIndex(data.positionsLonLat);
    const positions = data.positionsLonLat.slice(2 * central, 2 * central + 2);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellTopology',
          datasets: [data.id],
          retain: ['diskCells', 'distances', 'counts', 'boundaries', 'boundariesCenters'],
          keySpaces: [
            {id: 'cell', label: 'disk cell', size: DISK_SIZE, indexSpace: '1d'},
            {id: 'place', label: 'input place', size: 1, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: DISK_SIZE * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'h3', 'operation.type': 'disk', 'operation.k': 4},
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            const place = uploads.view('place', positions, 'float32x2');
            // The input cell: the place keyed at H3 resolution 9 (setup).
            const cells = addH3Cells(graph, place, 9, 'placeCells');
            const output = await runExample(graph, {cells});
            // Boundaries from GPUCellGeometry for display only.
            const display = addCellBoundaries(graph, output.diskCells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT
            });
            const boundaryOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(DISK_SIZE, CELL_DISPLAY_H3_VERTEX_COUNT),
              'uint32'
            );
            return {
              place: {
                view: place,
                role: 'input',
                keySpace: 'place',
                geometry: geographicGeometry('points')
              },
              diskCells: {
                view: output.diskCells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning:
                  'H3 cell key within 4 grid steps, (low, high) uint32 words, ordered by (distance, key)'
              },
              distances: {
                view: output.distances,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'grid steps from the input cell'
              },
              counts: {
                view: output.counts,
                role: 'intermediate',
                keySpace: 'place',
                meaning: 'valid cells written for the input row'
              },
              boundaryOffsets: {view: boundaryOffsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundaryOffsets',
                  valuesCapture: 'distances'
                }),
                meaning:
                  'disk cells valued by grid distance from the input cell; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
