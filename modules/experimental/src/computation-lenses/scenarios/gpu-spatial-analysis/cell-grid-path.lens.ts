// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-grid-path';
const MAXIMUM_PATH_LENGTH = 16;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellGridPath',
  title: 'H3 grid paths from 7 Wicker Park places to the central place',
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
      {runS3Lens, geographicGeometry, geographicMetadata},
      {runExample},
      {
        addCellBoundaries,
        addH3Cells,
        findCentralPlaceIndex,
        selectEveryNthPoint,
        CELL_DISPLAY_H3_VERTEX_COUNT
      },
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-grid-path.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // Origins: every 200th place (7 rows). Destination: the place nearest the mean position.
    const origins = selectEveryNthPoint(data.positionsLonLat, 200);
    const pairCount = origins.length / 2;
    const central = findCentralPlaceIndex(data.positionsLonLat);
    const hub = data.positionsLonLat.slice(2 * central, 2 * central + 2);
    const hubRows = new Float32Array(2 * pairCount);
    for (let pair = 0; pair < pairCount; pair++) hubRows.set(hub, 2 * pair);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellGridPath',
          datasets: [data.id],
          retain: ['distances', 'counts', 'pathCells', 'pathBoundariesCenters'],
          keySpaces: [
            {id: 'pair', label: 'origin/destination pair', size: pairCount, indexSpace: '1d'},
            {
              id: 'pathCell',
              label: 'path cell slot',
              size: pairCount * MAXIMUM_PATH_LENGTH,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'h3', maximumPathLength: MAXIMUM_PATH_LENGTH},
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            const originPlaces = uploads.view('originPlaces', origins, 'float32x2');
            const destinationPlaces = uploads.view('destinationPlaces', hubRows, 'float32x2');
            const hubPlace = uploads.view('hubPlace', hub, 'float32x2');
            // Origin and destination cells: the places keyed at H3 resolution 9 (setup).
            const originCells = addH3Cells(graph, originPlaces, 9, 'originCells');
            const destinationCells = addH3Cells(graph, destinationPlaces, 9, 'destinationCells');
            const output = await runExample(graph, {
              origins: originCells,
              destinations: destinationCells
            });
            // Path cell centres from GPUCellGeometry for display; unused slots decode to NaN and are not drawn.
            const display = addCellBoundaries(graph, output.pathCells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT,
              name: 'pathBoundaries'
            });
            return {
              originPlaces: {
                view: originPlaces,
                role: 'input',
                keySpace: 'pair',
                geometry: geographicGeometry('points'),
                meaning: 'origin places (every 200th of the 1,275)'
              },
              hubPlace: {
                view: hubPlace,
                role: 'context',
                geometry: geographicGeometry('points'),
                meaning: 'context: shared destination place'
              },
              distances: {
                view: output.distances,
                role: 'intermediate',
                keySpace: 'pair',
                meaning: 'H3 grid distance in cells'
              },
              counts: {
                view: output.counts,
                role: 'intermediate',
                keySpace: 'pair',
                meaning: 'cells on the path (distance + 1), 0 when no path exists'
              },
              pathCells: {
                view: output.pathCells,
                role: 'intermediate',
                keySpace: 'pathCell',
                meaning: 'path cell key as (low, high) uint32 words, 16 slots per pair, zero padded'
              },
              pathBoundariesCenters: {
                view: display.centers,
                role: 'output',
                keySpace: 'pathCell',
                geometry: geographicGeometry('points'),
                meaning: 'centres of the grid-path cells from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
