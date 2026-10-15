// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {checkDrawnPrefix} from './cell-display-check';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-aggregation';
const ZOOM = 18;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellAggregation',
  title: 'Wicker Park places counted per Quadbin cell at zoom 18',
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
      {runS3Lens, geographicGeometry, geographicMetadata, createFixedOffsets, sumCapture},
      {runExample},
      {
        addCellBoundaries,
        countQuadbinTiles,
        CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
        CELL_DISPLAY_TABLE_HEADROOM
      },
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-aggregation.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    const occupiedCells = countQuadbinTiles(data.positionsLonLat, ZOOM);
    const capacity = occupiedCells + CELL_DISPLAY_TABLE_HEADROOM;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellAggregation',
          datasets: [data.id],
          retain: ['cells', 'counts', 'cellCount', 'boundaries', 'boundariesCenters'],
          keySpaces: [
            {id: 'cell', label: 'cell', size: capacity, indexSpace: '1d'},
            {id: 'place', label: 'place', size: data.pointCount, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: capacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'quadbin', resolution: ZOOM, sumScale: 65536},
          metadata: snapshots => {
            // Draw exactly the rows the GPU wrote.
            checkDrawnPrefix(snapshots, [{capture: 'cellCount', drawn: occupiedCells}]);
            return {
              ...geographicMetadata(),
              mappedVariableExtent: 'extensive',
              // Every place is a finite lon/lat row, so all of them are accepted; counts are summed back.
              sumBefore: data.pointCount,
              sumAfter: sumCapture(snapshots, 'counts') ?? Number.NaN
            };
          },
          async build(graph, uploads) {
            const places = uploads.view('places', data.positionsLonLat, 'float32x2');
            const output = await runExample(graph, {places, capacity});
            // Boundaries from GPUCellGeometry for display only: the table holds cell keys.
            const display = addCellBoundaries(graph, output.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT
            });
            // Draw the occupied prefix of the table; later rows are empty keys.
            const boundaryOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(occupiedCells, CELL_DISPLAY_QUADBIN_VERTEX_COUNT),
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
                keySpace: 'cell',
                meaning:
                  'Quadbin cell key (zoom 18) as (low, high) uint32 words; empty rows are 0xffffffff'
              },
              counts: {
                view: output.counts,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'places per cell (count, an extensive total)'
              },
              cellCount: {view: output.status.count, role: 'intermediate'},
              boundaryOffsets: {view: boundaryOffsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundaryOffsets',
                  valuesCapture: 'counts'
                }),
                meaning:
                  'occupied zoom-18 cells valued by place count; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
