// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {checkDrawnPrefix} from './cell-display-check';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-rollup';
const SOURCE_ZOOM = 18;
const PARENT_ZOOM = 16;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellRollup',
  title: 'Zoom-18 place counts rolled up to zoom-16 parent cells',
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
        addQuadbinTable,
        countQuadbinTiles,
        CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
        CELL_DISPLAY_TABLE_HEADROOM
      },
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-rollup.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    const sourceCells = countQuadbinTiles(data.positionsLonLat, SOURCE_ZOOM);
    const parentCells = countQuadbinTiles(data.positionsLonLat, PARENT_ZOOM);
    const sourceCapacity = sourceCells + CELL_DISPLAY_TABLE_HEADROOM;
    const parentCapacity = parentCells + CELL_DISPLAY_TABLE_HEADROOM;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellRollup',
          datasets: [data.id],
          retain: [
            'sourceCells',
            'sourceCounts',
            'cells',
            'counts',
            'cellCount',
            'sourceBoundaries',
            'sourceBoundariesCenters',
            'boundaries',
            'boundariesCenters'
          ],
          keySpaces: [
            {id: 'cell', label: 'parent cell', size: parentCapacity, indexSpace: '1d'},
            {id: 'sourceCell', label: 'source cell', size: sourceCapacity, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'parent boundary vertex',
              size: parentCapacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            },
            {
              id: 'sourceBoundaryVertex',
              label: 'source boundary vertex',
              size: sourceCapacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {
            family: 'quadbin',
            sourceResolution: SOURCE_ZOOM,
            resolution: PARENT_ZOOM,
            sumScale: 65536
          },
          metadata: snapshots => {
            // Draw exactly the rows the GPU wrote.
            checkDrawnPrefix(snapshots, [{capture: 'cellCount', drawn: parentCells}]);
            return {
              ...geographicMetadata(),
              mappedVariableExtent: 'extensive',
              sumBefore: sumCapture(snapshots, 'sourceCounts') ?? Number.NaN,
              sumAfter: sumCapture(snapshots, 'counts') ?? Number.NaN
            };
          },
          async build(graph, uploads) {
            const places = uploads.view('places', data.positionsLonLat, 'float32x2');
            // The zoom-18 table the roll-up reads: GPUCellAggregation of the places (setup).
            const source = addQuadbinTable(graph, places, SOURCE_ZOOM, sourceCapacity);
            const output = await runExample(graph, {source, capacity: parentCapacity});
            const sourceDisplay = addCellBoundaries(graph, source.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              name: 'sourceBoundaries'
            });
            const parentDisplay = addCellBoundaries(graph, output.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT
            });
            // Boundaries from GPUCellGeometry for display only; draw each table's occupied prefix.
            const sourceOffsets = uploads.view(
              'sourceBoundaryOffsets',
              createFixedOffsets(sourceCells, CELL_DISPLAY_QUADBIN_VERTEX_COUNT),
              'uint32'
            );
            const parentOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(parentCells, CELL_DISPLAY_QUADBIN_VERTEX_COUNT),
              'uint32'
            );
            return {
              sourceBoundaries: {
                view: sourceDisplay.boundaries,
                role: 'input',
                keySpace: 'sourceBoundaryVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'sourceBoundaryOffsets'}),
                meaning:
                  'occupied zoom-18 source cells; boundaries from GPUCellGeometry for display'
              },
              sourceBoundaryOffsets: {view: sourceOffsets, role: 'intermediate'},
              sourceCells: {
                view: source.cells,
                role: 'intermediate',
                keySpace: 'sourceCell',
                meaning: 'zoom-18 Quadbin cell key as (low, high) uint32 words'
              },
              sourceCounts: {
                view: source.counts,
                role: 'intermediate',
                keySpace: 'sourceCell',
                meaning: 'places per zoom-18 source cell (count)'
              },
              cells: {
                view: output.cells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'zoom-16 parent Quadbin cell key as (low, high) uint32 words'
              },
              counts: {
                view: output.counts,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'places per parent cell (integer sum of its child counts)'
              },
              cellCount: {view: output.status.count, role: 'intermediate'},
              boundaryOffsets: {view: parentOffsets, role: 'intermediate'},
              boundaries: {
                view: parentDisplay.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundaryOffsets',
                  valuesCapture: 'counts'
                }),
                meaning:
                  'zoom-16 parent cells valued by place count; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
