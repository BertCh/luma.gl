// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {checkDrawnPrefix} from './cell-display-check';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-pyramid';
const ZOOMS = [18, 17, 16];

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellPyramid',
  title: 'Wicker Park place counts at Quadbin zooms 18, 17 and 16',
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
      import('./cell-pyramid.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    const occupied = ZOOMS.map(zoom => countQuadbinTiles(data.positionsLonLat, zoom));
    const capacities = occupied.map(count => count + CELL_DISPLAY_TABLE_HEADROOM);
    const finestCapacity = capacities[0] as number;
    const coarsestCapacity = capacities[2] as number;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellPyramid',
          datasets: [data.id],
          retain: [
            'level0Cells',
            'level0Counts',
            'level2Cells',
            'level2Counts',
            'levelCounts',
            'finestBoundaries',
            'finestBoundariesCenters',
            'boundaries',
            'boundariesCenters'
          ],
          keySpaces: [
            {id: 'cell', label: 'zoom-16 cell', size: coarsestCapacity, indexSpace: '1d'},
            {id: 'finestCell', label: 'zoom-18 cell', size: finestCapacity, indexSpace: '1d'},
            {id: 'level', label: 'pyramid level', size: ZOOMS.length, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'zoom-16 boundary vertex',
              size: coarsestCapacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            },
            {
              id: 'finestBoundaryVertex',
              label: 'zoom-18 boundary vertex',
              size: finestCapacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'quadbin', resolutions: ZOOMS.join(', '), sumScale: 65536},
          metadata: snapshots => {
            // Draw exactly the rows the GPU wrote.
            checkDrawnPrefix(snapshots, [
              {capture: 'levelCounts', drawn: occupied[0] as number, index: 0},
              {capture: 'levelCounts', drawn: occupied[2] as number, index: 2}
            ]);
            return {
              ...geographicMetadata(),
              mappedVariableExtent: 'extensive',
              // Totals compared between two individual levels, never summed across levels.
              sumBefore: sumCapture(snapshots, 'level0Counts') ?? Number.NaN,
              sumAfter: sumCapture(snapshots, 'level2Counts') ?? Number.NaN
            };
          },
          async build(graph, uploads) {
            const places = uploads.view('places', data.positionsLonLat, 'float32x2');
            const output = await runExample(graph, {places, capacities});
            const finest = output.levels[0];
            const coarsest = output.levels[2];
            if (!finest || !coarsest) throw new Error('pyramid levels missing');
            const finestDisplay = addCellBoundaries(graph, finest.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              name: 'finestBoundaries'
            });
            const coarsestDisplay = addCellBoundaries(graph, coarsest.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT
            });
            // Boundaries from GPUCellGeometry for display only; draw each level's occupied prefix.
            const finestOffsets = uploads.view(
              'finestBoundaryOffsets',
              createFixedOffsets(occupied[0] as number, CELL_DISPLAY_QUADBIN_VERTEX_COUNT),
              'uint32'
            );
            const coarsestOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(occupied[2] as number, CELL_DISPLAY_QUADBIN_VERTEX_COUNT),
              'uint32'
            );
            return {
              finestBoundaries: {
                view: finestDisplay.boundaries,
                role: 'input',
                keySpace: 'finestBoundaryVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'finestBoundaryOffsets'}),
                meaning:
                  'occupied zoom-18 cells (pyramid level 0); boundaries from GPUCellGeometry for display'
              },
              finestBoundaryOffsets: {view: finestOffsets, role: 'intermediate'},
              level0Cells: {
                view: finest.cells,
                role: 'intermediate',
                keySpace: 'finestCell',
                meaning: 'zoom-18 Quadbin cell key as (low, high) uint32 words'
              },
              level0Counts: {
                view: finest.counts,
                role: 'intermediate',
                keySpace: 'finestCell',
                meaning: 'places per zoom-18 cell (count)'
              },
              level2Cells: {
                view: coarsest.cells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'zoom-16 Quadbin cell key as (low, high) uint32 words'
              },
              level2Counts: {
                view: coarsest.counts,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'places per zoom-16 cell (count)'
              },
              levelCounts: {
                view: output.levelCounts,
                role: 'intermediate',
                keySpace: 'level',
                meaning: 'occupied cells per level, finest (zoom 18) first'
              },
              boundaryOffsets: {view: coarsestOffsets, role: 'intermediate'},
              boundaries: {
                view: coarsestDisplay.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundaryOffsets',
                  valuesCapture: 'level2Counts'
                }),
                meaning:
                  'zoom-16 pyramid level valued by place count; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
