// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-table-compare';
const ZOOM = 17;
/** Category rows of the places dataset: `restaurant_cafe` and `retail`. */
const BEFORE_CATEGORY = 0;
const AFTER_CATEGORY = 7;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellTableCompare',
  title: 'Restaurants versus shops per Quadbin cell in Wicker Park',
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
        addCellBoundariesWithOffsets,
        addQuadbinTable,
        countQuadbinTiles,
        CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
        CELL_DISPLAY_TABLE_HEADROOM
      },
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-table-compare.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // Each category's places, taken whole from the dataset (no resampling).
    const selectCategory = (category: number) => {
      const positions: number[] = [];
      data.category.forEach((value, row) => {
        if (value === category) {
          positions.push(
            data.positionsLonLat[2 * row] as number,
            data.positionsLonLat[2 * row + 1] as number
          );
        }
      });
      return Float32Array.from(positions);
    };
    const beforePositions = selectCategory(BEFORE_CATEGORY);
    const afterPositions = selectCategory(AFTER_CATEGORY);
    // Capacity only: every cell of either category is also a cell of all places.
    const capacity = countQuadbinTiles(data.positionsLonLat, ZOOM) + CELL_DISPLAY_TABLE_HEADROOM;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellTableCompare',
          datasets: [data.id],
          retain: [
            'cells',
            'presence',
            'before',
            'after',
            'delta',
            'zScore',
            'count',
            'overflow',
            'boundaries',
            'boundariesCenters',
            'boundariesOffsets'
          ],
          keySpaces: [
            {id: 'cell', label: 'union cell', size: capacity, indexSpace: '1d'},
            {
              id: 'restaurantPlace',
              label: 'restaurant or cafe',
              size: beforePositions.length / 2,
              indexSpace: '1d'
            },
            {id: 'shopPlace', label: 'shop', size: afterPositions.length / 2, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: capacity * CELL_DISPLAY_QUADBIN_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {measure: 'count', sumScale: 65536, zScore: 'poisson'},
          metadata: {...geographicMetadata(), mappedVariableExtent: 'extensive'},
          async build(graph, uploads) {
            const restaurantPlaces = uploads.view('restaurantPlaces', beforePositions, 'float32x2');
            const shopPlaces = uploads.view('shopPlaces', afterPositions, 'float32x2');
            const restaurants = addQuadbinTable(
              graph,
              restaurantPlaces,
              ZOOM,
              capacity,
              'restaurants'
            );
            const shops = addQuadbinTable(graph, shopPlaces, ZOOM, capacity, 'shops');
            const output = await runExample(graph, {restaurants, shops, capacity});
            // Boundaries from GPUCellGeometry for display only: the union table holds cell keys.
            const display = addCellBoundariesWithOffsets(graph, uploads, output.cells, {
              family: 'quadbin',
              vertexCount: CELL_DISPLAY_QUADBIN_VERTEX_COUNT
            });
            return {
              restaurantPlaces: {
                view: restaurantPlaces,
                role: 'input',
                keySpace: 'restaurantPlace',
                geometry: geographicGeometry('points'),
                meaning: 'restaurant_cafe places (the before table)'
              },
              shopPlaces: {
                view: shopPlaces,
                role: 'input',
                keySpace: 'shopPlace',
                geometry: geographicGeometry('points'),
                meaning: 'retail places (the after table)'
              },
              cells: {
                view: output.cells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning:
                  'union Quadbin cell key (zoom 17) as (low, high) uint32 words; rows past the count are empty (0xffffffff)'
              },
              presence: {
                view: output.presence,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'bit 1: cell in the restaurant table, bit 2: cell in the shop table'
              },
              before: {
                view: output.before,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'restaurants and cafes per cell (a count)'
              },
              after: {
                view: output.after,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'shops per cell (a count)'
              },
              delta: {
                view: output.delta,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'shops minus restaurants per cell (count difference)'
              },
              zScore: {
                view: output.zScore,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'Poisson z-score: delta / sqrt(before + after)'
              },
              count: {
                view: output.status.count,
                role: 'intermediate',
                meaning: 'union cells written'
              },
              overflow: {view: output.status.overflow, role: 'intermediate'},
              boundariesOffsets: {view: display.offsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundariesOffsets',
                  valuesCapture: 'delta'
                }),
                meaning:
                  'union cells valued by shops minus restaurants; boundaries from GPUCellGeometry for display, only valid rows drawn'
              }
            };
          }
        })
    };
  }
});
