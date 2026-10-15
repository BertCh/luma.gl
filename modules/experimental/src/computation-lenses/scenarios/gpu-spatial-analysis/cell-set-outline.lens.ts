// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-set-outline';
/** Rows of the cover and of the aggregated cell table (about 160 cells are expected). */
const CELL_CAPACITY = 512;
const SEGMENT_CAPACITY = 512;
const RING_CAPACITY = 32;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellSetOutline',
  title: 'Outline of the H3 cells covering Chicago community areas',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-community-areas',
    source: 'showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence: 'City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [
      {runS3Lens, geographicGeometry, geographicMetadata},
      {runExample},
      {runExample: runCover},
      {addCellBoundariesWithOffsets, createCellTable, CELL_DISPLAY_H3_VERTEX_COUNT},
      {GPUCellAggregation},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-set-outline.example'),
      import('./cell-cover.example'),
      import('./cell-display'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
    ]);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellSetOutline',
          datasets: [data.id],
          retain: [
            'tableCells',
            'tableCount',
            'segmentRows',
            'edgeIndices',
            'segmentCount',
            'outlineOverflow',
            'ringCount',
            'ringOffsets',
            'ringPositions',
            'cellBoundaries',
            'cellBoundariesCenters',
            'cellBoundariesOffsets'
          ],
          keySpaces: [
            {id: 'segment', label: 'exposed cell edge', size: SEGMENT_CAPACITY, indexSpace: '1d'},
            {id: 'cell', label: 'cell table row', size: CELL_CAPACITY, indexSpace: '1d'},
            {
              id: 'cellVertex',
              label: 'cell boundary vertex',
              size: CELL_CAPACITY * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            },
            {
              id: 'ringVertex',
              label: 'outline ring vertex',
              size: SEGMENT_CAPACITY + RING_CAPACITY,
              indexSpace: '1d'
            }
          ],
          // 5e-5 degrees is the element's own ring-matching preset (GPU_CELL_SET_OUTLINE_VERTEX_TOLERANCE).
          choices: {family: 'h3', 'rings.vertexTolerance': 5e-5, 'rings.normalizeWinding': true},
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            // Setup: cover the areas with H3 cells, then aggregate them into the sorted, distinct
            // table that the outline requires. Unwritten cover rows are empty keys (uploaded as zero).
            const areas = {
              positions: uploads.view('positions', data.positionsLonLat, 'float32x2'),
              ringOffsets: uploads.view('areaRingOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const cellSlots = uploads.view(
              'cellSlots',
              new Uint32Array(2 * CELL_CAPACITY),
              'uint32x2'
            );
            const cover = await runCover(graph, {areas, cellSlots});
            const table = createCellTable(graph, 'table', CELL_CAPACITY);
            graph.add(
              new GPUCellAggregation({
                id: 'cover-table',
                family: 'h3',
                resolution: 8,
                cells: cover.cells,
                output: table
              })
            );
            const output = await runExample(graph, {
              cells: table.cells,
              count: table.status.count,
              segmentCapacity: SEGMENT_CAPACITY,
              ringCapacity: RING_CAPACITY
            });
            // Boundaries from GPUCellGeometry for display only: the table holds cell keys.
            const display = addCellBoundariesWithOffsets(graph, uploads, table.cells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT,
              name: 'cellBoundaries'
            });
            return {
              cellBoundaries: {
                view: display.boundaries,
                role: 'input',
                keySpace: 'cellVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'cellBoundariesOffsets'}),
                meaning:
                  'H3 resolution-8 cells covering the 17 community areas (GPUCellCover, then GPUCellAggregation); boundaries from GPUCellGeometry for display'
              },
              cellBoundariesOffsets: {view: display.offsets, role: 'intermediate'},
              tableCount: {
                view: table.status.count,
                role: 'intermediate',
                meaning: 'distinct cells in the sorted table'
              },
              segmentRows: {
                view: output.rows,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'table row of the cell that owns each exposed edge'
              },
              edgeIndices: {
                view: output.edgeIndices,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'edge index within the owning hexagon (0 to 5)'
              },
              segmentCount: {
                view: output.status.count,
                role: 'intermediate',
                meaning: 'exposed edges written'
              },
              outlineOverflow: {view: output.status.overflow, role: 'intermediate'},
              ringCount: {
                view: output.rings.status.count,
                role: 'intermediate',
                meaning: 'closed rings written'
              },
              ringOffsets: {view: output.ringOffsets, role: 'intermediate'},
              ringPositions: {
                view: output.positions,
                role: 'output',
                keySpace: 'ringVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning:
                  'closed rings of the exposed edges (shells counter-clockwise, holes clockwise), longitude/latitude degrees; unused ring slots are empty'
              }
            };
          }
        })
    };
  }
});
