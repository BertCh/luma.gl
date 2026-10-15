// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-cover';
/** Cell rows the cover may write; about 160 cells are expected at H3 resolution 8. */
const CELL_CAPACITY = 512;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellCover',
  title: 'Chicago community areas covered by H3 resolution-8 cells',
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
      {addCellBoundariesWithOffsets, CELL_DISPLAY_H3_VERTEX_COUNT},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-cover.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas())
    ]);
    const vertexCount = data.positionsLonLat.length / 2;
    const ringCount = data.ringOffsets.length - 1;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellCover',
          datasets: [data.id],
          retain: [
            'featureIds',
            'count',
            'requiredCount',
            'overflow',
            'candidateOverflow',
            'boundaries',
            'boundariesCenters',
            'boundariesOffsets'
          ],
          keySpaces: [
            {id: 'cell', label: 'covering cell', size: CELL_CAPACITY, indexSpace: '1d'},
            {id: 'vertex', label: 'area ring vertex', size: vertexCount, indexSpace: '1d'},
            {id: 'ring', label: 'area ring', size: ringCount, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: CELL_CAPACITY * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {
            family: 'h3',
            resolution: 8,
            containment: 'center',
            candidateEnumeration: 'auto'
          },
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            const areas = {
              positions: uploads.view('positions', data.positionsLonLat, 'float32x2'),
              ringOffsets: uploads.view('ringOffsets', data.ringOffsets, 'uint32'),
              polygonOffsets: uploads.view('polygonOffsets', data.polygonOffsets, 'uint32'),
              featureOffsets: uploads.view('featureOffsets', data.featureOffsets, 'uint32')
            };
            const cellSlots = uploads.view(
              'cellSlots',
              new Uint32Array(2 * CELL_CAPACITY),
              'uint32x2'
            );
            const output = await runExample(graph, {areas, cellSlots});
            // Boundaries from GPUCellGeometry for display only; the GPU-derived offsets draw the valid rows.
            const display = addCellBoundariesWithOffsets(graph, uploads, output.cells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT
            });
            return {
              positions: {
                view: areas.positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'ringOffsets'}),
                meaning: '17 adjacent Chicago community areas, one ring each'
              },
              ringOffsets: {view: areas.ringOffsets, role: 'intermediate', keySpace: 'ring'},
              featureIds: {
                view: output.featureIds,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'area (feature row) that each covering cell belongs to'
              },
              cells: {
                view: output.cells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning:
                  'H3 cell key (resolution 8) as (low, high) uint32 words; rows past the count are empty'
              },
              count: {view: output.status.count, role: 'intermediate', meaning: 'cells written'},
              requiredCount: {view: output.status.requiredCount, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              candidateOverflow: {view: output.status.candidateOverflow, role: 'intermediate'},
              boundariesOffsets: {view: display.offsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundariesOffsets',
                  valuesCapture: 'featureIds'
                }),
                meaning:
                  'cells whose centre is inside an area, valued by area row; boundaries from GPUCellGeometry for display, only valid rows drawn'
              }
            };
          }
        })
    };
  }
});
