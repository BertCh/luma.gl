// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-measures';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellMeasures',
  title: 'Spherical areas of H3 resolution-9 cells over Wicker Park',
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
      {runS3Lens, geographicGeometry, geographicMetadata, createFixedOffsets},
      {runExample},
      {addCellBoundaries, addH3Cells, selectEveryNthPoint, CELL_DISPLAY_H3_VERTEX_COUNT},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-measures.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // Cells of every 8th place (about 160 rows; repeated cells overdraw).
    const positions = selectEveryNthPoint(data.positionsLonLat, 8);
    const cellCount = positions.length / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellMeasures',
          datasets: [data.id],
          retain: ['areas', 'perimeters', 'pentagons', 'boundaries', 'boundariesCenters'],
          keySpaces: [
            {id: 'cell', label: 'cell (one per place)', size: cellCount, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: cellCount * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'h3', units: 'meters', maximumEdgeCount: 6},
          metadata: geographicMetadata('geodesic'),
          async build(graph, uploads) {
            const places = uploads.view('places', positions, 'float32x2');
            // Cells keyed at H3 resolution 9 (setup).
            const cells = addH3Cells(graph, places, 9, 'placeCells');
            const output = await runExample(graph, {cells});
            // Boundaries from GPUCellGeometry for display only.
            const display = addCellBoundaries(graph, cells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT
            });
            const boundaryOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(cellCount, CELL_DISPLAY_H3_VERTEX_COUNT),
              'uint32'
            );
            return {
              places: {
                view: places,
                role: 'context',
                keySpace: 'cell',
                geometry: geographicGeometry('points'),
                meaning: 'context'
              },
              areas: {
                view: output.areas,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'spherical cell area in m² (WGS84 authalic sphere)'
              },
              perimeters: {
                view: output.perimeters,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'great-circle cell perimeter in m'
              },
              pentagons: {
                view: output.pentagons,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: '1 for the 12 pentagon cells of the resolution, else 0'
              },
              boundaryOffsets: {view: boundaryOffsets, role: 'intermediate'},
              boundaries: {
                view: display.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {
                  offsetsCapture: 'boundaryOffsets',
                  valuesCapture: 'areas'
                }),
                meaning:
                  'cells valued by spherical area in m²; boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
