// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-geometry';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellGeometry',
  title: 'Boundaries and centres of H3 cells over Wicker Park',
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
      {selectEveryNthPoint, CELL_DISPLAY_H3_VERTEX_COUNT},
      {GPUPointToCell},
      {createTransientView},
      {Buffer},
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-geometry.example'),
      import('./cell-display'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('@luma.gl/gpgpu/gpu-core'),
      import('@luma.gl/core'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // Cells of every 4th place (about 320 rows; repeated cells overdraw), without resampling.
    const positions = selectEveryNthPoint(data.positionsLonLat, 4);
    const placeCount = positions.length / 2;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellGeometry',
          datasets: [data.id],
          retain: ['cells', 'centers', 'boundaries', 'vertexCounts'],
          keySpaces: [
            {id: 'cell', label: 'cell (one per place)', size: placeCount, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'cell boundary vertex',
              size: placeCount * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {family: 'h3', maximumVertexCount: CELL_DISPLAY_H3_VERTEX_COUNT},
          metadata: geographicMetadata(),
          async build(graph, uploads) {
            const places = uploads.view('places', positions, 'float32x2');
            // Cell keys for the figure only (the element's input is a cell column).
            const cells = createTransientView(
              graph,
              'cells',
              'uint32x2',
              placeCount,
              Buffer.STORAGE | Buffer.COPY_SRC
            );
            graph.add(
              new GPUPointToCell({
                id: 'place-cells',
                family: 'h3',
                resolution: 9,
                positions: places,
                output: {cells}
              })
            );
            const output = await runExample(graph, {cells});
            const boundaryOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(placeCount, CELL_DISPLAY_H3_VERTEX_COUNT),
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
              cells: {
                view: cells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'H3 cell key (resolution 9) as (low, high) uint32 words'
              },
              centers: {
                view: output.centers,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'cell centre [longitude, latitude] in degrees'
              },
              vertexCounts: {
                view: output.vertexCounts,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'boundary vertices written per cell (6 for a hexagon)'
              },
              boundaryOffsets: {view: boundaryOffsets, role: 'intermediate'},
              boundaries: {
                view: output.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'boundaryOffsets'}),
                meaning: 'H3 cell boundary, 6 vertices per cell, longitude/latitude degrees'
              }
            };
          }
        })
    };
  }
});
