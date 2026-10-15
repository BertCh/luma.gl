// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.grid-generator';
/** Community area 24 (West Town) is the extent the intersection mask tests against. */
const EXTENT_AREA = 'area 24';
/** Hexagon grid in UTM 16N metres: width between flat sides, origin just outside the places. */
const GRID = {minX: 3560, minY: 9200, cellWidth: 120, columns: 13, rows: 13};

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGridGenerator',
  title: 'Hexagon grid over Wicker Park',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-places + s3-chicago-community-areas',
    source:
      'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1) + showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
    licence:
      'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces) + City of Chicago Data Portal Terms of Use'
  },
  load: async () => {
    const [
      {runS3Lens, projectedGeometry, projectedMetadata, createFixedOffsets},
      {runExample},
      places,
      areas,
      {getGPUGridCellCount, getGPUGridGeneratorParameterValues, getGPUGridVerticesPerCell}
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./grid-generator.example'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces()),
      import('./s3-lens-data').then(module => module.loadS3ChicagoCommunityAreas()),
      import('@luma.gl/experimental/gpu-spatial-analysis')
    ]);
    // The community-area dataset has exactly one polygon with one ring per feature.
    assert(areas.ringOffsets.length === areas.featureCount + 1);
    const feature = areas.names.indexOf(EXTENT_AREA);
    assert(feature >= 0);
    const areaPositions = areas.positionsProjected.slice(
      2 * areas.ringOffsets[feature],
      2 * areas.ringOffsets[feature + 1]
    );
    const {minX, minY, cellWidth, columns, rows} = GRID;
    const cellCount = getGPUGridCellCount('hex', columns, rows);
    const cornerCount = getGPUGridVerticesPerCell('hex');
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGridGenerator',
          datasets: [places.id, areas.id],
          retain: ['cellCorners', 'centers', 'intersects'],
          keySpaces: [
            {
              id: 'cell',
              label: 'hexagon, rows from south',
              size: cellCount,
              indexSpace: '2d',
              shape: [columns, rows]
            },
            {id: 'place', label: 'place', size: places.pointCount, indexSpace: '1d'},
            {
              id: 'areaVertex',
              label: 'West Town vertex',
              size: areaPositions.length / 2,
              indexSpace: '1d'
            },
            {id: 'corner', label: 'hexagon corner', size: cellCount * cornerCount, indexSpace: '1d'}
          ],
          choices: {
            gridType: 'hex',
            hexOrientation: 'pointy',
            columns,
            rows,
            minX,
            minY,
            cellWidth,
            // Ignored by hexagon grids (the pitch follows from the width).
            cellHeight: cellWidth
          },
          metadata: projectedMetadata(places.projected),
          async build(graph, uploads) {
            const parameters = uploads.view(
              'parameters',
              getGPUGridGeneratorParameterValues({minX, minY, cellWidth, cellHeight: cellWidth}),
              'float32'
            );
            const placePositions = uploads.view('places', places.positionsProjected, 'float32x2');
            const area = {
              positions: uploads.view('areaPositions', areaPositions, 'float32x2'),
              ringOffsets: uploads.view(
                'areaRingOffsets',
                Uint32Array.of(0, areaPositions.length / 2),
                'uint32'
              ),
              polygonOffsets: uploads.view('areaPolygonOffsets', Uint32Array.of(0, 1), 'uint32'),
              featureOffsets: uploads.view('areaFeatureOffsets', Uint32Array.of(0, 1), 'uint32')
            };
            // Display index for the hexagon corners (6 per cell); not data.
            const cornerOffsets = uploads.view(
              'cornerOffsets',
              createFixedOffsets(cellCount, cornerCount),
              'uint32'
            );
            const output = await runExample(graph, {columns, rows, parameters, area});
            return {
              places: {
                view: placePositions,
                role: 'context',
                keySpace: 'place',
                geometry: projectedGeometry('points')
              },
              area: {
                view: area.positions,
                role: 'input',
                keySpace: 'areaVertex',
                geometry: projectedGeometry('polygons', {offsetsCapture: 'areaRingOffsets'}),
                meaning: 'community area 24 (West Town), the extent the mask tests against'
              },
              areaRingOffsets: {view: area.ringOffsets, role: 'intermediate'},
              cornerOffsets: {view: cornerOffsets, role: 'intermediate'},
              centers: {
                view: output.centers,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'hexagon centre [x, y] in m (mean of its six corners)'
              },
              intersects: {
                view: output.intersects,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: '1 when the hexagon touches the area, else 0'
              },
              cellCorners: {
                view: output.positions,
                role: 'output',
                keySpace: 'corner',
                geometry: projectedGeometry('polygons', {
                  offsetsCapture: 'cornerOffsets',
                  valuesCapture: 'intersects'
                }),
                meaning: 'hexagon corners, six per cell, coloured by the area-intersection flag'
              }
            };
          }
        })
    };
  }
});
