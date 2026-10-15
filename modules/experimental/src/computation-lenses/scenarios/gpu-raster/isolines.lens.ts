// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.isolines';

/** Contour levels every 100 m across the 1074 to 1786 m range of the window. */
const LEVELS = [1100, 1200, 1300, 1400, 1500, 1600, 1700];
/** Segment capacity: 1270 segments exist at these levels. */
const SEGMENT_CAPACITY = 1300;
/** Vertex capacity: 26 polylines need 1296 vertices before collapsing repeats. */
const VERTEX_CAPACITY = 1400;
/** Polylines shown: 26 exist, a few spare rows are captured. */
const SHOWN_POLYLINES = 32;
const UNIT = 'm (relative to localOrigin)';

/**
 * Records GPUIsolines on the Dixie Fire window. The DEM file runs north to south, while the
 * contributor places row 0 at the southern edge, so rows are flipped before upload.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./isolines.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const terrain = await loadDixieDem();
  const entry = getRasterOperatorEntries()['GPUIsolines'];
  const {grid} = terrain;
  const southUp = new Float32Array(terrain.elevation.length);
  for (let row = 0; row < grid.height; row++) {
    const source = (grid.height - 1 - row) * grid.width;
    southUp.set(terrain.elevation.subarray(source, source + grid.width), row * grid.width);
  }
  const cellSize = grid.affine[0];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUIsolines on the Dixie Fire window',
    grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {id: 'polylines', label: 'contour polylines', size: SHOWN_POLYLINES, indexSpace: '1d'},
      {id: 'vertices', label: 'polyline vertices', size: VERTEX_CAPACITY, indexSpace: '1d'}
    ],
    choices: [
      {stage: 'operator', name: 'levelCount', value: LEVELS.length, sweepable: true},
      {stage: 'support', name: 'localOrigin', value: 'south-west corner', sweepable: false}
    ],
    build: async session => {
      const polylineLevels = session.output(
        'isolinesPolylineLevelRows',
        'uint32',
        false,
        SEGMENT_CAPACITY
      );
      const polylineClosed = session.output(
        'isolinesPolylineClosedRows',
        'uint32',
        false,
        SEGMENT_CAPACITY
      );
      const polylineOffsets = session.output(
        'isolinesPolylineOffsetRows',
        'uint32',
        false,
        SEGMENT_CAPACITY + 1
      );
      // The first rows only: a polyline owns one row, so the rest are empty capacity.
      const head = (view: typeof polylineLevels, length: number) =>
        session.graph.createDataView(view.buffer, {format: 'uint32', length});
      session.capture('isolinesPolylineLevels', head(polylineLevels, SHOWN_POLYLINES), {
        role: 'intermediate',
        keySpace: 'polylines',
        meaning: 'contour level index of each polyline (0 = 1100 m)'
      });
      session.capture('isolinesPolylineClosed', head(polylineClosed, SHOWN_POLYLINES), {
        role: 'intermediate',
        keySpace: 'polylines',
        meaning: '1 for a closed ring, 0 for an open chain'
      });
      session.capture('isolinesPolylineOffsets', head(polylineOffsets, SHOWN_POLYLINES + 1), {
        role: 'intermediate',
        meaning: 'vertex offsets of each polyline'
      });
      const polylineCount = session.output(
        'isolinesPolylineCount',
        'uint32',
        {role: 'intermediate', meaning: 'complete polylines written'},
        1
      );
      const vertexCount = session.output('isolinesVertexCount', 'uint32', false, 1);
      const count = session.output(
        'isolinesCount',
        'uint32',
        {role: 'intermediate', meaning: 'segments written (clamped to capacity)'},
        1
      );
      const overflow = session.output(
        'isolinesOverflow',
        'uint32',
        {role: 'intermediate', meaning: '1 when segments were dropped for lack of capacity'},
        1
      );
      const polylineOverflow = session.output('isolinesPolylineOverflow', 'uint32', false, 1);
      const segmentValues = session.output(
        'isolinesSegmentValues',
        'float32',
        false,
        4 * SEGMENT_CAPACITY
      );
      const segments = session.graph.createDataView(segmentValues.buffer, {
        format: 'float32x4',
        length: SEGMENT_CAPACITY
      });
      const segmentLevels = session.output(
        'isolinesSegmentLevels',
        'uint32',
        false,
        SEGMENT_CAPACITY
      );
      // Unused vertex capacity stays NaN, so only the written vertices are drawn.
      const vertexValues = session.input(
        'isolinesVertices',
        'float32',
        new Float32Array(2 * VERTEX_CAPACITY).fill(NaN),
        {
          role: 'output',
          components: 2,
          geometry: {
            kind: 'lines',
            coordinateModel: terrain.coordinateModel,
            unit: UNIT,
            offsetsCapture: 'isolinesPolylineOffsets',
            valuesCapture: 'isolinesPolylineLevels'
          },
          meaning: `contour polylines every 100 m, vertices in ${UNIT}`
        }
      );
      const vertices = session.graph.createDataView(vertexValues.buffer, {
        format: 'float32x2',
        length: VERTEX_CAPACITY
      });
      const dem = session.input('isolinesElevation', 'float32', southUp, {
        role: 'input',
        keySpace: 'cells',
        geometry: {
          kind: 'raster',
          shape: [grid.width, grid.height],
          coordinateModel: terrain.coordinateModel,
          cellSize,
          unit: UNIT,
          origin: [0, 0],
          rowDirection: 'north'
        },
        meaning: 'elevation (m), rows counted from the southern edge'
      });
      const levels = session.input('isolinesLevels', 'float32', LEVELS, false);
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        dem,
        levels,
        segments,
        segmentLevels,
        polylines: {
          vertices,
          polylineOffsets,
          polylineLevels,
          polylineClosed,
          polylineCount,
          vertexCount
        },
        count,
        overflow,
        polylineOverflow
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUIsolines',
  title: 'Isolines',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
