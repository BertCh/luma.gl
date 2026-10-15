// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.isobands';

/** Break values every 100 m across the 1074 to 1786 m range of the window. */
const BREAKS = [1100, 1200, 1300, 1400, 1500, 1600, 1700];
/** Bands 3 to 5 (1300 to 1600 m) are emitted; the other five bands keep the record small. */
const FIRST_BAND = 3;
const LAST_BAND = 5;
/** Triangle capacity: about 4800 triangles exist in bands 3 to 5. */
const TRIANGLE_CAPACITY = 5000;
const UNIT = 'm (relative to localOrigin)';

/**
 * Records GPUIsobands on the Dixie Fire window. The DEM file runs north to south, while the
 * contributor places row 0 at the southern edge, so rows are flipped before upload. Triangles are
 * three consecutive vertices each, so their offsets are the multiples of three.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./isobands.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const terrain = await loadDixieDem();
  const entry = getRasterOperatorEntries()['GPUIsobands'];
  const {grid} = terrain;
  const southUp = new Float32Array(terrain.elevation.length);
  for (let row = 0; row < grid.height; row++) {
    const source = (grid.height - 1 - row) * grid.width;
    southUp.set(terrain.elevation.subarray(source, source + grid.width), row * grid.width);
  }
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUIsobands on the Dixie Fire window',
    grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {id: 'triangles', label: 'band triangles', size: TRIANGLE_CAPACITY, indexSpace: '1d'}
    ],
    choices: [
      {stage: 'operator', name: 'breakCount', value: BREAKS.length, sweepable: true},
      {stage: 'operator', name: 'firstBand', value: FIRST_BAND, sweepable: false},
      {stage: 'operator', name: 'lastBand', value: LAST_BAND, sweepable: false},
      {stage: 'support', name: 'localOrigin', value: 'south-west corner', sweepable: false}
    ],
    build: async session => {
      const triangleBands = session.output(
        'isobandsTriangleBands',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'triangles',
          meaning: 'band index of each triangle (band 3 = 1300 m up to 1400 m)'
        },
        TRIANGLE_CAPACITY
      );
      const count = session.output(
        'isobandsTriangleCount',
        'uint32',
        {role: 'intermediate', meaning: 'triangles written (clamped to capacity)'},
        1
      );
      const overflow = session.output(
        'isobandsOverflow',
        'uint32',
        {role: 'intermediate', meaning: '1 when triangles were dropped for lack of capacity'},
        1
      );
      session.input(
        'isobandsTriangleOffsets',
        'uint32',
        Array.from({length: TRIANGLE_CAPACITY + 1}, (_, index) => 3 * index),
        {role: 'intermediate', meaning: 'vertex offsets of the triangles, three vertices each'}
      );
      // Unused triangle capacity stays NaN, so only the written triangles are drawn.
      const triangleValues = session.input(
        'isobandsTriangles',
        'float32',
        new Float32Array(6 * TRIANGLE_CAPACITY).fill(NaN),
        {
          role: 'output',
          components: 2,
          geometry: {
            kind: 'polygons',
            coordinateModel: terrain.coordinateModel,
            unit: UNIT,
            offsetsCapture: 'isobandsTriangleOffsets',
            valuesCapture: 'isobandsTriangleBands'
          },
          meaning: `band triangles of bands 3 to 5, vertices in ${UNIT}`
        }
      );
      const triangles = session.graph.createDataView(triangleValues.buffer, {
        format: 'float32x2',
        length: 3 * TRIANGLE_CAPACITY
      });
      const dem = session.input('isobandsElevation', 'float32', southUp, {
        role: 'input',
        keySpace: 'cells',
        geometry: {
          kind: 'raster',
          shape: [grid.width, grid.height],
          coordinateModel: terrain.coordinateModel,
          cellSize: grid.affine[0],
          unit: UNIT,
          origin: [0, 0],
          rowDirection: 'north'
        },
        meaning: 'elevation (m), rows counted from the southern edge'
      });
      const breaks = session.input('isobandsBreaks', 'float32', BREAKS, false);
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        dem,
        breaks,
        triangles,
        triangleBands,
        count,
        overflow
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUIsobands',
  title: 'Isobands',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
