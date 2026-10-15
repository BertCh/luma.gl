// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-sampling';

const UNIT = 'm (relative to localOrigin)';
/** Index of the off-centre point, which lies on a cell corner. */
const OFF_CENTRE_POINT = 7;

/**
 * Records GPURasterSampling on the Dixie Fire window: bilinear DEM elevation at 25 points. The
 * points form a 5 by 5 lattice 12 cells apart whose members sit on cell centres (column and row
 * 7.5, 19.5, 31.5, 43.5, 55.5 counted from the south-west corner), except point 7 (lattice column
 * 2, row 1), which sits on the cell corner at column 36, row 21 (4320 m east, 2520 m north), so
 * bilinear averages four cells (1305.4 m) where nearest would return one cell (1350.4 m).
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-sampling.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const terrain = await loadDixieDem();
  const entry = getRasterOperatorEntries()['GPURasterSampling'];
  const {grid} = terrain;
  const cellSize = grid.affine[0];
  const points: number[] = [];
  for (let row = 0; row < 5; row++) {
    for (let column = 0; column < 5; column++) {
      points.push((7.5 + 12 * column) * cellSize, (7.5 + 12 * row) * cellSize);
    }
  }
  points[2 * OFF_CENTRE_POINT] = 36 * cellSize;
  points[2 * OFF_CENTRE_POINT + 1] = 21 * cellSize;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterSampling on the Dixie Fire window',
    grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [{id: 'points', label: 'sample points', size: 25, indexSpace: '1d'}],
    choices: [
      {
        stage: 'operator',
        name: 'method',
        value: 'bilinear',
        alternatives: ['nearest', 'bilinear', 'bicubic'],
        sweepable: true
      },
      {stage: 'operator', name: 'noDataPolicy', value: 'strict', sweepable: false},
      {stage: 'support', name: 'localOrigin', value: 'south-west corner', sweepable: false}
    ],
    build: async session => {
      // Registered first so the datum rung pins a point.
      const elevations = session.output(
        'samplingValues',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'points',
          meaning: 'DEM elevation sampled at each point (m)'
        },
        25
      );
      const dem = session.input('samplingDem', 'float32', terrain.elevation, {
        role: 'input',
        keySpace: 'cells',
        geometry: {
          kind: 'raster',
          shape: [grid.width, grid.height],
          coordinateModel: terrain.coordinateModel,
          cellSize,
          unit: UNIT,
          origin: [0, grid.height * cellSize],
          rowDirection: 'south'
        },
        meaning: 'elevation (m), rows counted from the northern edge'
      });
      const pointView = session.input('samplingPoints', 'float32', points, {
        role: 'input',
        components: 2,
        geometry: {kind: 'points', coordinateModel: terrain.coordinateModel, unit: UNIT},
        meaning: `sample points, ${UNIT}`
      });
      const positions = session.graph.createDataView(pointView.buffer, {
        format: 'float32x2',
        length: 25
      });
      session.capture('samplingSampledPoints', pointView, {
        role: 'output',
        components: 2,
        geometry: {
          kind: 'points',
          coordinateModel: terrain.coordinateModel,
          unit: UNIT,
          valuesCapture: 'samplingValues'
        },
        highlight: {key: OFF_CENTRE_POINT, label: 'off-centre point: bilinear mean of four cells'},
        meaning: 'sample points coloured by bilinear DEM elevation (m)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        dem,
        points: positions,
        elevations
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterSampling',
  title: 'RasterSampling',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
