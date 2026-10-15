// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-profile';

const UNIT = 'm (relative to localOrigin)';
/** Profile line from (360 m, 600 m) to (7320 m, 4440 m) east and north of the south-west corner. */
const LINE_VERTICES = [360, 600, 7320, 4440];
/** The line is 7949 m long: samples at 0, 250, ... 7750 m plus the final vertex make 33 rows. */
const SAMPLE_COUNT = 33;

/**
 * Records GPURasterProfile on the Dixie Fire window: one straight line from the ridge in the
 * south-west (1747 m) to the valley in the north-east, sampled every 250 m with bilinear
 * interpolation. The window is UTM 10N metres, so distances are metres on the ground.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-profile.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const terrain = await loadDixieDem();
  const entry = getRasterOperatorEntries()['GPURasterProfile'];
  const {grid} = terrain;
  const cellSize = grid.affine[0];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterProfile on the Dixie Fire window',
    grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {id: 'samples', label: 'profile samples', size: SAMPLE_COUNT, indexSpace: '1d'},
      {id: 'lines', label: 'profile lines', size: 1, indexSpace: '1d'}
    ],
    choices: [
      {stage: 'operator', name: 'spacing', value: 250, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'method', value: 'bilinear', sweepable: false},
      {stage: 'support', name: 'localOrigin', value: 'south-west corner', sweepable: false}
    ],
    build: async session => {
      // Per-sample results are registered first so the datum rung pins a sample.
      const sampleValues = session.output(
        'profileValues',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'samples',
          meaning: 'DEM elevation at each sample (m)'
        },
        SAMPLE_COUNT
      );
      const sampleDistances = session.output(
        'profileDistances',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'samples',
          meaning: 'distance along the line (m)'
        },
        SAMPLE_COUNT
      );
      const lineGains = session.output(
        'profileGains',
        'float32',
        {role: 'intermediate', keySpace: 'lines', meaning: 'total ascent along the line (m)'},
        1
      );
      const count = session.output(
        'profileCount',
        'uint32',
        {role: 'intermediate', meaning: 'sample rows written'},
        1
      );
      const overflow = session.output(
        'profileOverflow',
        'uint32',
        {role: 'intermediate', meaning: '1 when samples were dropped for lack of capacity'},
        1
      );
      const dem = session.input('profileDem', 'float32', terrain.elevation, {
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
      const lineValues = session.input('profileLine', 'float32', LINE_VERTICES, {
        role: 'input',
        components: 2,
        geometry: {
          kind: 'lines',
          coordinateModel: terrain.coordinateModel,
          unit: UNIT,
          offsetsCapture: 'profileLineOffsets'
        },
        meaning: `profile line vertices, ${UNIT}`
      });
      const vertices = session.graph.createDataView(lineValues.buffer, {
        format: 'float32x2',
        length: 2
      });
      const lineOffsets = session.input('profileLineOffsets', 'uint32', [0, 2], {
        role: 'intermediate',
        meaning: 'vertex offsets of the profile line'
      });
      // Sample positions are drawn as points coloured by their elevation.
      const positionValues = session.output(
        'profilePositions',
        'float32',
        {
          role: 'output',
          components: 2,
          geometry: {
            kind: 'points',
            coordinateModel: terrain.coordinateModel,
            unit: UNIT,
            valuesCapture: 'profileValues'
          },
          meaning: 'profile samples coloured by bilinear DEM elevation (m)'
        },
        2 * SAMPLE_COUNT
      );
      const samplePositions = session.graph.createDataView(positionValues.buffer, {
        format: 'float32x2',
        length: SAMPLE_COUNT
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        dem,
        vertices,
        lineOffsets,
        sampleCount: SAMPLE_COUNT,
        output: {
          status: {count, overflow},
          samplePositions,
          sampleDistances,
          sampleValues,
          lineGains
        }
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterProfile',
  title: 'RasterProfile',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
