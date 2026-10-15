// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_BANDS} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-arithmetic';

/** Records GPURasterArithmetic on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieBands, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-arithmetic.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieBands(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPURasterArithmetic'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterArithmetic on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_BANDS.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'operation', value: 'subtract', sweepable: false},
      {stage: 'representation', name: 'scaleA', value: 0.0001, sweepable: false},
      {stage: 'representation', name: 'scaleB', value: 0.0001, sweepable: false},
      {stage: 'representation', name: 'noDataValue', value: 0, sweepable: false}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const nirBefore = session.input('nirBefore', 'float32', data.bands.nirBefore, {
        role: 'input',
        geometry: 'raster',
        meaning: 'NIR DN (reflectance = DN x 0.0001; 0 is no data)'
      });
      const nirAfter = session.input('nirAfter', 'float32', data.bands.nirAfter, {
        role: 'input',
        geometry: 'raster',
        meaning: 'NIR DN (reflectance = DN x 0.0001; 0 is no data)'
      });
      const nirDifference = session.output('nirDifference', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'NIR after minus before (reflectance), midpoint 0'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        nirBefore,
        nirAfter,
        nirDifference
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterArithmetic',
  title: 'RasterArithmetic',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_BANDS,
  load: async () => ({run})
});
