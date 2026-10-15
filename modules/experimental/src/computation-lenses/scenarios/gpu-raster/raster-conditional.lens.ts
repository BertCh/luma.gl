// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-conditional';

/** Records GPURasterConditional on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-conditional.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPURasterConditional'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterConditional on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_LANDCOVER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'comparison', value: '==', sweepable: false},
      {stage: 'operator', name: 'threshold', value: 50, sweepable: false},
      {stage: 'operator', name: 'constantA', value: 0.4, sweepable: true},
      {stage: 'operator', name: 'constantB', value: 1, sweepable: true}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const conditionalLandCover = session.input('conditionalLandCover', 'float32', data.classes, {
        role: 'input',
        geometry: 'raster',
        meaning: 'WorldCover class code (50 is built-up)'
      });
      const conditionalFriction = session.output('conditionalFriction', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'built-up 0.4, otherwise 1 (cost/m)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        landCover: conditionalLandCover,
        friction: conditionalFriction
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterConditional',
  title: 'RasterConditional',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
