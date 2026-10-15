// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_BANDS} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-stretch';

/** Records GPURasterStretch on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieBands, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-stretch.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieBands(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPURasterStretch'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterStretch on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_BANDS.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'presentation', name: 'mode', value: 'linear', sweepable: false},
      {stage: 'presentation', name: 'domain', value: 'auto', sweepable: false},
      {stage: 'presentation', name: 'gamma', value: 1, sweepable: true}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const stretchNirAfter = session.input(
        'stretchNirAfter',
        'float32',
        Float32Array.from(data.bands.nirAfter, value =>
          value === data.noData ? NaN : value * data.scale
        ),
        {
          role: 'input',
          geometry: 'raster',
          meaning: 'NIR after reflectance (DN x 0.0001; DN 0 becomes NaN)'
        }
      );
      const stretchedNir = session.output('stretchedNir', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'min-max stretched NIR (0–1)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        nirAfter: stretchNirAfter,
        stretchedNir
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterStretch',
  title: 'RasterStretch',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_BANDS,
  load: async () => ({run})
});
