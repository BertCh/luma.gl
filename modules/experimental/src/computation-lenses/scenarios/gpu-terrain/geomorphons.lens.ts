// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.geomorphons';

/** Geomorphon landform classes of the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./geomorphons.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUGeomorphons'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUGeomorphons on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        value: 'uniform',
        alternatives: ['uniform', 'web-mercator', 'geographic'],
        sweepable: false
      },
      {stage: 'neighbourhood', name: 'searchRadius', value: 8, unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'skipRadius', value: 0, unit: 'cell', sweepable: false},
      {stage: 'operator', name: 'flatThresholdDegrees', value: 1, unit: 'deg', sweepable: true},
      {
        stage: 'operator',
        name: 'comparison',
        value: 'anglev1',
        alternatives: ['anglev1', 'anglev2', 'anglev2-distance'],
        sweepable: false
      }
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const geomorphonForms = session.output('geomorphonForms', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning:
          'landform code: 1 flat, 2 peak, 3 ridge, 4 shoulder, 5 spur, 6 slope, 7 hollow, 8 footslope, 9 valley, 10 pit; 0 invalid'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        geomorphonForms
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUGeomorphons',
  title: 'Ten landform classes (geomorphons)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
