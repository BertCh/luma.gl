// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.solar-shadow-mask';

/** Sun visibility on the Dixie Fire DEM at 06:45 PDT on 5 August 2021, from a horizon map. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./solar-shadow-mask.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUSolarShadowMask'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUSolarShadowMask on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'directionCount', value: 16, sweepable: true},
      {stage: 'operator', name: 'azimuthDegrees', value: 73.2, unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'altitudeDegrees', value: 6.3, unit: 'deg', sweepable: true},
      {
        stage: 'operator',
        name: 'angularRadiusDegrees',
        value: 0.2666,
        unit: 'deg',
        sweepable: true
      }
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      // The horizon map (16 sectors per cell) is a working buffer: too large to record.
      const horizon = session.output(
        'terrainHorizon',
        'float32',
        false,
        data.grid.width * data.grid.height * 16
      );
      const sunVisibility = session.output('sunShadowVisibility', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'visible fraction of the solar disk (0–1)'
      });
      const {horizonSettings, sunSettings} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        horizon,
        sunVisibility
      });
      session.track(horizonSettings);
      session.track(sunSettings);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUSolarShadowMask',
  title: 'Sun shadow from a horizon map',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
