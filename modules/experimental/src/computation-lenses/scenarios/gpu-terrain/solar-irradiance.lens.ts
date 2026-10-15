// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.solar-irradiance';

/** Sun hours and insolation of 5 August 2021 on the Dixie Fire DEM, from a horizon map. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./solar-irradiance.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUSolarIrradiance'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUSolarIrradiance on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'directionCount', value: 16, sweepable: true},
      {stage: 'operator', name: 'stepMinutes', value: 15, unit: 'min', sweepable: true},
      {
        stage: 'operator',
        name: 'angularRadiusDegrees',
        value: 0.2666,
        unit: 'deg',
        sweepable: true
      },
      {stage: 'operator', name: 'diffuseIrradiance', value: 0, unit: 'W/m²', sweepable: true}
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
      const slope = session.output('irradianceSlope', 'float32', false);
      const aspect = session.output('irradianceAspect', 'float32', false);
      const hillshade = session.output('irradianceHillshade', 'float32', false);
      const sunHours = session.output('irradianceSunHours', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'hours the solar disk is visible'
      });
      const insolation = session.output('irradianceInsolation', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'direct insolation on the sloped surface (Wh/m²)',
        highlightExtreme: {
          pick: 'min',
          label: value => `least sun: ${(value / 1000).toFixed(1)} kWh/m²`
        }
      });
      const {horizonSettings, derivativeSettings, sunTable, settings} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        horizon,
        slope,
        aspect,
        hillshade,
        sunHours,
        insolation
      });
      session.track(horizonSettings);
      session.track(derivativeSettings);
      session.track(sunTable);
      session.track(settings);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUSolarIrradiance',
  title: 'Sun hours and insolation over one day',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
