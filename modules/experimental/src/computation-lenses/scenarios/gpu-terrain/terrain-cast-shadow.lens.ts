// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-cast-shadow';

/** Cast shadow on the Dixie Fire DEM at 06:45 PDT on 5 August 2021, by one hull sweep toward the sun. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-cast-shadow.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainCastShadow'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainCastShadow on the Dixie Fire DEM',
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
      {stage: 'neighbourhood', name: 'maximumRadius', value: 91, unit: 'cell', sweepable: true},
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
      const sunVisibility = session.output('castShadowVisibility', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'visible fraction of the solar disk (0–1)'
      });
      const horizonAngle = session.output('castShadowHorizon', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'horizon angle toward the sun (degrees)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        sunVisibility,
        horizonAngle
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainCastShadow',
  title: 'Cast shadow toward one sun',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
