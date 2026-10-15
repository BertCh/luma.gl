// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-curvature';

/** Evans-Young profile and plan curvature of the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-curvature.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainCurvature'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainCurvature on the Dixie Fire DEM',
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
      {
        stage: 'neighbourhood',
        name: 'method',
        value: 'evans-young',
        alternatives: ['evans-young', 'zevenbergen-thorne', 'florinsky'],
        sweepable: false
      },
      {stage: 'operator', name: 'zFactor', value: 1, sweepable: true},
      {stage: 'operator', name: 'flatGradient', value: 1e-6, sweepable: false}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const curvatureProfile = session.output('curvatureProfile', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'profile curvature (1/m, convex positive)'
      });
      const curvaturePlan = session.output('curvaturePlan', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'plan curvature (1/m, convex positive)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        curvatureProfile,
        curvaturePlan
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainCurvature',
  title: 'Profile and plan curvature (Evans-Young)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
