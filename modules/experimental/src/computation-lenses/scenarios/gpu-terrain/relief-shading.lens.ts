// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.relief-shading';

/** Multidirectional hillshade and relief luminance of the Dixie Fire DEM. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./relief-shading.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUReliefShading'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUReliefShading on the Dixie Fire DEM',
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
        stage: 'operator',
        name: 'lights',
        value: 'mdow',
        alternatives: ['mdow', 'custom'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'lightWeighting',
        value: 'aspect',
        alternatives: ['fixed', 'aspect', 'imhof-swing'],
        sweepable: false
      },
      {stage: 'presentation', name: 'hillshadeStrength', value: 0.7, sweepable: true},
      {stage: 'presentation', name: 'exposure', value: 1.1, sweepable: true}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const hillshade = session.output('reliefHillshade', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'multidirectional hillshade (0–1)'
      });
      const relief = session.output('reliefLuminance', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'relief luminance (0–1)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        hillshade,
        relief
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUReliefShading',
  title: 'Multidirectional relief shading',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
