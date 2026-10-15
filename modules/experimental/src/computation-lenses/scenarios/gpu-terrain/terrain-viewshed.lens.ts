// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-viewshed';

/** Runs GPUTerrainViewshed on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-viewshed.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainViewshed'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Visibility from ridge cell (45, 49)',
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
      {stage: 'operator', name: 'observerHeight', value: 1.7, unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        value: 6.8278187e-8,
        unit: '1/m',
        sweepable: true
      }
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      // Ridge observer: column 45, row 49, elevation 1651 m; eye 1.7 m above ground.
      const viewshedVisibility = session.output('viewshedVisibility', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning:
          'GPU_TERRAIN_VISIBILITY: 0 hidden, 1 visible, 2 out of range, 3 no data, 4 marginal',
        highlight: {key: 49 * 64 + 45, label: 'observer'}
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        viewshedVisibility
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainViewshed',
  title: 'Visibility from ridge cell (45, 49)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
