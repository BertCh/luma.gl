// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-summits';

/** Summits of the Dixie Fire DEM: strict maxima of a 600 m disc that stand 5 m above its ring. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-summits.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainSummits'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainSummits on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false},
      {stage: 'neighbourhood', name: 'radius', value: 600, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'minimumDrop', value: 5, unit: 'm', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'incompleteNeighborhood',
        value: 'reject',
        alternatives: ['reject', 'ignore'],
        sweepable: false
      }
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const summitMask = session.output('summitMask', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'summit mask (1 on a summit)'
      });
      const drop = session.output('summitsDrop', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'drop below the disc ring (m); NaN off summits'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        summitMask,
        drop
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainSummits',
  title: 'Summits',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
