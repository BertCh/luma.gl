// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-topographic-position';

/** Topographic position index at one 11 × 11 window scale on the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-topographic-position.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainTopographicPosition'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainTopographicPosition on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'neighbourhood', name: 'radius', value: 5, unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'innerRadius', value: 0, unit: 'cell', sweepable: false},
      {stage: 'representation', name: 'quantum', value: 1 / 256, unit: 'm', sweepable: false}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const topographicPositionIndex = session.output('topographicPositionIndex', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'topographic position index (m above the neighbourhood mean, centre excluded)'
      });
      await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        topographicPositionIndex
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainTopographicPosition',
  title: 'Topographic position index (5-cell window)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
