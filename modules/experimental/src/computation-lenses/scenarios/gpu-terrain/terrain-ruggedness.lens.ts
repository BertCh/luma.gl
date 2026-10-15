// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-ruggedness';

/** Riley terrain ruggedness index of the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-ruggedness.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainRuggedness'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainRuggedness on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'operator',
        name: 'terrainRuggednessAlgorithm',
        value: 'riley',
        alternatives: ['riley', 'wilson'],
        sweepable: false
      },
      {
        stage: 'neighbourhood',
        name: 'edgeMode',
        value: 'nodata',
        alternatives: ['nodata', 'extrapolate'],
        sweepable: false
      }
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const terrainRuggednessIndex = session.output('terrainRuggednessIndex', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'terrain ruggedness index (m); the outer ring is NaN',
        highlightExtreme: {pick: 'max', label: value => `most rugged cell: ${value.toFixed(0)} m`}
      });
      await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        terrainRuggednessIndex
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainRuggedness',
  title: 'Terrain ruggedness index (Riley)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
