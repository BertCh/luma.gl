// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-critical-points';

/** Peaks, pits and saddles of the Dixie Fire DEM from sign changes around each 8-neighbour ring. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-critical-points.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainCriticalPoints'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainCriticalPoints on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'neighbourhood',
        name: 'connectivity',
        value: 8,
        alternatives: [8, 6],
        sweepable: false
      }
    ],
    keySpaces: [
      {id: 'critical-classes', label: 'critical-point classes', size: 6, indexSpace: '1d'}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const classes = session.output('criticalClasses', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'critical-point class (0 regular, 1 peak, 2 pit, 3 saddle, 4 boundary, 5 nodata)'
      });
      const signChanges = session.output('criticalSignChanges', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'sign changes c around the ring'
      });
      const counts = session.output(
        'criticalCounts',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'critical-classes',
          geometry: {kind: 'raster', shape: [6, 1], coordinateModel: 'unknown'},
          meaning:
            'cells per class (regular, peak, pit, saddle, boundary, nodata); not a terrain raster'
        },
        6
      );
      await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        classes,
        signChanges,
        counts
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainCriticalPoints',
  title: 'Critical points (Peucker–Douglas)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
