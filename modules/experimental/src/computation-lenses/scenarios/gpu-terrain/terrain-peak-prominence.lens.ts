// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-peak-prominence';

/** Elder-rule peak prominence of the Dixie Fire DEM, computed entirely on the GPU. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-peak-prominence.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainPeakProminence'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainPeakProminence on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [{stage: 'neighbourhood', name: 'connectivity', value: 8, sweepable: false}],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const prominence = session.output('peakProminence', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'prominence (m); the elder peak is +Infinity',
        highlightExtreme: {
          pick: 'max',
          label: value => `most prominent finite peak: ${value.toFixed(0)} m`
        }
      });
      const keySaddle = session.output('prominenceKeySaddle', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'key saddle elevation (m); -Infinity for the elder peak'
      });
      const parentPeak = session.output('prominenceParentPeak', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'elder peak cell at the merge (0xffffffff for survivors)'
      });
      await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        prominence,
        keySaddle,
        parentPeak
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainPeakProminence',
  title: 'Peak prominence',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
