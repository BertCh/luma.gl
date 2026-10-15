// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-vector-ruggedness';

/** Vector ruggedness measure (5 × 5 window) of the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-vector-ruggedness.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainVectorRuggedness'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainVectorRuggedness on the Dixie Fire DEM',
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
      {stage: 'neighbourhood', name: 'radius', value: 2, unit: 'cell', sweepable: true},
      {stage: 'operator', name: 'zFactor', value: 1, sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'borderMode',
        value: 'clamp',
        alternatives: ['clamp', 'nodata'],
        sweepable: false
      }
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const vectorRuggednessMeasure = session.output('vectorRuggednessMeasure', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'vector ruggedness measure (0 smooth to 1 rugged)',
        highlightExtreme: {pick: 'max', label: value => `most rugged cell: ${value.toFixed(3)}`}
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        vectorRuggednessMeasure
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainVectorRuggedness',
  title: 'Vector ruggedness measure (Sappington)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
