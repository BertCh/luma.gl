// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-cumulative-viewshed';

/** Runs GPUTerrainCumulativeViewshed on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-cumulative-viewshed.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainCumulativeViewshed'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Visibility counts from six ridge observers',
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
    keySpaces: [{id: 'ridge-observers', label: 'ridge observers', size: 6, indexSpace: '1d'}],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      // Distinct 5×5 local DEM maxima (column, row): six ridge cells, selected from the data.
      const observerCells = [45, 49, 50, 52, 19, 43, 4, 10, 56, 47, 10, 17];
      const observerRows = session.input('cumulativeObservers', 'float32', observerCells, {
        role: 'input',
        components: 2,
        keySpace: 'ridge-observers',
        geometry: {kind: 'points', coordinateModel: 'projected', unit: 'cell'},
        meaning: 'six ridge observer cell-center indices; eye height 1.7 m'
      });
      const observers = session.graph.createDataView(observerRows.buffer, {
        format: 'float32x2',
        length: 6
      });
      const visibleCount = session.output('visibleCount', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'number of observers with visible code (0–6 observers)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        observers,
        visibleCount
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainCumulativeViewshed',
  title: 'Visibility counts from six ridge observers',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
