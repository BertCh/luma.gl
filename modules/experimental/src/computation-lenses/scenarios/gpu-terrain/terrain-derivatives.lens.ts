// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-derivatives';

/** Horn slope, aspect and hillshade of the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-derivatives.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainDerivatives'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainDerivatives on the Dixie Fire DEM',
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
      {stage: 'operator', name: 'zFactor', value: 1, sweepable: true},
      {stage: 'presentation', name: 'azimuthDegrees', value: 315, unit: 'deg', sweepable: true},
      {stage: 'presentation', name: 'altitudeDegrees', value: 45, unit: 'deg', sweepable: true}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const slope = session.output('slope', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'slope (degrees)',
        highlightExtreme: {pick: 'max', label: value => `steepest cell: ${value.toFixed(1)}°`}
      });
      const aspect = session.output('aspect', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'aspect (degrees clockwise from north)'
      });
      const hillshade = session.output('hillshade', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'hillshade (0–1)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        slope,
        aspect,
        hillshade
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainDerivatives',
  title: 'Slope, aspect and hillshade (Horn)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
