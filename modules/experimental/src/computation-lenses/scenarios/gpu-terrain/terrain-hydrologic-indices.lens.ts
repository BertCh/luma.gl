// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-hydrologic-indices';

/** Topographic wetness and stream power of the Dixie Fire DEM, on top of a filled D8 routing. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-hydrologic-indices.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainHydrologicIndices'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainHydrologicIndices on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false},
      {stage: 'operator', name: 'minimumSlope', value: 0.001, sweepable: true}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const filledElevation = session.output('flowFilledElevation', 'float32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'filled elevation (m)'
      });
      const accumulation = session.output('flowAccumulation', 'float32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'upstream area (m²)'
      });
      const slope = session.output('indicesSlope', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'tan β after the minimum-slope clamp (m/m)'
      });
      const specificCatchmentArea = session.output('specificCatchmentArea', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'specific catchment area a (m)'
      });
      const wetnessIndex = session.output('wetnessIndex', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'topographic wetness index ln(a / tan β)',
        highlightExtreme: {pick: 'max', label: value => `wettest cell: ${value.toFixed(1)}`}
      });
      const streamPowerIndex = session.output('streamPowerIndex', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'stream power index a · tan β (m)'
      });
      const {parameters, flowParameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        filledElevation,
        accumulation,
        slope,
        specificCatchmentArea,
        wetnessIndex,
        streamPowerIndex
      });
      session.track(parameters);
      session.track(flowParameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainHydrologicIndices',
  title: 'Topographic wetness and stream power',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
