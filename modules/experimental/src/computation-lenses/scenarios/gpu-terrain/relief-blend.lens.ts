// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.relief-blend';

/** Hillshade and slope of the Dixie Fire DEM blended into one grey image, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runExample: runDerivativesExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./relief-blend.example'),
    import('./terrain-derivatives.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUReliefBlend'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUReliefBlend of hillshade and slope on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'presentation',
        name: 'blendMode',
        value: 'luminosity',
        alternatives: ['normal', 'multiply', 'screen', 'overlay', 'soft-light', 'luminosity'],
        sweepable: false
      },
      {stage: 'presentation', name: 'opacity', value: 0.5, sweepable: true},
      {stage: 'presentation', name: 'minimum', value: 0, unit: 'deg', sweepable: false},
      {stage: 'presentation', name: 'maximum', value: 50, unit: 'deg', sweepable: false}
    ],
    build: async session => {
      // The elevation only feeds the hillshade and slope layers: drawn as context, not as an input.
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const hillshade = session.output('hillshade', 'float32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'bottom layer: hillshade (0–1)'
      });
      const slope = session.output('slope', 'float32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'top layer: slope (degrees), stretched 0–50 and inverted'
      });
      const aspect = session.output('aspect', 'float32', false);
      const reliefBlend = session.output('reliefBlend', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'blended grey (0–1)'
      });
      // The layers come from GPUTerrainDerivatives in the same graph.
      const derivatives = await runDerivativesExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        slope,
        aspect,
        hillshade
      });
      session.track(derivatives.parameters);
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        hillshade,
        slope,
        reliefBlend
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUReliefBlend',
  title: 'Hillshade and slope blend',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
