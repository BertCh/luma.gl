// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.distance-field';

/** Records GPUDistanceField on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./distance-field.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPUDistanceField'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUDistanceField on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_LANDCOVER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false},
      {
        stage: 'operator',
        name: 'mode',
        value: 'exact',
        alternatives: ['exact', 'jump-flood'],
        sweepable: false
      }
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const waterSeeds = session.input(
        'waterSeeds',
        'uint32',
        Uint32Array.from(data.classes, value => Number(value === 80)),
        {role: 'input', geometry: 'raster', meaning: 'water seed mask (WorldCover class 80)'}
      );
      const waterDistance = session.output('waterDistance', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'exact Euclidean distance to water (m)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        waterSeeds,
        waterDistance
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUDistanceField',
  title: 'DistanceField',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
