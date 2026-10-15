// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-stream-order';

/** Strahler stream order of the Dixie Fire stream network, on top of a filled D8 routing. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-stream-order.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainStreamOrder'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainStreamOrder on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [{stage: 'iteration', name: 'maximumIterations', value: 64, sweepable: true}],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const flowDirections = session.output('flowDirections', 'uint32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'D8 flow direction (ESRI code)'
      });
      const streams = session.output('flowStreams', 'uint32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'stream mask (1 where upstream cells ≥ 40)'
      });
      const streamOrder = session.output('streamOrder', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'Strahler stream order (0 off streams)'
      });
      const converged = session.output(
        'streamOrderConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: terrain-stream-order-tour-round'},
        1
      );
      const iterationCount = session.output(
        'streamOrderIterationCount',
        'uint32',
        {role: 'intermediate', meaning: 'executed iterations: terrain-stream-order-tour-round'},
        1
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        flowDirections,
        streams,
        streamOrder,
        converged,
        iterationCount
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainStreamOrder',
  title: 'Strahler stream order',
  audience: 'analyst',
  rungs: ['result', 'datum', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
