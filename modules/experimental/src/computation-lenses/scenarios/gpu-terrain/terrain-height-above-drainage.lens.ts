// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-height-above-drainage';

/** Height above the nearest drainage of the Dixie Fire DEM, on top of a filled D8 routing. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-height-above-drainage.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainHeightAboveDrainage'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainHeightAboveDrainage on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [{stage: 'iteration', name: 'maximumIterations', value: 32, sweepable: true}],
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
      const heightAboveDrainage = session.output('heightAboveDrainage', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'height above drainage (m)',
        highlightExtreme: {
          pick: 'max',
          label: value => `highest above drainage: ${value.toFixed(0)} m`
        }
      });
      const drainageCells = session.output('drainageCells', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'drainage cell index (first stream cell on the D8 path)'
      });
      const converged = session.output(
        'handConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: terrain-height-above-drainage-pointer-round'},
        1
      );
      const iterationCount = session.output(
        'handIterationCount',
        'uint32',
        {
          role: 'intermediate',
          meaning: 'executed iterations: terrain-height-above-drainage-pointer-round'
        },
        1
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        filledElevation,
        flowDirections,
        streams,
        heightAboveDrainage,
        drainageCells,
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
  element: 'GPUTerrainHeightAboveDrainage',
  title: 'Height above nearest drainage (HAND)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
