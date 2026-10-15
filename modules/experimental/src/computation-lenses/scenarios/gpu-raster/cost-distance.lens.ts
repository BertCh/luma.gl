// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.cost-distance';

/** Records GPUCostDistance on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./cost-distance.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPUCostDistance'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUCostDistance on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_LANDCOVER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false},
      {stage: 'iteration', name: 'maximumIterations', value: 128, sweepable: true}
    ],
    keyFrames: {loop: 'cost-distance-relax', maximumFrames: 8},
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      // WorldCover: tree 1, grassland 0.7, cropland 0.6, built-up 0.4 cost/m; water NaN is impassable.
      const frictionTable: Record<number, number> = {10: 1, 30: 0.7, 40: 0.6, 50: 0.4, 80: NaN};
      const costFriction = session.input(
        'costFriction',
        'float32',
        Float32Array.from(data.classes, value => frictionTable[value] ?? NaN),
        {
          role: 'input',
          geometry: 'raster',
          meaning:
            'friction (cost/m): tree 1, grassland 0.7, cropland 0.6, built-up 0.4, water impassable'
        }
      );
      // Verified WorldCover built-up cell nearest (37, 29): Greenville itself, column 37, row 29.
      const sourceValues = new Uint32Array(4096);
      sourceValues[29 * 64 + 37] = 1;
      const costSourceMask = session.input('costSourceMask', 'uint32', sourceValues, {
        role: 'input',
        geometry: 'raster',
        meaning: 'Greenville source (1 at column 37, row 29)',
        highlight: {key: 29 * 64 + 37, label: 'Greenville source'}
      });
      const costs = session.output('costs', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'accumulated cost (friction × m)',
        keyFrames: true
      });
      const costConverged = session.output(
        'costConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: cost-distance-relax'},
        1
      );
      const costIterationCount = session.output(
        'costIterationCount',
        'uint32',
        {role: 'intermediate', meaning: 'executed iterations: cost-distance-relax'},
        1
      );
      const costSettledAt = session.output('costSettledAt', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'settledAt round',
        stamp: true
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        friction: costFriction,
        sourceMask: costSourceMask,
        costs,
        converged: costConverged,
        iterationCount: costIterationCount,
        settledAt: costSettledAt
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUCostDistance',
  title: 'CostDistance',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
