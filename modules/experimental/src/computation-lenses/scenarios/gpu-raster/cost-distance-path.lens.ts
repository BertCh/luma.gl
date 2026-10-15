// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.cost-distance-path';

/** Records GPUCostDistancePath on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./cost-distance-path.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPUCostDistancePath'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUCostDistancePath on the Dixie Fire window',
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
        role: 'intermediate',
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
      const costBackLinks = session.output('costBackLinks', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'D8 predecessor code'
      });
      // Tree-cover targets checked in WorldCover: (15,15), (50,45), (20,50).
      // Full-grid path capacity prevents truncation. Unwritten IDs stay UINT_MAX, outside the histogram domain.
      const targets = [
        [15, 15],
        [50, 45],
        [20, 50]
      ].map(([column, row], targetIndex) => {
        const target = session.input(`pathTarget${targetIndex}`, 'uint32', [row * 64 + column], {
          role: 'input',
          meaning: `target cell index (${column}, ${row})`
        });
        const ids = session.input(
          `pathCells${targetIndex}`,
          'uint32',
          new Uint32Array(4096).fill(0xffffffff),
          {
            role: 'intermediate',
            meaning: 'ordered cell IDs, target to source; use path count'
          }
        );
        const count = session.output(
          `pathCount${targetIndex}`,
          'uint32',
          {role: 'intermediate', meaning: 'path cell count'},
          1
        );
        const overflow = session.output(
          `pathOverflow${targetIndex}`,
          'uint32',
          {role: 'intermediate', meaning: 'path capacity exceeded'},
          1
        );
        const mask = session.output(`pathMask${targetIndex}`, 'uint32', {
          role: 'output',
          geometry: 'raster',
          meaning: 'path membership (1 on traced cells, 0 elsewhere)'
        });
        return {target, path: {ids, status: {count, overflow}}, mask};
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        friction: costFriction,
        sourceMask: costSourceMask,
        costs,
        converged: costConverged,
        iterationCount: costIterationCount,
        backLinks: costBackLinks,
        targets
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUCostDistancePath',
  title: 'CostDistancePath',
  audience: 'analyst',
  rungs: ['result', 'datum', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
