// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-extrema-pyramid';

/** Level 0 has 16 × 16 cells of 4 pixels; level 1 starts after them and has 8 × 8 cells. */
const LEVEL_0_CELLS = 256;
const LEVEL_1_SIDE = 8;
const LEVEL_1_CELLS = LEVEL_1_SIDE * LEVEL_1_SIDE;
const PYRAMID_LENGTH = LEVEL_0_CELLS + LEVEL_1_CELLS;

/**
 * Records GPURasterExtremaPyramid on the Dixie Fire window and shows level 1: 8 × 8 blocks of
 * 960 m. The 64 × 64 window has no odd edge, so every block reduces four children.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-extrema-pyramid.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const terrain = await loadDixieDem();
  const entry = getRasterOperatorEntries()['GPURasterExtremaPyramid'];
  const {grid} = terrain;
  // Level 1 in the window's own frame: same corner as the DEM, blocks of 8 cells.
  const levelOneGeometry = {
    kind: 'raster' as const,
    shape: [LEVEL_1_SIDE, LEVEL_1_SIDE] as [number, number],
    coordinateModel: terrain.coordinateModel,
    cellSize: 8 * grid.affine[0],
    unit: terrain.unit,
    origin: [grid.affine[2], grid.affine[5]] as [number, number],
    rowDirection: 'south' as const
  };
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterExtremaPyramid on the Dixie Fire window',
    grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {
        id: 'pyramid-level-1',
        label: 'level-1 blocks',
        size: LEVEL_1_CELLS,
        indexSpace: '2d',
        shape: [LEVEL_1_SIDE, LEVEL_1_SIDE]
      }
    ],
    choices: [
      {stage: 'support', name: 'firstBlockSize', value: 4, sweepable: false},
      {stage: 'support', name: 'footprint', value: 'cell', sweepable: false},
      {stage: 'support', name: 'levelCount', value: 2, sweepable: false}
    ],
    build: async session => {
      const maximum = session.output('pyramidMaximumLevels', 'float32', false, PYRAMID_LENGTH);
      const minimum = session.output('pyramidMinimumLevels', 'float32', false, PYRAMID_LENGTH);
      const levelOne = (view: typeof maximum) =>
        session.graph.createDataView(view.buffer, {
          format: 'float32',
          length: LEVEL_1_CELLS,
          byteOffset: LEVEL_0_CELLS * Float32Array.BYTES_PER_ELEMENT
        });
      // Level 1 first so the datum rung pins a block.
      session.capture('pyramidMinimumLevel1', levelOne(minimum), {
        role: 'intermediate',
        geometry: levelOneGeometry,
        keySpace: 'pyramid-level-1',
        meaning: 'lowest elevation of each 8 × 8 block (m)'
      });
      session.capture('pyramidMaximumLevel1', levelOne(maximum), {
        role: 'output',
        geometry: levelOneGeometry,
        keySpace: 'pyramid-level-1',
        meaning: 'highest elevation of each 8 × 8 block (m)'
      });
      const dem = session.input('pyramidElevation', 'float32', terrain.elevation, {
        role: 'input',
        keySpace: 'cells',
        geometry: {...levelOneGeometry, shape: [grid.width, grid.height], cellSize: grid.affine[0]},
        meaning: 'elevation (m)'
      });
      await runExample(device, {graph: session.graph, grid, dem, maximum, minimum});
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterExtremaPyramid',
  title: 'RasterExtremaPyramid',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
