// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-weiss-landforms';

/** Weiss landform classes of the Dixie Fire DEM, from the example's own graph. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-weiss-landforms.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainWeissLandforms'];
  const cellCount = data.grid.width * data.grid.height;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainWeissLandforms on the Dixie Fire DEM',
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
      {stage: 'neighbourhood', name: 'smallRadius', value: 3, unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'largeRadius', value: 15, unit: 'cell', sweepable: true},
      {
        stage: 'operator',
        name: 'standardization',
        value: 'global',
        alternatives: ['global', 'local'],
        sweepable: false
      },
      {stage: 'operator', name: 'standardThreshold', value: 1, sweepable: true},
      {stage: 'operator', name: 'slopeThresholdDegrees', value: 5, unit: 'deg', sweepable: true}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const weissLandforms = session.output('weissLandforms', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning:
          'landform code: 1 canyon, 2 midslope drainage, 3 upland drainage, 4 U-shaped valley, 5 plain, 6 open slope, 7 upper slope, 8 local ridge, 9 midslope ridge, 10 mountain top; 0 invalid'
      });
      // Two planes (small, then large scale); each plane is captured as its own raster.
      const standardizedPosition = session.output(
        'weissStandardizedPosition',
        'float32',
        false,
        2 * cellCount
      );
      for (const [plane, name, scale] of [
        [0, 'weissSmallPosition', 'small (3 cells)'],
        [1, 'weissLargePosition', 'large (15 cells)']
      ] as const) {
        session.capture(
          name,
          session.graph.createDataView(standardizedPosition.buffer, {
            format: 'float32',
            length: cellCount,
            byteOffset: standardizedPosition.byteOffset + plane * cellCount * 4
          }),
          {
            role: 'intermediate',
            geometry: 'raster',
            meaning: `standardized topographic position, ${scale} scale (z-score)`
          }
        );
      }
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        weissLandforms,
        standardizedPosition
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainWeissLandforms',
  title: 'Ten Weiss landform classes',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
