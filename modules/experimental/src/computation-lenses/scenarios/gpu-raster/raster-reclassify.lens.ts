// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-reclassify';

/** Records GPURasterReclassify on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-reclassify.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPURasterReclassify'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterReclassify on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_LANDCOVER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'breakCount', value: 4, sweepable: false},
      {stage: 'operator', name: 'closed', value: 'left', sweepable: false}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const suitabilityLandCover = session.input('suitabilityLandCover', 'float32', data.classes, {
        role: 'input',
        geometry: 'raster',
        meaning: 'WorldCover class code'
      });
      // Intervals isolate the five observed classes: tree 2, grass 4, crop 5, built-up 3, water 1.
      const breaks = session.input('suitabilityBreaks', 'float32', [20, 35, 45, 65], false);
      const suitabilityValues = session.input(
        'suitabilityValues',
        'float32',
        [2, 4, 5, 3, 1],
        false
      );
      const suitability = session.output('suitability', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'land-cover suitability rank (1 low to 5 high)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        landCover: suitabilityLandCover,
        classBreaks: breaks,
        suitabilityValues,
        suitability
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterReclassify',
  title: 'RasterReclassify',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
