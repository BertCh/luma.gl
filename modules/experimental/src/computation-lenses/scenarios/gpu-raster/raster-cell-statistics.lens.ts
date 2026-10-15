// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_BANDS} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-cell-statistics';

/** Records GPURasterCellStatistics on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieBands, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-cell-statistics.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieBands(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPURasterCellStatistics'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterCellStatistics on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_BANDS.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'noDataPolicy', value: 'propagate', sweepable: false},
      {stage: 'operator', name: 'minimumValidCount', value: 2, sweepable: true}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const reflectance = ['nirBefore', 'nirAfter'].flatMap(name =>
        Array.from(data.bands[name as 'nirBefore' | 'nirAfter'], value =>
          value === data.noData ? NaN : value * data.scale
        )
      );
      const nirStack = session.input('nirStack', 'float32', reflectance, false);
      for (const [layer, name] of ['nirMeanBefore', 'nirMeanAfter'].entries()) {
        session.capture(
          name,
          session.graph.createDataView(nirStack.buffer, {
            format: 'float32',
            length: 4096,
            byteOffset: layer * 4096 * 4
          }),
          {
            role: 'input',
            geometry: 'raster',
            meaning: 'NIR reflectance (DN x 0.0001; DN 0 becomes NaN)'
          }
        );
      }
      const nirMean = session.output('nirMean', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'mean before/after NIR reflectance; both layers required'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        nirStack,
        nirMean
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterCellStatistics',
  title: 'RasterCellStatistics',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_BANDS,
  load: async () => ({run})
});
