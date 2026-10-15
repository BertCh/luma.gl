// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_BANDS} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.change-detection';

/** Records GPUChangeDetection on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieBands, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./change-detection.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieBands(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPUChangeDetection'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUChangeDetection on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_BANDS.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'beforeSlice', value: 0, sweepable: false},
      {stage: 'operator', name: 'afterSlice', value: 1, sweepable: false}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      const bands = (['nir', 'swir'] as const).map(name => {
        const before = data.bands[name === 'nir' ? 'nirBefore' : 'swirBefore'];
        const after = data.bands[name === 'nir' ? 'nirAfter' : 'swirAfter'];
        const values = new Float32Array(4096 * 2);
        for (let cell = 0; cell < 4096; cell++) {
          values[cell * 2] = before[cell] === data.noData ? NaN : before[cell] * data.scale;
          values[cell * 2 + 1] = after[cell] === data.noData ? NaN : after[cell] * data.scale;
        }
        const slices = session.input(`${name}ChangeSlices`, 'float32', values, false);
        // The stack interleaves the two dates per cell; captures must be packed, so the same
        // values are also uploaded and captured as one raster per date.
        for (const [slice, period] of ['Before', 'After'].entries()) {
          session.input(
            `${name}Change${period}`,
            'float32',
            Float32Array.from({length: 4096}, (_, cell) => values[cell * 2 + slice]),
            {
              role: 'input',
              geometry: 'raster',
              meaning: `${name} ${period} reflectance (DN x 0.0001; DN 0 becomes NaN)`
            }
          );
        }
        const difference = session.output(`${name}ChangeDifference`, 'float32', {
          role: 'output',
          geometry: 'raster',
          meaning: `${name} after minus before (reflectance), midpoint 0`
        });
        return {name, slices, difference};
      });
      const {parameters} = await runExample(device, {graph: session.graph, grid: data.grid, bands});
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUChangeDetection',
  title: 'ChangeDetection',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_BANDS,
  load: async () => ({run})
});
