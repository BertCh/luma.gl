// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-contours';

/** Runs GPUTerrainContours on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-contours.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainContours'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Seven elevation contours at 100 m intervals',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [{stage: 'operator', name: 'levelInterval', value: 100, unit: 'm', sweepable: false}],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const contourOverflow = session.output(
        'contourOverflow',
        'uint32',
        {
          role: 'intermediate',
          meaning: 'any contour level overflowed (0 or 1)'
        },
        1
      );
      // The DEM spans 1073.9–1785.7 m: seven levels, 1100–1700 m every 100 m.
      // At most two segments per marching square. Unwritten slots stay NaN, never fake vertices.
      const capacity = 2 * (data.grid.width - 1) * (data.grid.height - 1);
      const levels = Array.from({length: 7}, (_, index) => {
        const level = 1100 + index * 100;
        const coordinates = session.input(
          `contourVertices${level}`,
          'float32',
          new Float32Array(capacity * 4).fill(Number.NaN),
          false
        );
        // GPU contours have no path offsets; ResultView requires offsets for lines.
        // Capture the GPU-written vertices as points in local pixel-center coordinates instead.
        session.capture(`contourVertices${level}`, coordinates, {
          role: 'output',
          components: 2,
          geometry: {kind: 'points', coordinateModel: 'projected', unit: 'cell'},
          meaning: `${level} m contour endpoints in local pixel-center coordinates; points because no GPU line offsets exist`
        });
        const vertices = session.graph.createDataView(coordinates.buffer, {
          format: 'float32x2',
          length: capacity * 2
        });
        const count = session.output(
          `contourCount${level}`,
          'uint32',
          {
            role: 'intermediate',
            meaning: `${level} m contour segment count (two vertices per segment)`
          },
          1
        );
        return {level, vertices, status: {count}, capacity};
      });
      await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        levels,
        overflow: contourOverflow
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainContours',
  title: 'Seven elevation contours at 100 m intervals',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
