// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_GFS_WIND} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.line-integral-convolution';

const OUTPUT_WIDTH = 128;
const OUTPUT_HEIGHT = 96;
/** Metres per degree of latitude on a sphere of the mean Earth radius, 6371008.8 m. */
const METRES_PER_DEGREE = (6371008.8 * Math.PI) / 180;

/**
 * Records GPULineIntegralConvolution on the GFS 10 m wind (2024-09-26 12Z): a 128 by 96 pixel
 * texture of 0.125 degrees over the 64 by 48 cell wind grid. The white noise is made inside the
 * contributor from seed 7, so the seed is the noise input. Streamlines follow the wind converted
 * to degrees per hour (u / (111195 m * cos(latitude)), v / 111195 m), so directions are right in
 * the lon/lat plane in which the output pixels are laid out.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadGFSWind, T_DATA_VERSION},
    {getRasterOperatorEntries},
    {makeGPURasterMetadata}
  ] = await Promise.all([
    import('./line-integral-convolution.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index'),
    import('../../../gpu-raster/raster-grid/index')
  ]);
  const wind = await loadGFSWind();
  const entry = getRasterOperatorEntries()['GPULineIntegralConvolution'];
  const field = wind.grid;
  const fieldCell = field.affine[0];
  const west = field.affine[2];
  const north = field.affine[5];
  const south = north - field.height * fieldCell;
  const outputCell = (field.width * fieldCell) / OUTPUT_WIDTH;
  const grid = makeGPURasterMetadata({
    width: OUTPUT_WIDTH,
    height: OUTPUT_HEIGHT,
    cellSize: outputCell,
    origin: [west, south],
    rowDirection: 'north'
  });
  // Wind in degrees per hour with row 0 at the southern edge, as the contributor expects.
  const velocities = new Float32Array(2 * field.width * field.height);
  for (let row = 0; row < field.height; row++) {
    const latitude = south + (row + 0.5) * fieldCell;
    const metresPerDegreeLongitude = METRES_PER_DEGREE * Math.cos((latitude * Math.PI) / 180);
    for (let column = 0; column < field.width; column++) {
      const source = (field.height - 1 - row) * field.width + column;
      const target = 2 * (row * field.width + column);
      velocities[target] = (wind.u[source] * 3600) / metresPerDegreeLongitude;
      velocities[target + 1] = (wind.v[source] * 3600) / METRES_PER_DEGREE;
    }
  }
  const windSpeed = Float32Array.from(wind.u, (u, cell) => Math.hypot(u, wind.v[cell]));
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPULineIntegralConvolution on the GFS wind',
    grid,
    coordinateModel: wind.coordinateModel,
    unit: wind.unit,
    dataset: {id: T_GFS_WIND.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {
        id: 'fieldCells',
        label: 'wind cells',
        size: field.width * field.height,
        indexSpace: '2d',
        shape: [field.width, field.height]
      }
    ],
    choices: [
      {stage: 'neighbourhood', name: 'stepCount', value: 24, sweepable: true},
      {stage: 'neighbourhood', name: 'stepLength', value: 0.5, unit: 'pixel', sweepable: true},
      {stage: 'operator', name: 'seed', value: 7, sweepable: false},
      {stage: 'operator', name: 'period', value: 0, sweepable: false}
    ],
    build: async session => {
      const outputRaster = {
        kind: 'raster' as const,
        shape: [OUTPUT_WIDTH, OUTPUT_HEIGHT] as [number, number],
        coordinateModel: wind.coordinateModel,
        cellSize: outputCell,
        unit: wind.unit,
        origin: [west, south] as [number, number],
        rowDirection: 'north' as const
      };
      session.input('licWindSpeed', 'float32', windSpeed, {
        role: 'input',
        keySpace: 'fieldCells',
        geometry: {
          kind: 'raster',
          shape: [field.width, field.height],
          coordinateModel: wind.coordinateModel,
          cellSize: fieldCell,
          unit: wind.unit,
          origin: [west, north],
          rowDirection: 'south'
        },
        meaning: 'GFS 10 m wind speed (m/s)'
      });
      const velocityValues = session.input('licVelocities', 'float32', velocities, false);
      const velocityView = session.graph.createDataView(velocityValues.buffer, {
        format: 'float32x2',
        length: field.width * field.height
      });
      const speeds = session.output('licSpeeds', 'float32', {
        role: 'intermediate',
        keySpace: 'cells',
        geometry: outputRaster,
        meaning: 'wind speed at each pixel centre (deg/h), NaN without data'
      });
      const values = session.output('licValues', 'float32', {
        role: 'output',
        keySpace: 'cells',
        geometry: outputRaster,
        meaning: 'LIC texture, 0 to 1: white noise smeared along the wind (seed 7)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        field,
        grid,
        velocities: velocityView,
        values,
        speeds
      });
      for (const buffer of parameters) session.track(buffer);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPULineIntegralConvolution',
  title: 'LineIntegralConvolution',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_GFS_WIND,
  load: async () => ({run})
});
