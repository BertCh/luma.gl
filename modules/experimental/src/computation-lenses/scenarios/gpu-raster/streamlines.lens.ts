// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_GFS_WIND} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.streamlines';

/** Line capacity: the CPU oracle accepts 21 of the 30 seeds, three spare rows are kept. */
const LINE_CAPACITY = 24;
/** Point capacity: the 21 accepted lines hold 1078 points, 1200 rows are kept. */
const POINT_CAPACITY = 1200;
/** Metres per degree of latitude on a sphere of the mean Earth radius, 6371008.8 m. */
const METRES_PER_DEGREE = (6371008.8 * Math.PI) / 180;

/**
 * Records GPUStreamlines on the GFS 10 m wind (2024-09-26 12Z, lon/lat degrees): 30 seeds on a
 * Philox-jittered 6 by 5 lattice over the 16 by 12 degree window, each traced 40 steps of 0.1
 * degrees both ways and pruned to 0.5 degree separation (seed 1, at least 8 points). The wind is
 * converted to degrees per hour first, so the field direction is right in the lon/lat plane.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [{runExample}, {runRasterLens}, {loadGFSWind, T_DATA_VERSION}, {getRasterOperatorEntries}] =
    await Promise.all([
      import('./streamlines.example'),
      import('./raster-lens-harness'),
      import('./raster-lens-data'),
      import('../../../gpu-raster/computation-pack/index')
    ]);
  const wind = await loadGFSWind();
  const entry = getRasterOperatorEntries()['GPUStreamlines'];
  const {grid} = wind;
  const cellSize = grid.affine[0];
  const west = grid.affine[2];
  const north = grid.affine[5];
  const south = north - grid.height * cellSize;
  // Wind in degrees per hour with row 0 at the southern edge, as the contributor expects.
  const velocities = new Float32Array(2 * grid.width * grid.height);
  for (let row = 0; row < grid.height; row++) {
    const latitude = south + (row + 0.5) * cellSize;
    const metresPerDegreeLongitude = METRES_PER_DEGREE * Math.cos((latitude * Math.PI) / 180);
    for (let column = 0; column < grid.width; column++) {
      const source = (grid.height - 1 - row) * grid.width + column;
      const target = 2 * (row * grid.width + column);
      velocities[target] = (wind.u[source] * 3600) / metresPerDegreeLongitude;
      velocities[target + 1] = (wind.v[source] * 3600) / METRES_PER_DEGREE;
    }
  }
  const windSpeed = Float32Array.from(wind.u, (u, cell) => Math.hypot(u, wind.v[cell]));
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUStreamlines on the GFS wind',
    grid,
    coordinateModel: wind.coordinateModel,
    unit: wind.unit,
    dataset: {id: T_GFS_WIND.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [{id: 'lines', label: 'streamlines', size: LINE_CAPACITY, indexSpace: '1d'}],
    choices: [
      {stage: 'support', name: 'separation', value: 0.5, unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'stepLength', value: 0.1, unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'stepsPerDirection', value: 40, sweepable: false},
      {stage: 'operator', name: 'minimumPoints', value: 8, sweepable: false},
      {stage: 'iteration', name: 'maximumIterations', value: 32, sweepable: false}
    ],
    build: async session => {
      // Per-line results are registered first so the datum rung pins a line.
      const ids = session.output(
        'streamlinesLineIds',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'lines',
          meaning: 'seed index (0 to 29) of each published line'
        },
        LINE_CAPACITY
      );
      const lineOffsets = session.output(
        'streamlinesLineOffsets',
        'uint32',
        {role: 'intermediate', meaning: 'first point of each line'},
        LINE_CAPACITY + 1
      );
      const count = session.output(
        'streamlinesLineCount',
        'uint32',
        {role: 'intermediate', meaning: 'lines published'},
        1
      );
      const requiredCount = session.output(
        'streamlinesRequiredCount',
        'uint32',
        {role: 'intermediate', meaning: 'lines accepted before capacity'},
        1
      );
      const overflow = session.output(
        'streamlinesOverflow',
        'uint32',
        {
          role: 'intermediate',
          meaning: 'bit 0: lines dropped for capacity; bit 1: pruning rounds ran out'
        },
        1
      );
      const positionCount = session.output('streamlinesPositionCount', 'uint32', false, 1);
      session.input('streamlinesWindSpeed', 'float32', windSpeed, {
        role: 'input',
        keySpace: 'cells',
        geometry: {
          kind: 'raster',
          shape: [grid.width, grid.height],
          coordinateModel: wind.coordinateModel,
          cellSize,
          unit: wind.unit,
          origin: [west, north],
          rowDirection: 'south'
        },
        meaning: 'GFS 10 m wind speed (m/s)'
      });
      const velocityValues = session.input('streamlinesVelocities', 'float32', velocities, false);
      const velocityView = session.graph.createDataView(velocityValues.buffer, {
        format: 'float32x2',
        length: grid.width * grid.height
      });
      // Unwritten point rows stay NaN, so only the published points are drawn.
      const pointValues = session.input(
        'streamlinesPositions',
        'float32',
        new Float32Array(2 * POINT_CAPACITY).fill(NaN),
        {
          role: 'output',
          components: 2,
          geometry: {
            kind: 'lines',
            coordinateModel: wind.coordinateModel,
            unit: wind.unit,
            offsetsCapture: 'streamlinesLineOffsets',
            valuesCapture: 'streamlinesLineIds'
          },
          meaning: 'evenly spaced streamlines of the wind, lon/lat degrees'
        }
      );
      const positions = session.graph.createDataView(pointValues.buffer, {
        format: 'float32x2',
        length: POINT_CAPACITY
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        velocities: velocityView,
        lines: {ids, status: {count, overflow, requiredCount}},
        lineOffsets,
        positions,
        positionCount
      });
      for (const buffer of parameters) session.track(buffer);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUStreamlines',
  title: 'Streamlines',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_GFS_WIND,
  load: async () => ({run})
});
