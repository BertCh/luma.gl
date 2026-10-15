// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_GFS_WIND} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.particle-advection';

const COLUMNS = 20;
const ROWS = 10;
const PARTICLE_COUNT = COLUMNS * ROWS;
/** Twelve steps of two hours: 24 hours of drift. */
const STEP_COUNT = 12;
/** Metres per degree of latitude on a sphere of the mean Earth radius, 6371008.8 m. */
const METRES_PER_DEGREE = (6371008.8 * Math.PI) / 180;
const SPEED_UNIT = 'deg/h';

/**
 * Records GPUParticleAdvection on the GFS 10 m wind (2024-09-26 12Z, lon/lat degrees): 200
 * particles start on a 20 by 10 lattice (longitude -96 + 0.4 i, latitude 33 + 0.65 j) and take 12
 * steps of 2 hours. The wind is converted to degrees per hour first, because the contributor adds
 * `h * v` to the position directly: u / (111195 m * cos(latitude of the row)) and v / 111195 m,
 * times 3600 s. No particle leaves the window, so none respawns and every trail is a real path.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [{runExample}, {runRasterLens}, {loadGFSWind, T_DATA_VERSION}, {getRasterOperatorEntries}] =
    await Promise.all([
      import('./particle-advection.example'),
      import('./raster-lens-harness'),
      import('./raster-lens-data'),
      import('../../../gpu-raster/computation-pack/index')
    ]);
  const wind = await loadGFSWind();
  const entry = getRasterOperatorEntries()['GPUParticleAdvection'];
  const {grid} = wind;
  const cellSize = grid.affine[0];
  const west = grid.affine[2];
  const north = grid.affine[5];
  // Wind in degrees per hour with row 0 at the southern edge, as the contributor expects.
  const velocities = new Float32Array(2 * grid.width * grid.height);
  for (let row = 0; row < grid.height; row++) {
    const latitude = north - (grid.height - 1 - row + 0.5) * cellSize;
    const metresPerDegreeLongitude = METRES_PER_DEGREE * Math.cos((latitude * Math.PI) / 180);
    for (let column = 0; column < grid.width; column++) {
      const source = (grid.height - 1 - row) * grid.width + column;
      velocities[2 * (row * grid.width + column)] =
        (wind.u[source] * 3600) / metresPerDegreeLongitude;
      velocities[2 * (row * grid.width + column) + 1] = (wind.v[source] * 3600) / METRES_PER_DEGREE;
    }
  }
  const windSpeed = Float32Array.from(wind.u, (u, cell) => Math.hypot(u, wind.v[cell]));
  const seeds: number[] = [];
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < COLUMNS; column++) {
      seeds.push(-96 + 0.4 * column, 33 + 0.65 * row);
    }
  }
  const trailOffsets = Array.from(
    {length: PARTICLE_COUNT + 1},
    (_, particle) => particle * STEP_COUNT
  );
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUParticleAdvection on the GFS wind',
    grid,
    coordinateModel: wind.coordinateModel,
    unit: wind.unit,
    dataset: {id: T_GFS_WIND.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [{id: 'particles', label: 'particles', size: PARTICLE_COUNT, indexSpace: '1d'}],
    choices: [
      {stage: 'operator', name: 'timeStep', value: 2, unit: 'h', sweepable: true},
      {stage: 'operator', name: 'speedScale', value: 1, sweepable: false},
      {stage: 'operator', name: 'dropRate', value: 0, sweepable: false},
      {stage: 'iteration', name: 'stepCount', value: STEP_COUNT, sweepable: true}
    ],
    build: async session => {
      // Per-particle results are registered first so the datum rung pins a particle.
      const speeds = session.output(
        'particleSpeeds',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'particles',
          meaning: `wind speed at the final position (${SPEED_UNIT})`
        },
        PARTICLE_COUNT
      );
      session.input('particleWindSpeed', 'float32', windSpeed, {
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
      session.input('particleSeeds', 'float32', seeds, {
        role: 'input',
        components: 2,
        geometry: {kind: 'points', coordinateModel: wind.coordinateModel, unit: wind.unit},
        meaning: 'start positions, lon/lat degrees on a 20 by 10 lattice'
      });
      const velocityValues = session.input('particleVelocities', 'float32', velocities, false);
      const velocityView = session.graph.createDataView(velocityValues.buffer, {
        format: 'float32x2',
        length: grid.width * grid.height
      });
      const statePositions = session.input('particlePositions', 'float32', seeds, {
        role: 'intermediate',
        components: 2,
        geometry: {kind: 'points', coordinateModel: wind.coordinateModel, unit: wind.unit},
        meaning: 'positions after the last step, lon/lat degrees'
      });
      const positions = session.graph.createDataView(statePositions.buffer, {
        format: 'float32x2',
        length: PARTICLE_COUNT
      });
      const ages = session.input(
        'particleAges',
        'uint32',
        new Array(PARTICLE_COUNT).fill(0),
        false
      );
      const generations = session.input(
        'particleGenerations',
        'uint32',
        new Array(PARTICLE_COUNT).fill(0),
        false
      );
      session.input('particleTrailOffsets', 'uint32', trailOffsets, {
        role: 'intermediate',
        meaning: 'first trail vertex of each particle'
      });
      const trailValues = session.output(
        'particleTrails',
        'float32',
        {
          role: 'output',
          components: 2,
          geometry: {
            kind: 'lines',
            coordinateModel: wind.coordinateModel,
            unit: wind.unit,
            offsetsCapture: 'particleTrailOffsets',
            valuesCapture: 'particleSpeeds'
          },
          meaning: '24 h trails, one vertex per 2 h step, coloured by final speed'
        },
        2 * PARTICLE_COUNT * STEP_COUNT
      );
      const trails = session.graph.createDataView(trailValues.buffer, {
        format: 'float32x2',
        length: PARTICLE_COUNT * STEP_COUNT
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        velocities: velocityView,
        positions,
        ages,
        generations,
        speeds,
        trails
      });
      for (const buffer of parameters) session.track(buffer);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUParticleAdvection',
  title: 'ParticleAdvection',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_GFS_WIND,
  load: async () => ({run})
});
