// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.profile-peaks';

/** Peaks of one west-east elevation profile of the Dixie Fire DEM (row 40, 64 samples). */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./profile-peaks.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUProfilePeaks'];
  // The profile is row 40 of the DEM from west to east: 64 samples 120 m apart.
  const profileValues = data.elevation.slice(40 * 64, 41 * 64);
  const stripGeometry = {kind: 'raster', shape: [64, 1], coordinateModel: 'unknown'} as const;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUProfilePeaks on one Dixie Fire DEM profile',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'minimumProminence', value: 20, unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'window', value: 8, unit: 'samples', sweepable: true},
      {stage: 'neighbourhood', name: 'minimumSide', value: 4, unit: 'samples', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'suppressionRadius',
        value: 2,
        unit: 'samples',
        sweepable: true
      },
      {stage: 'iteration', name: 'maximumSuppressionIterations', value: 16, sweepable: true}
    ],
    keySpaces: [{id: 'profile-samples', label: 'profile samples', size: 64, indexSpace: '1d'}],
    build: async session => {
      const profile = session.input('profileValues', 'float32', profileValues, {
        role: 'input',
        keySpace: 'profile-samples',
        geometry: stripGeometry,
        meaning: 'elevation along row 40 (m); 64 samples west to east, not a terrain raster'
      });
      const offsets = session.input('profileOffsets', 'uint32', [0, 64], false);
      const prominence = session.output(
        'profilePeakProminence',
        'float32',
        {
          role: 'output',
          keySpace: 'profile-samples',
          geometry: stripGeometry,
          meaning: 'prominence of kept peaks (m); NaN elsewhere',
          highlightExtreme: {
            pick: 'max',
            label: value => `most prominent peak: ${value.toFixed(0)} m`
          }
        },
        64
      );
      const refinedIndex = session.output(
        'profileRefinedIndex',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'profile-samples',
          geometry: stripGeometry,
          meaning: 'refined peak position (sample index)'
        },
        64
      );
      const refinedValue = session.output(
        'profileRefinedValue',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'profile-samples',
          geometry: stripGeometry,
          meaning: 'parabola-refined peak elevation (m)'
        },
        64
      );
      const peakMask = session.output(
        'profilePeakMask',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'profile-samples',
          geometry: stripGeometry,
          meaning: 'kept peak mask (1 on a kept peak)'
        },
        64
      );
      const converged = session.output(
        'profilePeaksConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: profile-peaks-nms'},
        1
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        profile,
        offsets,
        prominence,
        refinedIndex,
        refinedValue,
        peakMask,
        converged
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUProfilePeaks',
  title: 'Profile peaks',
  audience: 'analyst',
  rungs: ['result', 'formula', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
