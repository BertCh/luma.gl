// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-peak-snap';

/** Snaps three candidate points on the Dixie Fire DEM to the summit of a 600 m disc around them. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-peak-snap.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainPeakSnap'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainPeakSnap on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'neighbourhood', name: 'radius', value: 600, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'maximumMove', value: 300, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'maximumHeightChange', value: 80, unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'interior',
        value: true,
        alternatives: [true, false],
        sweepable: false
      }
    ],
    keySpaces: [{id: 'snap-candidates', label: 'candidate points', size: 3, indexSpace: '1d'}],
    build: async session => {
      // Candidates in cell-centre index space (column, row): two a cell or two from the summits
      // at (45, 49) and (40, 21), and one on a flank at (30, 30) that has no summit in its disc.
      const pointGeometry = {kind: 'points', coordinateModel: 'projected', unit: 'cell'} as const;
      const candidateValues = session.input(
        'peakCandidates',
        'float32',
        [46.5, 48, 41.2, 22.4, 30, 30],
        {
          role: 'input',
          components: 2,
          keySpace: 'snap-candidates',
          geometry: pointGeometry,
          meaning: 'candidate position (column, row; rows grow downward)'
        }
      );
      const positionValues = session.output(
        'peakSnapPositions',
        'float32',
        {
          role: 'output',
          components: 2,
          keySpace: 'snap-candidates',
          geometry: pointGeometry,
          meaning: 'snapped position (column, row; rows grow downward)'
        },
        6
      );
      const heights = session.output(
        'peakSnapHeights',
        'float32',
        {role: 'intermediate', keySpace: 'snap-candidates', meaning: 'height at the position (m)'},
        3
      );
      const snapStatus = session.output(
        'peakSnapStatus',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'snap-candidates',
          meaning:
            'snap status (0 unchanged, 1 snapped, 2 on ring, 3 move rejected, 4 height rejected)'
        },
        3
      );
      const snapDistance = session.output(
        'peakSnapDistance',
        'float32',
        {role: 'intermediate', keySpace: 'snap-candidates', meaning: 'move length (m)'},
        3
      );
      // The DEM is not drawn: candidate positions are cell indices, not metres on the raster.
      const dem = session.input('dem', 'float32', data.elevation, false);
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        candidateValues,
        positionValues,
        heights,
        snapStatus,
        snapDistance
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainPeakSnap',
  title: 'Peak snapping',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
