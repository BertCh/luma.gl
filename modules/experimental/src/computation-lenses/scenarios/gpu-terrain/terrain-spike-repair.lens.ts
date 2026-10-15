// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-spike-repair';

/** Runs GPUTerrainSpikeRepair on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-spike-repair.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainSpikeRepair'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Repair three injected 900 m terrain spikes',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'step', value: 900, unit: 'm', sweepable: false},
      {stage: 'neighbourhood', name: 'jump', value: 200, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'tolerance', value: 80, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'agreement', value: 0.8, sweepable: true},
      {stage: 'operator', name: 'maximumComponentFraction', value: 0.25, sweepable: true},
      {stage: 'iteration', name: 'maximumComponentIterations', value: 32, sweepable: false}
    ],
    // GPUGraphConnectedComponents annotates its kernels with the `vertex` key space; its vertices
    // are the grid cells (one vertex per pixel, the same row-major index).
    keySpaces: [
      {
        id: 'vertex',
        label: 'component-graph vertices (one per grid cell)',
        size: data.grid.width * data.grid.height,
        indexSpace: '1d'
      }
    ],
    relations: [
      {
        id: 'vertex-to-cells',
        from: 'vertex',
        to: 'cells',
        cardinality: '1:1',
        kind: 'function',
        ref: {function: 'identity'},
        timeDependent: false
      }
    ],
    keyFrames: {loop: 'terrain-spike-repair-components-iterations', maximumFrames: 8},
    build: async session => {
      // Inject +900 m at (column,row) = (20,20), (32,32), (44,44).
      const elevation = data.elevation.slice();
      for (const [column, row] of [
        [20, 20],
        [32, 32],
        [44, 44]
      ])
        elevation[row * 64 + column] += 900;
      const dem = session.input('spikedElevation', 'float32', elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation with 3 injected +900 m spikes (m)'
      });
      // No per-cell repair mask is published, so no repaired-cell highlight is inferred.
      // NaN until the final repair pass: component-label rounds do not yet produce heights.
      const repairedElevation = session.input(
        'repairedElevation',
        'float32',
        new Float32Array(elevation.length).fill(Number.NaN),
        false
      );
      session.capture('repairedElevation', repairedElevation, {
        role: 'output',
        geometry: 'raster',
        meaning: 'repaired elevation (m); unavailable until component labelling completes',
        keyFrames: true
      });
      const repairValidity = session.output('repairValidity', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'repair validity (0 or 1; 0xffffffff unresolved)'
      });
      const repairLabels = session.output('repairLabels', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'published component labels (lowest member cell index)'
      });
      const repairStatistics = session.output(
        'repairStatistics',
        'uint32',
        {
          role: 'intermediate',
          meaning:
            'jump pairs, repaired pixels, shifted components, remaining jump pairs, convergence flag'
        },
        5
      );
      session.capture(
        'repairConverged',
        session.graph.createDataView(repairStatistics.buffer, {
          format: 'uint32',
          byteOffset: 16,
          length: 1
        }),
        {role: 'intermediate', meaning: 'converged: terrain-spike-repair-components-iterations'}
      );
      const {repair} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        repairedElevation,
        repairValidity,
        repairLabels,
        repairStatistics
      });
      session.track(repair);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainSpikeRepair',
  title: 'Repair three injected 900 m terrain spikes',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
