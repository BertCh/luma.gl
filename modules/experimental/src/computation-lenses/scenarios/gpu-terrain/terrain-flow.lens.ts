// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-flow';

/** D8 hydrology of the Dixie Fire DEM: epsilon fill, routing, accumulation and streams. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-flow.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainFlow'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUTerrainFlow on the Dixie Fire DEM',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false},
      {stage: 'operator', name: 'fillEpsilon', value: 0.01, unit: 'm', sweepable: true},
      {stage: 'operator', name: 'streamThreshold', value: 40, unit: 'cells', sweepable: true},
      {stage: 'operator', name: 'accumulationUnits', value: 'cells', sweepable: false},
      {stage: 'iteration', name: 'maximumFillIterations', value: 64, sweepable: true},
      {stage: 'iteration', name: 'maximumAccumulationIterations', value: 16, sweepable: true}
    ],
    // The filled surface is the part of this element that iterates visibly: the epsilon fill is
    // a gated relaxation, while pull accumulation finishes in a few rounds.
    keyFrames: {loop: 'terrain-flow-fill-relax', maximumFrames: 8},
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      // Uniform rainfall of 1 per cell; a runoff view selects the pull accumulation.
      const rainfall = session.input('flowRainfall', 'float32', new Array(4096).fill(1), false);
      const filledElevation = session.output('flowFilledElevation', 'float32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'filled elevation (m)'
      });
      // The same buffer under a second name keeps one snapshot per relaxation key frame.
      session.capture('flowFillRelaxation', filledElevation, {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'filled elevation during relaxation (m)',
        keyFrames: true
      });
      const flowDirections = session.output('flowDirections', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'D8 flow direction (ESRI code)'
      });
      const cellClasses = session.output('flowCellClasses', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'cell class (0 draining, 1 flat, 2 pit, 3 outlet, 4 invalid)'
      });
      const accumulation = session.output('flowAccumulation', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'upstream cells',
        highlightExtreme: {pick: 'max', label: value => `outlet: ${value} cells`}
      });
      const streams = session.output('flowStreams', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'stream mask (1 where upstream cells ≥ 40)'
      });
      const fillConverged = session.output(
        'flowFillConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: terrain-flow-fill-relax'},
        1
      );
      const fillIterationCount = session.output(
        'flowFillIterationCount',
        'uint32',
        {role: 'intermediate', meaning: 'executed iterations: terrain-flow-fill-relax'},
        1
      );
      const accumulationConverged = session.output(
        'flowAccumulationConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: terrain-flow-accumulate-round'},
        1
      );
      const accumulationIterationCount = session.output(
        'flowAccumulationIterationCount',
        'uint32',
        {role: 'intermediate', meaning: 'executed iterations: terrain-flow-accumulate-round'},
        1
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        rainfall,
        filledElevation,
        flowDirections,
        cellClasses,
        accumulation,
        streams,
        fillConverged,
        fillIterationCount,
        accumulationConverged,
        accumulationIterationCount
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainFlow',
  title: 'D8 flow, accumulation and streams',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'iterations', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
