// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.weighted-overlay';

/** Records GPUWeightedOverlay on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./weighted-overlay.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [data, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPUWeightedOverlay'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUWeightedOverlay on the Dixie Fire window',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_LANDCOVER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'operator', name: 'landCoverWeight', value: 0.7, sweepable: true},
      {stage: 'operator', name: 'elevationWeight', value: 0.3, sweepable: true},
      {stage: 'operator', name: 'normalizeWeights', value: false, sweepable: false},
      {stage: 'operator', name: 'noDataPolicy', value: 'propagate', sweepable: false}
    ],
    build: async session => {
      session.input('landscapeElevation', 'float32', terrain.elevation, {
        role: 'context',
        geometry: 'raster',
        meaning: 'elevation context (m)'
      });
      // Demonstration suitability, not a calibrated risk model: land cover weight 0.7, elevation weight 0.3.
      // Tree/grass/crop/built-up/water scores: 0.25/0.75/1/0.5/0; elevation <1100/<1400/<1700/else: 1/0.75/0.5/0.25.
      const landscapeStack = session.input(
        'landscapeStack',
        'float32',
        [...data.classes, ...terrain.elevation],
        false
      );
      for (const [layer, name] of ['overlayLandCover', 'overlayElevation'].entries()) {
        session.capture(
          name,
          session.graph.createDataView(landscapeStack.buffer, {
            format: 'float32',
            length: 4096,
            byteOffset: layer * 4096 * 4
          }),
          {
            role: 'input',
            geometry: 'raster',
            meaning: layer === 0 ? 'WorldCover code; weight 0.7' : 'elevation (m); weight 0.3'
          }
        );
      }
      const remapBreaks = session.input(
        'overlayBreaks',
        'float32',
        [20, 35, 45, 65, 1100, 1400, 1700, 2000],
        false
      );
      const remapValues = session.input(
        'overlayValues',
        'float32',
        [0.25, 0.75, 1, 0.5, 0, 1, 0.75, 0.5, 0.25, 0.25],
        false
      );
      const suitabilityScore = session.output('suitabilityScore', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'suitability (0–1), land cover 70%, elevation 30%'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        landscapeStack,
        remapBreaks,
        remapValues,
        suitabilityScore
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUWeightedOverlay',
  title: 'WeightedOverlay',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
