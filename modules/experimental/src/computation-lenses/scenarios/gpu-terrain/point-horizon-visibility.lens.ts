// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.point-horizon-visibility';

/** Runs GPUPointHorizonVisibility on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./point-horizon-visibility.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUPointHorizonVisibility'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Target visibility from three ridge observers',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false},
      {stage: 'support', name: 'projection', value: 'planar', sweepable: false},
      {stage: 'neighbourhood', name: 'maximumDistance', value: 10000, unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'traversal',
        value: 'march',
        alternatives: ['march', 'pyramid'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        value: 6.8278187e-8,
        unit: '1/m',
        sweepable: true
      },
      {stage: 'operator', name: 'toleranceDegrees', value: 0.02, unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'sigmaZ', value: 5, unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'skylineToleranceDegrees',
        value: 0.05,
        unit: 'deg',
        sweepable: true
      },
      {
        stage: 'neighbourhood',
        name: 'targetIgnoreDistance',
        value: 'one ground pixel',
        sweepable: false
      }
    ],
    keySpaces: [
      {id: 'horizon-observers', label: 'ridge observers', size: 3, indexSpace: '1d'},
      {id: 'horizon-targets', label: 'four targets per observer', size: 12, indexSpace: '1d'}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      // Ridge cells (45,49), (19,43), (4,10), with 1.7 m eyes; shared by both point-horizon lenses.
      const observerRows = session.input(
        'horizonObservers',
        'float32',
        [45, 49, 1.7, 0, 19, 43, 1.7, 0, 4, 10, 1.7, 0],
        {
          role: 'input',
          components: 4,
          keySpace: 'horizon-observers',
          meaning: 'observer column, row (cell-center indices), eye height (m), padding'
        }
      );
      const observers = session.graph.createDataView(observerRows.buffer, {
        format: 'float32x4',
        length: 3
      });
      // Four fixed targets per observer: northwest, northeast, southeast and southwest of the window.
      const targetCells = [
        [16, 16],
        [52, 16],
        [52, 55],
        [16, 55]
      ];
      const targetValues = Array.from({length: 3}, (_, observerIndex) =>
        targetCells.flatMap(([column, row]) => [column, row, 0, observerIndex])
      ).flat();
      const targetRows = session.input('horizonTargets', 'float32', targetValues, {
        role: 'input',
        components: 4,
        keySpace: 'horizon-targets',
        meaning: 'target column, row (cell-center indices), height (m), observer index'
      });
      const targets = session.graph.createDataView(targetRows.buffer, {
        format: 'float32x4',
        length: 12
      });
      const visibility = session.output(
        'pointHorizonVisibility',
        'uint32',
        {
          role: 'output',
          keySpace: 'horizon-targets',
          meaning:
            'GPU_TERRAIN_VISIBILITY per target: 0 hidden, 1 visible, 2 out of range, 3 no data, 4 marginal'
        },
        12
      );
      const detailValues = session.output(
        'pointHorizonDetails',
        'float32',
        {
          role: 'intermediate',
          components: 4,
          keySpace: 'horizon-targets',
          meaning: 'target, occluder, skyline elevation angles (degrees), on-skyline flag (0 or 1)'
        },
        48
      );
      const details = session.graph.createDataView(detailValues.buffer, {
        format: 'float32x4',
        length: 12
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        observers,
        targets,
        visibility,
        details
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUPointHorizonVisibility',
  title: 'Target visibility from three ridge observers',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
