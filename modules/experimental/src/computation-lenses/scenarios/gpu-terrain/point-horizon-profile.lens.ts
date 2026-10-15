// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.point-horizon-profile';

/** Runs GPUPointHorizonProfile on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./point-horizon-profile.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUPointHorizonProfile'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Skyline angles for three ridge observers',
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
      {stage: 'neighbourhood', name: 'azimuthCount', value: 36, sweepable: false}
    ],
    keySpaces: [
      {id: 'horizon-observers', label: 'ridge observers', size: 3, indexSpace: '1d'},
      {
        id: 'horizon-bins',
        label: 'azimuth bins by observer',
        size: 108,
        indexSpace: '2d',
        shape: [36, 3]
      }
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
      const skylineAngle = session.output(
        'pointHorizonSkylineAngle',
        'float32',
        {
          role: 'output',
          keySpace: 'horizon-bins',
          geometry: {kind: 'raster', shape: [36, 3], coordinateModel: 'unknown'},
          meaning:
            'skyline angle (degrees): 36 azimuth bins × 3 observers strip, 10° clockwise from north per bin; not a terrain raster'
        },
        108
      );
      const tangent = session.output(
        'pointHorizonTangent',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'horizon-bins',
          geometry: {kind: 'raster', shape: [36, 3], coordinateModel: 'unknown'},
          meaning:
            'maximum apparent tangent (m/m): 36 azimuth bins × 3 observers strip; not a terrain raster'
        },
        108
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        observers,
        skylineAngle,
        tangent
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUPointHorizonProfile',
  title: 'Skyline angles for three ridge observers',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
