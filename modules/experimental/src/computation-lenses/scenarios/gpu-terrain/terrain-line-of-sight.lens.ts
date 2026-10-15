// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-line-of-sight';

/** Runs GPUTerrainLineOfSight on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-line-of-sight.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUTerrainLineOfSight'];
  // Shared ridge observer (45, 49); fixed targets around it, in column/row order.
  const targets: [number, number][] = [
    [45, 25],
    [58, 28],
    [60, 40],
    [60, 52],
    [56, 60],
    [45, 61],
    [33, 60],
    [25, 54],
    [20, 43],
    [28, 33],
    [36, 39],
    [50, 52]
  ];
  const cellCenter = (column: number, row: number): [number, number] => [
    data.grid.affine[2] + (column + 0.5) * data.grid.affine[0],
    data.grid.affine[5] + (row + 0.5) * data.grid.affine[4]
  ];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Twelve sight lines from ridge cell (45, 49)',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        value: 'uniform',
        alternatives: ['uniform', 'web-mercator', 'geographic'],
        sweepable: false
      },
      {stage: 'operator', name: 'observerHeight', value: 1.7, unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        value: 6.8278187e-8,
        unit: '1/m',
        sweepable: true
      }
    ],
    keySpaces: [
      {id: 'sight-line-pairs', label: 'observer-target pairs', size: 12, indexSpace: '1d'}
    ],
    build: async session => {
      const dem = session.input('dem', 'float32', data.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const pairRows = session.input(
        'sightLinePairs',
        'float32',
        targets.flatMap(([column, row]) => [45, 49, column, row]),
        {
          role: 'input',
          components: 4,
          keySpace: 'sight-line-pairs',
          meaning: 'observer column/row and target column/row (cell-center indices)'
        }
      );
      const pairs = session.graph.createDataView(pairRows.buffer, {
        format: 'float32x4',
        length: 12
      });
      session.input(
        'sightLineOffsets',
        'uint32',
        Array.from({length: 13}, (_, index) => index * 2),
        {
          role: 'intermediate',
          meaning: 'caller-owned offsets of the 12 input two-vertex lines'
        }
      );
      const coordinates = session.input(
        'sightLineCoordinates',
        'float32',
        targets.flatMap(target => [...cellCenter(45, 49), ...cellCenter(...target)]),
        {
          role: 'input',
          components: 2,
          geometry: {
            kind: 'lines',
            coordinateModel: 'projected',
            unit: 'm',
            offsetsCapture: 'sightLineOffsets'
          },
          meaning: '12 observer-to-target input segments in UTM 10N metres'
        }
      );
      const visibility = session.output(
        'sightLineVisibility',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'sight-line-pairs',
          meaning:
            'GPU_TERRAIN_VISIBILITY code per pair (0 hidden, 1 visible, 2 out of range, 3 no data, 4 marginal)'
        },
        12
      );
      const clearance = session.output(
        'sightLineClearance',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'sight-line-pairs',
          meaning: 'target clearance above the highest intervening tangent (m)'
        },
        12
      );
      session.capture('sightLineVisibilityGeometry', coordinates, {
        role: 'output',
        components: 2,
        geometry: {
          kind: 'lines',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'sightLineOffsets',
          valuesCapture: 'sightLineVisibility'
        },
        meaning: 'input sight lines coloured by GPU visibility code'
      });
      session.capture('sightLineClearanceGeometry', coordinates, {
        role: 'output',
        components: 2,
        geometry: {
          kind: 'lines',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'sightLineOffsets',
          valuesCapture: 'sightLineClearance'
        },
        meaning: 'input sight lines coloured by GPU clearance (m)'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        dem,
        pairs,
        visibility,
        clearance
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainLineOfSight',
  title: 'Twelve sight lines from ridge cell (45, 49)',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
