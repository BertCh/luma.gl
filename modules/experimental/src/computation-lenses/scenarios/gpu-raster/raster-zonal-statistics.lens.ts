// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_LANDCOVER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-zonal-statistics';

/** WorldCover codes of the five observed classes; the zone ID is the index in this list. */
const ZONE_CODES = [10, 30, 40, 50, 80];

/** Records GPURasterZonalStatistics on the Dixie Fire window. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieLandCover, loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-zonal-statistics.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [landCover, terrain] = await Promise.all([loadDixieLandCover(), loadDixieDem()]);
  const entry = getRasterOperatorEntries()['GPURasterZonalStatistics'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterZonalStatistics on the Dixie Fire window',
    grid: terrain.grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_LANDCOVER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {id: 'zones', label: 'land-cover zones', size: ZONE_CODES.length, indexSpace: '1d'}
    ],
    choices: [
      {stage: 'support', name: 'zoneCapacity', value: ZONE_CODES.length, sweepable: false},
      {stage: 'operator', name: 'sumOrder', value: 'sorted', sweepable: false}
    ],
    build: async session => {
      // Zone columns are registered first so the datum rung pins the zones key space.
      const zoneNames = 'zones: tree cover, grassland, cropland, built-up, water';
      const zoneCellCounts = session.output(
        'zoneCellCounts',
        'uint32',
        {
          role: 'output',
          keySpace: 'zones',
          meaning: `column cells per zone (${zoneNames})`
        },
        ZONE_CODES.length
      );
      const zoneMeans = session.output(
        'zoneMeans',
        'float32',
        {
          role: 'output',
          keySpace: 'zones',
          meaning: `column mean elevation per zone (m; ${zoneNames})`
        },
        ZONE_CODES.length
      );
      const zoneMinimums = session.output(
        'zoneMinimums',
        'float32',
        {role: 'output', keySpace: 'zones', meaning: 'column lowest elevation per zone (m)'},
        ZONE_CODES.length
      );
      const zoneMaximums = session.output(
        'zoneMaximums',
        'float32',
        {role: 'output', keySpace: 'zones', meaning: 'column highest elevation per zone (m)'},
        ZONE_CODES.length
      );
      const zonalElevation = session.input('zonalElevation', 'float32', terrain.elevation, {
        role: 'input',
        geometry: 'raster',
        meaning: 'elevation (m)'
      });
      const zonalZones = session.input(
        'zonalZones',
        'uint32',
        Uint32Array.from(landCover.classes, code => ZONE_CODES.indexOf(code)),
        {
          role: 'input',
          geometry: 'raster',
          meaning: 'zone ID: 0 tree cover, 1 grassland, 2 cropland, 3 built-up, 4 water'
        }
      );
      await runExample(device, {
        graph: session.graph,
        grid: terrain.grid,
        dem: zonalElevation,
        zones: zonalZones,
        cellCounts: zoneCellCounts,
        means: zoneMeans,
        minimums: zoneMinimums,
        maximums: zoneMaximums
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterZonalStatistics',
  title: 'RasterZonalStatistics',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_LANDCOVER,
  load: async () => ({run})
});
