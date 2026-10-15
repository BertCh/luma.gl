// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.solar-position';

/** Sun azimuth, altitude and daylight at eight northern California sites at 13:00 PDT on 5 August 2021. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./solar-position.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  // The DEM only supplies the lens frame: the sites are points, not raster cells.
  const data = await loadDixieDem();
  const entry = getTerrainOperatorEntries()['GPUSolarPosition'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'GPUSolarPosition at eight northern California sites',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'operator',
        name: 'timestamp',
        value: '2021-08-05T20:00:00Z',
        sweepable: true
      },
      {
        stage: 'operator',
        name: 'refraction',
        value: true,
        alternatives: [true, false],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'daylightAltitudeDegrees',
        value: -0.833,
        unit: 'deg',
        sweepable: true
      }
    ],
    keySpaces: [{id: 'sites', label: 'sun sites', size: 8, indexSpace: '1d'}],
    build: async session => {
      // Greenville, Chester, Quincy, Susanville, Chico, Redding, Reno, Sacramento (longitude, latitude).
      const siteCoordinates = [
        [-120.95, 40.14],
        [-121.23, 40.3],
        [-120.95, 39.94],
        [-120.65, 40.42],
        [-121.84, 39.73],
        [-122.39, 40.59],
        [-119.81, 39.53],
        [-121.49, 38.58]
      ].flat();
      const siteRows = session.input('solarSites', 'float32', siteCoordinates, {
        role: 'input',
        components: 2,
        keySpace: 'sites',
        meaning: 'site longitude, latitude (degrees east, north)'
      });
      const positions = session.graph.createDataView(siteRows.buffer, {
        format: 'float32x2',
        length: 8
      });
      const azimuth = session.output(
        'solarAzimuth',
        'float32',
        {
          role: 'intermediate',
          keySpace: 'sites',
          meaning: 'sun azimuth (degrees clockwise from north)'
        },
        8
      );
      const altitude = session.output(
        'solarAltitude',
        'float32',
        {
          role: 'output',
          keySpace: 'sites',
          meaning: 'sun altitude above the horizon (degrees, refracted)'
        },
        8
      );
      const daylight = session.output(
        'solarDaylight',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'sites',
          meaning: 'daylight (1 when the sun is above -0.833°)'
        },
        8
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        positions,
        azimuth,
        altitude,
        daylight
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUSolarPosition',
  title: 'Sun position at eight sites',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
