// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.geodesic-destination';
/** Stated range-ring inputs: 8 bearings at 45 degree steps, 3 distances in metres. */
const BEARING_COUNT = 8;
const RANGE_DISTANCES = [500_000, 1_000_000, 2_000_000];

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeodesicDestination',
  title: 'Range rings around Chicago O’Hare',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-openflights-ord',
    source: 'showcase public/data/openflights (OpenFlights.org airports.dat, routes.dat)',
    licence: 'ODbL 1.0, OpenFlights.org'
  },
  load: async () => {
    const [{runS3Lens, geographicGeometry, geographicMetadata}, {runExample}, data] =
      await Promise.all([
        import('./s3-lens-run'),
        import('./geodesic-destination.example'),
        import('./s3-lens-data').then(module => module.loadS3OpenFlightsHub())
      ]);
    // Rows are the stated bearings and distances at the ORD position; no values are measured.
    const rowCount = BEARING_COUNT * RANGE_DISTANCES.length;
    const originPositions = new Float32Array(rowCount * 2);
    const bearingValues = new Float32Array(rowCount);
    const distanceValues = new Float32Array(rowCount);
    for (let ring = 0; ring < RANGE_DISTANCES.length; ring++) {
      for (let step = 0; step < BEARING_COUNT; step++) {
        const row = ring * BEARING_COUNT + step;
        originPositions.set(data.hub.lonLat, row * 2);
        bearingValues[row] = (360 / BEARING_COUNT) * step;
        distanceValues[row] = RANGE_DISTANCES[ring];
      }
    }
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGeodesicDestination',
          datasets: [data.id],
          retain: ['destinations', 'finalBearings', 'converged'],
          keySpaces: [{id: 'row', label: 'bearing and distance', size: rowCount, indexSpace: '1d'}],
          choices: {
            spatialContext: 'longitude-latitude, ellipsoidal WGS84, meters',
            maximumIterations: 16
          },
          metadata: geographicMetadata('geodesic'),
          async build(graph, uploads) {
            const origins = uploads.view('origins', originPositions, 'float32x2');
            const bearings = uploads.view('bearings', bearingValues, 'float32');
            const distances = uploads.view('distances', distanceValues, 'float32');
            const output = await runExample(graph, {origins, bearings, distances});
            return {
              origins: {
                view: origins,
                role: 'input',
                keySpace: 'row',
                geometry: geographicGeometry('points'),
                meaning: 'ORD position repeated for each bearing and distance row'
              },
              bearings: {
                view: bearings,
                role: 'intermediate',
                keySpace: 'row',
                meaning: 'stated initial bearing in degrees clockwise from north'
              },
              distances: {
                view: distances,
                role: 'intermediate',
                keySpace: 'row',
                meaning: 'stated distance in m: 500 km, 1,000 km and 2,000 km rings'
              },
              finalBearings: {
                view: output.finalBearings,
                role: 'intermediate',
                keySpace: 'row',
                meaning: 'bearing on arrival in degrees clockwise from north, in (-180, 180]'
              },
              converged: {
                view: output.converged,
                role: 'intermediate',
                keySpace: 'row',
                meaning: '1 when the Vincenty direct iteration converged within 16 iterations'
              },
              destinations: {
                view: output.destinations,
                role: 'output',
                keySpace: 'row',
                geometry: geographicGeometry('points'),
                meaning: 'WGS84 geodesic destination (capped Vincenty direct) of each row'
              }
            };
          }
        })
    };
  }
});
