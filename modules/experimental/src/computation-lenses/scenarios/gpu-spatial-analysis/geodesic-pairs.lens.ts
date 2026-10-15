// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.geodesic-pairs';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeodesicPairs',
  title: 'WGS84 geodesics from Chicago O’Hare',
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
        import('./geodesic-pairs.example'),
        import('./s3-lens-data').then(module => module.loadS3OpenFlightsHub())
      ]);
    const pairCount = data.destinationCount;
    // Every route starts at the hub: the origin column repeats the ORD position.
    const originPositions = new Float32Array(pairCount * 2);
    for (let pair = 0; pair < pairCount; pair++) {
      originPositions.set(data.hub.lonLat, pair * 2);
    }
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGeodesicPairs',
          datasets: [data.id],
          retain: ['distances', 'initialBearings', 'midpoints', 'converged'],
          keySpaces: [{id: 'pair', label: 'route from ORD', size: pairCount, indexSpace: '1d'}],
          choices: {
            spatialContext: 'longitude-latitude, ellipsoidal WGS84, meters',
            maximumIterations: 16
          },
          metadata: geographicMetadata('geodesic'),
          async build(graph, uploads) {
            // The pinned datum of the docs card is the ORD to LHR route.
            assert(data.iata[15] === 'LHR');
            const origins = uploads.view('origins', originPositions, 'float32x2');
            const targets = uploads.view('targets', data.destinationsLonLat, 'float32x2');
            const hub = uploads.view('hub', Float32Array.from(data.hub.lonLat), 'float32x2');
            const output = await runExample(graph, {origins, targets});
            return {
              hub: {
                view: hub,
                role: 'context',
                geometry: geographicGeometry('points'),
                meaning: 'context'
              },
              targets: {
                view: targets,
                role: 'input',
                keySpace: 'pair',
                geometry: geographicGeometry('points'),
                meaning: '40 destination airports of the busiest routes touching ORD'
              },
              distances: {
                view: output.distances,
                role: 'intermediate',
                keySpace: 'pair',
                meaning:
                  'WGS84 geodesic distance in m; capped Vincenty inverse (16 iterations), spherical fallback when not converged'
              },
              initialBearings: {
                view: output.initialBearings,
                role: 'intermediate',
                keySpace: 'pair',
                meaning: 'initial bearing at ORD in degrees clockwise from north, in (-180, 180]'
              },
              converged: {
                view: output.converged,
                role: 'intermediate',
                keySpace: 'pair',
                meaning: '1 when Vincenty converged; 0 for near-antipodal pairs (sphere fallback)'
              },
              midpoints: {
                view: output.midpoints,
                role: 'output',
                keySpace: 'pair',
                geometry: geographicGeometry('points'),
                meaning:
                  'spherical great-circle midpoint of each route (not the WGS84 geodesic midpoint)'
              }
            };
          }
        })
    };
  }
});
