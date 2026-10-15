// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.great-circle-arcs';
const SEGMENTS_PER_ARC = 32;

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUGreatCircleArcs',
  title: 'Great-circle arcs from Chicago O’Hare',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
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
        import('./great-circle-arcs.example'),
        import('./s3-lens-data').then(module => module.loadS3OpenFlightsHub())
      ]);
    const {getGPUGreatCircleArcsParameterValues} = await import(
      '@luma.gl/experimental/gpu-spatial-analysis'
    );
    const pairCount = data.destinationCount;
    // Every route starts at the hub: the source column repeats the ORD position.
    const sourcePositions = new Float32Array(pairCount * 2);
    for (let pair = 0; pair < pairCount; pair++) {
      sourcePositions.set(data.hub.lonLat, pair * 2);
    }
    // Zero maximum length: only minimumSegments decides, so every arc has 33 vertices.
    const vertexCapacity = pairCount * (SEGMENTS_PER_ARC + 1);
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUGreatCircleArcs',
          datasets: [data.id],
          retain: ['arcs', 'arcOffsets', 'arcSources', 'count', 'overflow'],
          keySpaces: [{id: 'route', label: 'route from ORD', size: pairCount, indexSpace: '1d'}],
          choices: {
            spatialContext: 'longitude-latitude, great-circle sphere, meters',
            maximumSegmentLength: 0,
            minimumSegments: SEGMENTS_PER_ARC,
            maximumSegmentCount: 256
          },
          metadata: geographicMetadata('geodesic'),
          async build(graph, uploads) {
            // The pinned datum of the docs card is the ORD to JFK route.
            assert(data.iata[11] === 'JFK');
            const sources = uploads.view('sources', sourcePositions, 'float32x2');
            const targets = uploads.view('targets', data.destinationsLonLat, 'float32x2');
            const hub = uploads.view('hub', Float32Array.from(data.hub.lonLat), 'float32x2');
            const parameters = uploads.view(
              'parameters',
              getGPUGreatCircleArcsParameterValues({
                maximumSegmentLength: 0,
                minimumSegments: SEGMENTS_PER_ARC
              }),
              'float32'
            );
            const output = await runExample(graph, {sources, targets, parameters, vertexCapacity});
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
                keySpace: 'route',
                geometry: geographicGeometry('points'),
                meaning: '40 destination airports of the busiest routes touching ORD'
              },
              arcOffsets: {view: output.lineOffsets, role: 'intermediate'},
              arcSources: {
                view: output.sourceLineIndices,
                role: 'intermediate',
                keySpace: 'route',
                meaning: 'source route index (zero-based) of each output arc'
              },
              count: {view: output.status.count, role: 'intermediate'},
              overflow: {view: output.status.overflow, role: 'intermediate'},
              arcs: {
                view: output.arcs,
                role: 'output',
                geometry: geographicGeometry('lines', {offsetsCapture: 'arcOffsets'}),
                meaning:
                  'GPU-emitted spherical great-circle arcs (mean-radius sphere); GPU offsets exclude unused capacity'
              }
            };
          }
        })
    };
  }
});
