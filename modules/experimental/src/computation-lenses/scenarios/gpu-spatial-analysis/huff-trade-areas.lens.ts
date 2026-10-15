// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.huff-trade-areas',
  entry: 'gpu-spatial-analysis',
  element: 'GPUHuffTradeAreas',
  title: 'Park trade areas of Chicago North Side tracts, Huff model',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.parks,
  load: async () => {
    const [
      {
        runS2Lens,
        createS2ChicagoFrame,
        projectS2Positions,
        computeS2RingCentroids,
        uploadS2JoinPolygons
      },
      {runExample},
      tracts,
      parks
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./huff-trade-areas.example'),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default),
      import('../../data/s2-chicago-north-parks.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const tractVertices = projectS2Positions(frame, tracts.polygons.vertices);
        const parkVertices = projectS2Positions(frame, parks.polygons.vertices);
        const tractCentroids = computeS2RingCentroids(
          tractVertices.positions,
          tracts.polygons.ringOffsets
        );
        const parkCentroids = computeS2RingCentroids(
          parkVertices.positions,
          parks.polygons.ringOffsets
        );
        const bounds: [number, number, number, number] = [
          Math.min(parkVertices.extent[0], tractVertices.extent[0]) - 500,
          Math.min(parkVertices.extent[1], tractVertices.extent[1]) - 500,
          Math.max(parkVertices.extent[2], tractVertices.extent[2]) + 500,
          Math.max(parkVertices.extent[3], tractVertices.extent[3]) + 500
        ];
        const slotCapacity = 8192;
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.huff-trade-areas',
          element: 'GPUHuffTradeAreas',
          datasets: ['s2-chicago-north-parks', 's2-chicago-north-tracts'],
          retain: [
            'tradeArea',
            'tradeAreaProbability',
            'probabilities',
            'expectedDemand',
            'tractOffsets',
            'tractNeighbors',
            'tractOverflow',
            'parkOverflow'
          ],
          keySpaces,
          choices: {distanceDecay: 'inverse-distance', power: 2, radius: 2500, exponent: 1},
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            const parkGeometry = uploadS2JoinPolygons(uploads, frame, 'parks', parks.polygons);
            keySpaces.push(
              ...tractGeometry.keySpaces,
              ...parkGeometry.keySpaces,
              {
                id: 'tract-row-offset',
                label: 'tract row offset',
                size: tracts.population.length + 1,
                indexSpace: '1d'
              },
              {id: 'slot', label: 'tract-park slot', size: slotCapacity, indexSpace: '1d'},
              {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
            );
            const captures: Record<string, S2CaptureSpec> = {
              ...tractGeometry.captures,
              ...parkGeometry.captures
            };
            captures['tracts'].meaning = 'context';
            const parkView = uploads.view('parkCentroids', parkCentroids, 'float32x2');
            const attractiveness = uploads.view(
              'attractiveness',
              Float32Array.from(parks.areaKm2),
              'float32'
            );
            const tractCentroidView = uploads.view('tractCentroids', tractCentroids, 'float32x2');
            const population = uploads.view(
              'population',
              Float32Array.from(tracts.population),
              'float32'
            );
            captures['attractiveness'] = {view: attractiveness, role: 'input', keySpace: 'parks'};
            captures['population'] = {view: population, role: 'input', keySpace: 'tracts'};
            const result = await runExample(graph, {
              parks: parkView,
              attractiveness,
              tractCentroids: tractCentroidView,
              population,
              spatialContext: frame.spatialContext,
              bounds
            });
            uploads.own(result.searchParameters);
            uploads.own(result.huffParameters);
            captures['resultTracts'] = {
              view: tractGeometry.display,
              role: 'output',
              keySpace: 'tracts',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'tractsDisplayOffsets',
                valuesCapture: 'tradeAreaProbability',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['tradeAreaProbability'] = {
              view: result.tradeAreaProbability,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['tradeArea'] = {view: result.tradeArea, role: 'output', keySpace: 'tracts'};
            captures['expectedDemand'] = {
              view: result.expectedDemand,
              role: 'output',
              keySpace: 'parks'
            };
            captures['tractOffsets'] = {
              view: result.tractOffsets,
              role: 'intermediate',
              keySpace: 'tract-row-offset'
            };
            captures['tractNeighbors'] = {
              view: result.tractNeighbors,
              role: 'intermediate',
              keySpace: 'slot'
            };
            captures['probabilities'] = {
              view: result.probabilities,
              role: 'output',
              keySpace: 'slot'
            };
            captures['tractOverflow'] = {
              view: result.tractOverflow,
              role: 'output',
              keySpace: 'status'
            };
            captures['parkOverflow'] = {
              view: result.parkOverflow,
              role: 'output',
              keySpace: 'status'
            };
            return captures;
          }
        });
      }
    };
  }
});
