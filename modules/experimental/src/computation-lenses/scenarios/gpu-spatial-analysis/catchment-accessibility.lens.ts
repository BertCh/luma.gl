// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.catchment-accessibility',
  entry: 'gpu-spatial-analysis',
  element: 'GPUCatchmentAccessibility',
  title: 'Library accessibility of Chicago North Side tracts, 2 km straight-line catchments',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
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
      facilities
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./catchment-accessibility.example'),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default),
      import('../../data/s2-chicago-north-facilities.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const libraryCategory = facilities.categories.indexOf('library');
        const libraryRows = facilities.category.flatMap((category, index) =>
          category === libraryCategory ? [index] : []
        );
        const libraryPositions = projectS2Positions(
          frame,
          libraryRows.flatMap(index => facilities.positions.slice(index * 2, index * 2 + 2))
        );
        const tractVertices = projectS2Positions(frame, tracts.polygons.vertices);
        const tractCentroids = computeS2RingCentroids(
          tractVertices.positions,
          tracts.polygons.ringOffsets
        );
        const bounds: [number, number, number, number] = [
          Math.min(libraryPositions.extent[0], tractVertices.extent[0]) - 500,
          Math.min(libraryPositions.extent[1], tractVertices.extent[1]) - 500,
          Math.max(libraryPositions.extent[2], tractVertices.extent[2]) + 500,
          Math.max(libraryPositions.extent[3], tractVertices.extent[3]) + 500
        ];
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.catchment-accessibility',
          element: 'GPUCatchmentAccessibility',
          datasets: ['s2-chicago-north-facilities', 's2-chicago-north-tracts'],
          retain: [
            'accessibility',
            'ratios',
            'reachableFacilities',
            'facilityOverflow',
            'tractOverflow'
          ],
          keySpaces,
          choices: {method: '2sfca', radius: 2000, weightKind: 'binary'},
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(
              ...tractGeometry.keySpaces,
              {id: 'library', label: 'library', size: libraryRows.length, indexSpace: '1d'},
              {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
            );
            const captures: Record<string, S2CaptureSpec> = {...tractGeometry.captures};
            captures['tracts'].meaning = 'context';
            const libraries = uploads.view('libraries', libraryPositions.positions, 'float32x2');
            const supply = uploads.view(
              'supply',
              new Float32Array(libraryRows.length).fill(1),
              'float32'
            );
            const tractCentroidView = uploads.view('tractCentroids', tractCentroids, 'float32x2');
            const population = uploads.view(
              'population',
              Float32Array.from(tracts.population),
              'float32'
            );
            captures['libraries'] = {
              view: libraries,
              role: 'input',
              keySpace: 'library',
              geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
            };
            captures['supply'] = {view: supply, role: 'input', keySpace: 'library'};
            captures['tractCentroids'] = {
              view: tractCentroidView,
              role: 'input',
              keySpace: 'tracts',
              geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
            };
            captures['population'] = {view: population, role: 'input', keySpace: 'tracts'};
            const result = await runExample(graph, {
              libraries,
              supply,
              tractCentroids: tractCentroidView,
              population,
              spatialContext: frame.spatialContext,
              bounds
            });
            uploads.own(result.searchParameters);
            captures['resultTracts'] = {
              view: tractGeometry.display,
              role: 'output',
              keySpace: 'tracts',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'tractsDisplayOffsets',
                valuesCapture: 'accessibility',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['accessibility'] = {
              view: result.accessibility,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['ratios'] = {view: result.ratios, role: 'output', keySpace: 'library'};
            captures['reachableFacilities'] = {
              view: result.reachableFacilities,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['facilityOverflow'] = {
              view: result.facilityOverflow,
              role: 'output',
              keySpace: 'status'
            };
            captures['tractOverflow'] = {
              view: result.tractOverflow,
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
