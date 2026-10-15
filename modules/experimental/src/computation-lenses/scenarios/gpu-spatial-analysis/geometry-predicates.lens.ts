// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.geometry-predicates',
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeometryPredicates',
  title: 'Geometry predicates of Chicago North Side parks',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.parks,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, parks] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./geometry-predicates.example'),
        import('../../data/s2-chicago-north-parks.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.geometry-predicates',
          element: 'GPUGeometryPredicates',
          datasets: ['s2-chicago-north-parks'],
          retain: ['isSimple', 'numCoordinates', 'numInteriorRings', 'overflow'],
          keySpaces,
          choices: {
            columns: 'isSimple, numCoordinates, numInteriorRings',
            intersectionCapacity: 4096
          },
          async build(graph, uploads) {
            const parkGeometry = uploadS2JoinPolygons(uploads, frame, 'parks', parks.polygons);
            keySpaces.push(...parkGeometry.keySpaces, {
              id: 'status',
              label: 'run status',
              size: 1,
              indexSpace: '1d'
            });
            const result = await runExample(graph, {
              parks: parkGeometry.geometry,
              spatialContext: frame.spatialContext
            });
            const captures: Record<string, S2CaptureSpec> = {...parkGeometry.captures};
            captures['resultParks'] = {
              view: parkGeometry.display,
              role: 'output',
              keySpace: 'parks',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'parksDisplayOffsets',
                valuesCapture: 'numCoordinates',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['numCoordinates'] = {
              view: result.numCoordinates,
              role: 'output',
              keySpace: 'parks',
              meaning: 'stored vertex count per park'
            };
            captures['isSimple'] = {view: result.isSimple, role: 'output', keySpace: 'parks'};
            captures['numInteriorRings'] = {
              view: result.numInteriorRings,
              role: 'output',
              keySpace: 'parks'
            };
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            return captures;
          }
        });
      }
    };
  }
});
